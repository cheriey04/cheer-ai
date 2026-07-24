"""
batch_predict.py

Run the trained models on all videos in test_videos/ and output
a CSV with predictions, confidences, anomaly scores, and top deviations.

Usage:
  python scripts/batch_predict.py
"""

import os
import sys
import json
import math
import ssl
import pickle
import urllib.request
import csv
from collections import namedtuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

# ---------------------------------------------------------------------------
# Constants & math helpers (same as process_videos.py / predict_video.py)
# ---------------------------------------------------------------------------

LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12; LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_WRIST = 15; RIGHT_WRIST = 16; LEFT_HIP = 23; RIGHT_HIP = 24
LEFT_KNEE = 25; RIGHT_KNEE = 26; LEFT_ANKLE = 27; RIGHT_ANKLE = 28

POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task'

PARAM_NAMES = ['shoulder_tilt', 'pelvic_tilt', 'trunk_shift',
               'left_knee_curvature', 'right_knee_curvature',
               'knee_curvature', 'arm_misalignment',
               'distance_between_feet',
               'left_arm_curvature', 'right_arm_curvature',
               'left_arm_angle', 'right_arm_angle']
STAT_NAMES = ['mean', 'std', 'min', 'max', 'range', 'rate_of_change', 'skewness', 'auc']
MOVE_NAMES = ['standing', 'high_v', 'liberty', 't_jump', 'tuck_jump']

PixelLandmark = namedtuple('PixelLandmark', ['x', 'y'])


def _vec(a, b): return np.array([b.x - a.x, b.y - a.y])
def _norm(v): return float(np.linalg.norm(v))

def _angle_between(v1, v2):
    dot = float(np.dot(v1, v2)); n1, n2 = _norm(v1), _norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))

def _tilt_from_horizontal(vec):
    raw = _angle_between(vec, np.array([1.0, 0.0]))
    return min(raw, 180.0 - raw)

# ---- Parameter calculators ----
def calc_shoulder_tilt(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]))
def calc_pelvic_tilt(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_HIP], lm[RIGHT_HIP]))
def calc_trunk_shift(lm):
    sm = np.array([(lm[LEFT_SHOULDER].x+lm[RIGHT_SHOULDER].x)/2,
                   (lm[LEFT_SHOULDER].y+lm[RIGHT_SHOULDER].y)/2])
    hm = np.array([(lm[LEFT_HIP].x+lm[RIGHT_HIP].x)/2,
                   (lm[LEFT_HIP].y+lm[RIGHT_HIP].y)/2])
    return _angle_between(sm - hm, np.array([0.0, -1.0]))
def calc_left_knee_curvature(lm):
    bend = _angle_between(_vec(lm[RIGHT_HIP],lm[RIGHT_KNEE]),_vec(lm[RIGHT_KNEE],lm[RIGHT_ANKLE]))
    return 180.0 - bend
def calc_right_knee_curvature(lm):
    bend = _angle_between(_vec(lm[LEFT_HIP],lm[LEFT_KNEE]),_vec(lm[LEFT_KNEE],lm[LEFT_ANKLE]))
    return 180.0 - bend
def calc_knee_curvature(lm):
    return (calc_left_knee_curvature(lm) + calc_right_knee_curvature(lm)) / 2.0
def calc_distance_between_feet(lm):
    return _norm(_vec(lm[LEFT_ANKLE],lm[RIGHT_ANKLE]))
def calc_left_arm_curvature(lm):
    bend = _angle_between(_vec(lm[RIGHT_WRIST],lm[RIGHT_ELBOW]),_vec(lm[RIGHT_ELBOW],lm[RIGHT_SHOULDER]))
    return 180.0 - bend
def calc_right_arm_curvature(lm):
    bend = _angle_between(_vec(lm[LEFT_WRIST],lm[LEFT_ELBOW]),_vec(lm[LEFT_ELBOW],lm[LEFT_SHOULDER]))
    return 180.0 - bend
def calc_left_arm_angle(lm):
    return _tilt_from_horizontal(_vec(lm[RIGHT_SHOULDER],lm[RIGHT_ELBOW]))
def calc_right_arm_angle(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER],lm[LEFT_ELBOW]))
def calc_arm_misalignment(lm):
    wa = _tilt_from_horizontal(_vec(lm[LEFT_WRIST], lm[RIGHT_WRIST]))
    ea = _tilt_from_horizontal(_vec(lm[LEFT_ELBOW], lm[RIGHT_ELBOW]))
    return (wa+ea)/2.0

# ---- Aggregated stats ----
def compute_aggregated_stats(values):
    if len(values) < 2: return {s: float('nan') for s in STAT_NAMES}
    v = np.asarray(values, dtype=np.float64)
    s = {'mean': float(np.mean(v)), 'std': float(np.std(v)),
         'min': float(np.min(v)), 'max': float(np.max(v)),
         'range': float(np.max(v)-np.min(v))}
    d = np.abs(np.diff(v))
    s['rate_of_change'] = float(np.mean(d)) if len(d) > 0 else 0.0
    s['skewness'] = float(pd.Series(v).skew())
    s['auc'] = float(np.trapezoid(v))
    return s

# ---- MediaPipe helpers ----
def download_model_if_needed(model_path):
    if os.path.exists(model_path): return
    print("Downloading PoseLandmarker model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ctx = ssl._create_unverified_context()
    try: urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    except Exception:
        with urllib.request.urlopen(POSE_MODEL_URL, context=ctx) as r, \
             open(model_path, 'wb') as f: f.write(r.read())

def create_landmarker(model_path):
    o = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    return vision.PoseLandmarker.create_from_options(o)


# ===========================================================================
# Predict one video — returns dict of results
# ===========================================================================

def predict_one_video(video_path, move_clf, quality_clfs, iso_forests,
                      normal_avgs, model_path):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    lmkr = create_landmarker(model_path)  # fresh landmarker per video

    param_vals = {p: [] for p in PARAM_NAMES}
    n_proc, n_skip = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        res = lmkr.detect_for_video(mp_img, ts)
        if not res.pose_landmarks: n_skip += 1; continue
        lm = res.pose_landmarks[0]
        h, w = frame.shape[:2]
        pixel_lm = [PixelLandmark(lm[i].x * w, lm[i].y * h) for i in range(33)]
        try:
            param_vals['shoulder_tilt'].append(calc_shoulder_tilt(pixel_lm))
            param_vals['pelvic_tilt'].append(calc_pelvic_tilt(pixel_lm))
            param_vals['trunk_shift'].append(calc_trunk_shift(pixel_lm))
            param_vals['left_knee_curvature'].append(calc_left_knee_curvature(pixel_lm))
            param_vals['right_knee_curvature'].append(calc_right_knee_curvature(pixel_lm))
            param_vals['knee_curvature'].append(calc_knee_curvature(pixel_lm))
            param_vals['arm_misalignment'].append(calc_arm_misalignment(pixel_lm))
            param_vals['distance_between_feet'].append(calc_distance_between_feet(pixel_lm))
            param_vals['left_arm_curvature'].append(calc_left_arm_curvature(pixel_lm))
            param_vals['right_arm_curvature'].append(calc_right_arm_curvature(pixel_lm))
            param_vals['left_arm_angle'].append(calc_left_arm_angle(pixel_lm))
            param_vals['right_arm_angle'].append(calc_right_arm_angle(pixel_lm))
        except Exception: n_skip += 1; continue
        n_proc += 1

    cap.release()
    lmkr.close()

    if n_proc == 0:
        return {
            'filename': video_name,
            'predicted_move': 'ERROR',
            'predicted_move_confidence': 0.0,
            'predicted_quality': 'ERROR',
            'predicted_quality_confidence': 0.0,
            'anomaly_score': 1.0,
            'top_deviations': 'No pose detected',
            'actual_move': '',
            'actual_quality': '',
        }

    # ---- Compute 96 aggregated features ------------------------------
    feat = {}
    for p in PARAM_NAMES:
        s = compute_aggregated_stats(np.array(param_vals[p]))
        for sn in STAT_NAMES: feat[f'{p}_{sn}'] = s[sn]
    X = np.array([list(feat.values())])

    # ---- Predict -----------------------------------------------------
    move_pred = move_clf.predict(X)[0]
    mpb = move_clf.predict_proba(X)[0]
    move_conf = float(max(mpb))

    quality_pred = 'Unknown'
    qpb = np.array([0.5, 0.5])
    if move_pred in quality_clfs:
        quality_pred = quality_clfs[move_pred].predict(X)[0]
        qpb = quality_clfs[move_pred].predict_proba(X)[0]
    quality_conf = float(max(qpb))

    anomaly_score = 0.0
    if move_pred in iso_forests:
        try:
            df_val = iso_forests[move_pred].decision_function(X)[0]
            anomaly_score = max(0.0, min(1.0, 0.5 - float(df_val)))
        except ValueError:
            anomaly_score = 0.0

    # ---- Top 3 deviations from Normal --------------------------------
    top_dev_str = ''
    if move_pred in normal_avgs:
        norms = normal_avgs[move_pred]
        deviations = []
        for fname, fval in feat.items():
            # Skip stats with inflated z-scores (tiny stds across videos)
            if fname.endswith(('_auc', '_rate_of_change', '_skewness')):
                continue
            if fname in norms:
                n_mean = norms[fname]
                std_key = fname.replace('_mean', '_std').replace('_min', '_std') \
                               .replace('_max', '_std').replace('_range', '_std') \
                               .replace('_skewness', '_std')
                n_std = norms.get(std_key, 1.0)
                if n_std > 0:
                    z = abs(fval - n_mean) / n_std
                else:
                    z = 0.0
                diff = fval - n_mean
                deviations.append((fname, fval, n_mean, n_std, z, diff))
        deviations.sort(key=lambda x: x[4], reverse=True)

        parts = []
        for fname, fval, n_mean, n_std, z, diff in deviations[:3]:
            direction = 'above' if diff >= 0 else 'below'
            parts.append(
                f"{fname}: {fval:.2f} ({direction} normal {n_mean:.2f}±{n_std:.2f}, z={z:.1f}σ)"
            )
        top_dev_str = ' | '.join(parts)

    return {
        'filename': video_name,
        'predicted_move': move_pred,
        'predicted_move_confidence': round(move_conf, 4),
        'predicted_quality': quality_pred,
        'predicted_quality_confidence': round(quality_conf, 4),
        'anomaly_score': round(anomaly_score, 4),
        'top_deviations': top_dev_str,
        'actual_move': '',
        'actual_quality': '',
    }


# ===========================================================================
# Batch main
# ===========================================================================

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'test_videos')
    model_path = os.path.join(base_dir, 'data', 'models', 'pose_landmarker.task')

    # Find all .mp4 files in test_videos/
    video_files = sorted([
        f for f in os.listdir(test_dir)
        if f.endswith('.mp4')
    ])
    if not video_files:
        print(f"No .mp4 files found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(video_files)} test videos\n")

    # ---- Download model & load classifiers (once) --------------------
    download_model_if_needed(model_path)

    def _load(name):
        with open(os.path.join(base_dir, 'models', name), 'rb') as f:
            return pickle.load(f)

    print("Loading models...")
    move_clf = _load('move_classifier.pkl')
    iso_forests = {}
    quality_clfs = {}
    for m in MOVE_NAMES:
        fn = f'isolation_forest_{m}.pkl'
        if os.path.exists(os.path.join(base_dir, 'models', fn)):
            iso_forests[m] = _load(fn)
        qfn = f'quality_{m}.pkl'
        if os.path.exists(os.path.join(base_dir, 'models', qfn)):
            quality_clfs[m] = _load(qfn)
    print(f"  ✓ move_classifier.pkl")
    print(f"  ✓ {len(quality_clfs)} per-move quality classifiers")
    print(f"  ✓ {len(iso_forests)} isolation forests")

    normal_avgs = {}
    navg_path = os.path.join(base_dir, 'models', 'normal_averages.json')
    if os.path.exists(navg_path):
        with open(navg_path, 'r') as f:
            normal_avgs = json.load(f)
        print(f"  ✓ normal_averages.json\n")

    # ---- Process each video ------------------------------------------
    results = []
    for i, vf in enumerate(video_files):
        vpath = os.path.join(test_dir, vf)
        print(f"[{i+1}/{len(video_files)}] {vf} ...", end=' ', flush=True)
        result = predict_one_video(
            vpath, move_clf, quality_clfs, iso_forests, normal_avgs, model_path
        )
        results.append(result)
        print(f"→ {result['predicted_move']} | "
              f"{result['predicted_quality']} | "
              f"anomaly={result['anomaly_score']:.2f}")

    # ---- Write CSV ---------------------------------------------------
    csv_path = os.path.join(base_dir, 'test_videos_predictions.csv')
    fieldnames = [
        'filename', 'predicted_move', 'predicted_move_confidence',
        'predicted_quality', 'predicted_quality_confidence',
        'anomaly_score', 'top_deviations',
        'actual_move', 'actual_quality',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results written to {csv_path}")
    print(f"   {len(results)} videos processed")


if __name__ == '__main__':
    main()
