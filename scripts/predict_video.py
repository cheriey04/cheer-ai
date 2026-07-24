"""
predict_video.py

Run the trained aggregated models on a single video — one verdict per video:
  1. Predict the move name (which cheer move is being performed)
  2. Predict the quality (Normal vs Bad)
  3. Flag outlier via Isolation Forest for the predicted move

Computes 8 stats per parameter across ALL frames, then runs the models
on the 40 aggregated features (same format as training_data_aggregated.csv).

Usage:
  python scripts/predict_video.py path/to/any_cheer_video.mp4
"""

import os
import sys
import json
import math
import ssl
import pickle
import urllib.request
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
# Constants & math helpers (same as process_videos.py)
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

# ---- 5 parameter calculators ----
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
    """Athlete's left leg = MediaPipe RIGHT landmarks (facing camera)."""
    bend = _angle_between(_vec(lm[RIGHT_HIP],lm[RIGHT_KNEE]),_vec(lm[RIGHT_KNEE],lm[RIGHT_ANKLE]))
    return 180.0 - bend
def calc_right_knee_curvature(lm):
    """Athlete's right leg = MediaPipe LEFT landmarks (facing camera)."""
    bend = _angle_between(_vec(lm[LEFT_HIP],lm[LEFT_KNEE]),_vec(lm[LEFT_KNEE],lm[LEFT_ANKLE]))
    return 180.0 - bend
def calc_distance_between_feet(lm):
    """Magnitude of vector between left and right ankles."""
    return _norm(_vec(lm[LEFT_ANKLE],lm[RIGHT_ANKLE]))
def calc_left_arm_curvature(lm):
    """Athlete's left arm = MediaPipe RIGHT. Angle at elbow."""
    bend = _angle_between(_vec(lm[RIGHT_WRIST],lm[RIGHT_ELBOW]),_vec(lm[RIGHT_ELBOW],lm[RIGHT_SHOULDER]))
    return 180.0 - bend
def calc_right_arm_curvature(lm):
    """Athlete's right arm = MediaPipe LEFT. Angle at elbow."""
    bend = _angle_between(_vec(lm[LEFT_WRIST],lm[LEFT_ELBOW]),_vec(lm[LEFT_ELBOW],lm[LEFT_SHOULDER]))
    return 180.0 - bend
def calc_left_arm_angle(lm):
    """Athlete's left upper arm vs horizontal (0-90 deg)."""
    return _tilt_from_horizontal(_vec(lm[RIGHT_SHOULDER],lm[RIGHT_ELBOW]))
def calc_right_arm_angle(lm):
    """Athlete's right upper arm vs horizontal (0-90 deg)."""
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER],lm[LEFT_ELBOW]))
def calc_arm_misalignment(lm):
    wa = _tilt_from_horizontal(_vec(lm[LEFT_WRIST], lm[RIGHT_WRIST]))
    ea = _tilt_from_horizontal(_vec(lm[LEFT_ELBOW], lm[RIGHT_ELBOW]))
    return (wa+ea)/2.0

def calc_knee_curvature(lm):
    """Combined Knee Curvature — average of left and right (0–180 deg)."""
    return (calc_left_knee_curvature(lm) + calc_right_knee_curvature(lm)) / 2.0

# ---- Aggregated stats (same as process_videos.py) ----
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
# Main
# ===========================================================================

def predict_video(video_path):
    video_path = os.path.abspath(video_path)
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}"); sys.exit(1)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'data', 'models', 'pose_landmarker.task')
    download_model_if_needed(model_path)

    # ---- Load models -------------------------------------------------
    def _load(name):
        with open(os.path.join(base_dir, 'models', name), 'rb') as f:
            return pickle.load(f)
    print("Loading models...")
    move_clf = _load('move_classifier.pkl')
    iso_forests = {}
    quality_clfs = {}  # per-move quality classifiers
    for m in MOVE_NAMES:
        fn = f'isolation_forest_{m}.pkl'
        if os.path.exists(os.path.join(base_dir, 'models', fn)):
            iso_forests[m] = _load(fn)
            print(f"  ✓ {fn}")
        qfn = f'quality_{m}.pkl'
        if os.path.exists(os.path.join(base_dir, 'models', qfn)):
            quality_clfs[m] = _load(qfn)
            print(f"  ✓ {qfn}")
    print(f"  ✓ move_classifier.pkl\n")

    # ---- Load normal averages for deviation analysis -----------------
    normal_avgs = {}
    navg_path = os.path.join(base_dir, 'models', 'normal_averages.json')
    if os.path.exists(navg_path):
        with open(navg_path, 'r') as f:
            normal_avgs = json.load(f)
        print(f"  ✓ Loaded normal_averages.json ({len(normal_avgs)} moves)\n")

    # ---- Process video — collect per-frame params --------------------
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_name}  ({total} frames)")

    lmkr = create_landmarker(model_path)
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
        if n_proc % 50 == 0: print(f"  Processed {n_proc} frames...", flush=True)

    cap.release(); lmkr.close()

    if n_proc == 0:
        print("\nERROR: No frames with a detected pose."); sys.exit(1)
    print(f"  Done — {n_proc} frames processed"
          + (f", {n_skip} skipped" if n_skip else "") + "\n")

    # ---- Compute 40 aggregated features ------------------------------
    feat = {}
    for p in PARAM_NAMES:
        s = compute_aggregated_stats(np.array(param_vals[p]))
        for sn in STAT_NAMES: feat[f'{p}_{sn}'] = s[sn]
    X = np.array([list(feat.values())])

    # ---- Predict -----------------------------------------------------
    move_pred = move_clf.predict(X)[0]; mpb = move_clf.predict_proba(X)[0]

    # Use the per-move quality classifier for the predicted move
    quality_pred = 'Unknown'
    qpb = np.array([0.5, 0.5])  # default: uncertain
    if move_pred in quality_clfs:
        quality_pred = quality_clfs[move_pred].predict(X)[0]
        qpb = quality_clfs[move_pred].predict_proba(X)[0]
    else:
        print(f"  ⚠️  No per-move quality classifier for {move_pred}")
    outlier = False
    anomaly_score = 0.0  # 0 = completely normal, 1 = completely anomalous
    if move_pred in iso_forests:
        try:
            outlier = iso_forests[move_pred].predict(X)[0] == -1
            # decision_function: positive = inlier, negative = outlier
            df_val = iso_forests[move_pred].decision_function(X)[0]
            # Normalize to [0, 1] where 1 = most anomalous
            anomaly_score = max(0.0, min(1.0, 0.5 - df_val))
        except ValueError:
            # Isolation Forest has wrong number of features (likely old model)
            print(f"  ⚠️  Skipping isolation_forest_{move_pred}.pkl —"
                  f" incompatible feature count")
            outlier = False

    quality_score = int((1.0 - anomaly_score) * 100)  # 0-100, higher = better

    # ---- Output ------------------------------------------------------
    mi = np.argmax(mpb)
    qi = np.argmax(qpb)
    q_classes = quality_clfs[move_pred].classes_ if move_pred in quality_clfs else ['Normal', 'Bad']
    print("=" * 60)
    print(f"RESULTS: {video_name}")
    print("=" * 60)
    print(f"\n  PREDICTED MOVE:     {move_pred}  (confidence: {max(mpb):.1%})")
    print(f"  PREDICTED QUALITY:  {quality_pred}  (confidence: {max(qpb):.1%})"
          + (f"  [using quality_{move_pred}.pkl]" if move_pred in quality_clfs else ""))
    print(f"  QUALITY SCORE:      {quality_score}/100  "
          + ('🟢' if quality_score >= 70 else ('🟡' if quality_score >= 40 else '🔴'))
          + f"  (anomaly: {anomaly_score:.2f})")
    print(f"  ISOLATION FOREST:   {'⚠️  OUTLIER' if outlier else '✓  normal'}")
    print("\n  Move probabilities:")
    for i, n in enumerate(move_clf.classes_):
        b = '█'*int(mpb[i]*40)+'░'*(40-int(mpb[i]*40))
        print(f"    {n:12s}  {mpb[i]:.1%}  {b}{' ←' if i==mi else ''}")
    print("\n  Quality probabilities:")
    for i, n in enumerate(q_classes):
        b = '█'*int(qpb[i]*40)+'░'*(40-int(qpb[i]*40))
        print(f"    {n:12s}  {qpb[i]:.1%}  {b}{' ←' if i==qi else ''}")

    # ---- Top deviations from Normal ----------------------------------
    TOP_N = 5
    if move_pred in normal_avgs:
        norms = normal_avgs[move_pred]
        deviations = []
        for fname, fval in feat.items():
            # Skip stats with inflated z-scores (tiny stds across videos)
            if fname.endswith(('_auc', '_rate_of_change', '_skewness')):
                continue
            if fname in norms:
                n_mean = norms[fname]
                # std is in the same dict as {param}_{stat}; look up the std key
                std_key = fname.replace('_mean', '_std').replace('_min', '_std').replace('_max', '_std') \
                               .replace('_range', '_std').replace('_skewness', '_std')
                n_std = norms.get(std_key, 1.0)
                if n_std > 0:
                    z = abs(fval - n_mean) / n_std
                else:
                    z = 0.0
                deviations.append((fname, fval, n_mean, n_std, z))
        deviations.sort(key=lambda x: x[4], reverse=True)

        print(f"\n  🔴 Top {TOP_N} Deviations from Normal ({move_pred}):")
        print(f"     {'Feature':<38s} {'Value':>8s}  {'Normal':>14s}  {'Z':>6s}")
        print(f"     {'─'*38}  {'─'*8}  {'─'*14}  {'─'*6}")
        for fname, fval, n_mean, n_std, z in deviations[:TOP_N]:
            normal_str = f"{n_mean:.2f} ± {n_std:.2f}"
            marker = '🔴' if z >= 3 else ('🟡' if z >= 2 else '  ')
            print(f"  {marker}  {fname:<36s} {fval:8.2f}  {normal_str:>14s}  {z:5.1f}σ")

    print(f"\n{'='*60}")
    e = "✅" if quality_pred == "Normal" and not outlier else "⚠️"
    ot = " (outlier)" if outlier else ""
    print(f"  {e} Move: {move_pred} | Quality: {quality_pred}{ot}")
    print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_video.py <path_to_video>")
        sys.exit(1)
    predict_video(sys.argv[1])
