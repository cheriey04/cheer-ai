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
import math
import ssl
import pickle
import urllib.request

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
               'knee_curvature', 'arm_misalignment']
STAT_NAMES = ['mean', 'std', 'min', 'max', 'range', 'rate_of_change', 'skewness', 'auc']
MOVE_NAMES = ['standing', 'high_v', 'liberty', 't_jump', 'tuck_jump']


def _vec(a, b): return np.array([b.x - a.x, b.y - a.y, b.z - a.z])
def _norm(v): return float(np.linalg.norm(v))

def _angle_between(v1, v2):
    dot = float(np.dot(v1, v2)); n1, n2 = _norm(v1), _norm(v2)
    if n1 < 1e-9 or n2 < 1e-9: return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))

def _tilt_from_horizontal(vec):
    raw = _angle_between(vec, np.array([1.0, 0.0, 0.0]))
    return min(raw, 180.0 - raw)

# ---- 5 parameter calculators ----
def calc_shoulder_tilt(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]))
def calc_pelvic_tilt(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_HIP], lm[RIGHT_HIP]))
def calc_trunk_shift(lm):
    sm = np.array([(lm[LEFT_SHOULDER].x+lm[RIGHT_SHOULDER].x)/2,
                   (lm[LEFT_SHOULDER].y+lm[RIGHT_SHOULDER].y)/2,
                   (lm[LEFT_SHOULDER].z+lm[RIGHT_SHOULDER].z)/2])
    hm = np.array([(lm[LEFT_HIP].x+lm[RIGHT_HIP].x)/2,
                   (lm[LEFT_HIP].y+lm[RIGHT_HIP].y)/2,
                   (lm[LEFT_HIP].z+lm[RIGHT_HIP].z)/2])
    return _angle_between(sm - hm, np.array([0.0, 1.0, 0.0]))
def calc_knee_curvature(lm):
    la = _angle_between(_vec(lm[LEFT_HIP],lm[LEFT_KNEE]),_vec(lm[LEFT_KNEE],lm[LEFT_ANKLE]))
    ra = _angle_between(_vec(lm[RIGHT_HIP],lm[RIGHT_KNEE]),_vec(lm[RIGHT_KNEE],lm[RIGHT_ANKLE]))
    return (la+ra)/2.0
def calc_arm_misalignment(lm):
    wa = _tilt_from_horizontal(_vec(lm[LEFT_WRIST], lm[RIGHT_WRIST]))
    ea = _tilt_from_horizontal(_vec(lm[LEFT_ELBOW], lm[RIGHT_ELBOW]))
    return (wa+ea)/2.0

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
        try:
            param_vals['shoulder_tilt'].append(calc_shoulder_tilt(lm))
            param_vals['pelvic_tilt'].append(calc_pelvic_tilt(lm))
            param_vals['trunk_shift'].append(calc_trunk_shift(lm))
            param_vals['knee_curvature'].append(calc_knee_curvature(lm))
            param_vals['arm_misalignment'].append(calc_arm_misalignment(lm))
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
    if move_pred in iso_forests:
        try:
            outlier = iso_forests[move_pred].predict(X)[0] == -1
        except ValueError:
            # Isolation Forest has wrong number of features (likely old model)
            print(f"  ⚠️  Skipping isolation_forest_{move_pred}.pkl —"
                  f" incompatible feature count")
            outlier = False

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
    print(f"  ISOLATION FOREST:   {'⚠️  OUTLIER' if outlier else '✓  normal'}")
    print("\n  Move probabilities:")
    for i, n in enumerate(move_clf.classes_):
        b = '█'*int(mpb[i]*40)+'░'*(40-int(mpb[i]*40))
        print(f"    {n:12s}  {mpb[i]:.1%}  {b}{' ←' if i==mi else ''}")
    print("\n  Quality probabilities:")
    for i, n in enumerate(q_classes):
        b = '█'*int(qpb[i]*40)+'░'*(40-int(qpb[i]*40))
        print(f"    {n:12s}  {qpb[i]:.1%}  {b}{' ←' if i==qi else ''}")
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
