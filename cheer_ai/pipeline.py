"""
cheer_ai/pipeline.py — Shared ML pipeline for cheerleading form analysis.

All scripts (process_videos, predict_video, batch_predict, visualize_frames)
import from this single module to avoid code duplication.

Exports:
    Constants:   LANDMARK_IDS, PARAM_NAMES, STAT_NAMES, MOVE_NAMES, POSE_MODEL_URL
    Math:        _vec, _norm, _angle_between, _tilt_from_horizontal
    Params:      calc_shoulder_tilt, calc_pelvic_tilt, calc_trunk_shift, ...
    Aggregation: compute_aggregated_stats
    MediaPipe:   download_model_if_needed, create_landmarker
    Prediction:  load_models, analyze_video
"""

import os
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
# Constants
# ---------------------------------------------------------------------------

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_WRIST = 15; RIGHT_WRIST = 16
LEFT_HIP = 23; RIGHT_HIP = 24
LEFT_KNEE = 25; RIGHT_KNEE = 26
LEFT_ANKLE = 27; RIGHT_ANKLE = 28

POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task'
)

PARAM_NAMES = [
    'shoulder_tilt', 'pelvic_tilt', 'trunk_shift',
    'left_knee_curvature', 'right_knee_curvature', 'knee_curvature',
    'arm_misalignment', 'distance_between_feet',
    'left_arm_curvature', 'right_arm_curvature',
    'left_arm_angle', 'right_arm_angle',
]
STAT_NAMES = [
    'mean', 'std', 'min', 'max', 'range', 'rate_of_change', 'skewness', 'auc',
]
MOVE_NAMES = ['standing', 'high_v', 'liberty', 't_jump', 'tuck_jump']

# Stats excluded from deviation ranking (near-zero variance across Normal videos)
SKIP_DEV_STATS = ('_auc', '_rate_of_change', '_skewness')

# Lightweight container for pixel-space landmark coords (avoids aspect-ratio
# distortion when computing angles from normalized coords).
PixelLandmark = namedtuple('PixelLandmark', ['x', 'y'])


# ===========================================================================
# Vector / Angle Helpers
# ===========================================================================

def _vec(a, b):
    """2D vector from landmark a to landmark b (pixel coordinates)."""
    return np.array([b.x - a.x, b.y - a.y])


def _norm(v):
    """Euclidean norm of a 2D vector."""
    return float(np.linalg.norm(v))


def _angle_between(v1, v2):
    """Angle in degrees between two 2D vectors (0–180)."""
    dot = float(np.dot(v1, v2))
    n1, n2 = _norm(v1), _norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_theta))


def _tilt_from_horizontal(vec):
    """Angle (0–90°) between vec and horizontal.  0 = level."""
    raw = _angle_between(vec, np.array([1.0, 0.0]))
    return min(raw, 180.0 - raw)


# ===========================================================================
# Parameter Calculators (all take pixel-space landmarks, return degrees)
# ===========================================================================

def calc_shoulder_tilt(lm):
    """Shoulder tilt (0–90°).  0 = level shoulders."""
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]))


def calc_pelvic_tilt(lm):
    """Pelvic tilt (0–90°).  0 = level hips."""
    return _tilt_from_horizontal(_vec(lm[LEFT_HIP], lm[RIGHT_HIP]))


def calc_trunk_shift(lm):
    """Trunk shift (0–90°).  0 = perfectly upright.
    Angle between torso vector (hips→shoulders) and upward-vertical (0,-1).
    """
    sm = np.array([(lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2,
                   (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2])
    hm = np.array([(lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2,
                   (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2])
    return _angle_between(sm - hm, np.array([0.0, -1.0]))


def calc_left_knee_curvature(lm):
    """Left knee curvature (0–180°).  180 = fully straight.
    Athlete's left = MediaPipe RIGHT (facing camera).
    """
    bend = _angle_between(
        _vec(lm[RIGHT_HIP], lm[RIGHT_KNEE]),
        _vec(lm[RIGHT_KNEE], lm[RIGHT_ANKLE]),
    )
    return 180.0 - bend


def calc_right_knee_curvature(lm):
    """Right knee curvature (0–180°).  180 = fully straight.
    Athlete's right = MediaPipe LEFT (facing camera).
    """
    bend = _angle_between(
        _vec(lm[LEFT_HIP], lm[LEFT_KNEE]),
        _vec(lm[LEFT_KNEE], lm[LEFT_ANKLE]),
    )
    return 180.0 - bend


def calc_knee_curvature(lm):
    """Combined knee curvature — average of left and right."""
    return (calc_left_knee_curvature(lm) + calc_right_knee_curvature(lm)) / 2.0


def calc_distance_between_feet(lm):
    """Distance between ankles (normalized pixel units, 0+)."""
    return _norm(_vec(lm[LEFT_ANKLE], lm[RIGHT_ANKLE]))


def calc_left_arm_curvature(lm):
    """Left arm curvature (0–180°).  180 = fully straight.
    Athlete's left = MediaPipe RIGHT.
    """
    bend = _angle_between(
        _vec(lm[RIGHT_WRIST], lm[RIGHT_ELBOW]),
        _vec(lm[RIGHT_ELBOW], lm[RIGHT_SHOULDER]),
    )
    return 180.0 - bend


def calc_right_arm_curvature(lm):
    """Right arm curvature (0–180°).  180 = fully straight.
    Athlete's right = MediaPipe LEFT.
    """
    bend = _angle_between(
        _vec(lm[LEFT_WRIST], lm[LEFT_ELBOW]),
        _vec(lm[LEFT_ELBOW], lm[LEFT_SHOULDER]),
    )
    return 180.0 - bend


def calc_left_arm_angle(lm):
    """Left upper arm angle vs horizontal (0–90°).  0 = arm at side."""
    return _tilt_from_horizontal(_vec(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW]))


def calc_right_arm_angle(lm):
    """Right upper arm angle vs horizontal (0–90°).  0 = arm at side."""
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[LEFT_ELBOW]))


def calc_arm_misalignment(lm):
    """Arm misalignment (0–90°).  Average of wrist-line & elbow-line tilts."""
    wa = _tilt_from_horizontal(_vec(lm[LEFT_WRIST], lm[RIGHT_WRIST]))
    ea = _tilt_from_horizontal(_vec(lm[LEFT_ELBOW], lm[RIGHT_ELBOW]))
    return (wa + ea) / 2.0


# Registry: param name → calculator function
PARAM_CALCULATORS = {
    'shoulder_tilt': calc_shoulder_tilt,
    'pelvic_tilt': calc_pelvic_tilt,
    'trunk_shift': calc_trunk_shift,
    'left_knee_curvature': calc_left_knee_curvature,
    'right_knee_curvature': calc_right_knee_curvature,
    'knee_curvature': calc_knee_curvature,
    'arm_misalignment': calc_arm_misalignment,
    'distance_between_feet': calc_distance_between_feet,
    'left_arm_curvature': calc_left_arm_curvature,
    'right_arm_curvature': calc_right_arm_curvature,
    'left_arm_angle': calc_left_arm_angle,
    'right_arm_angle': calc_right_arm_angle,
}


# ===========================================================================
# Aggregated Stats
# ===========================================================================

def compute_aggregated_stats(values):
    """Compute 8 summary stats from a per-frame parameter series."""
    if len(values) < 2:
        return {s: float('nan') for s in STAT_NAMES}
    v = np.asarray(values, dtype=np.float64)
    s = {
        'mean': float(np.mean(v)),
        'std': float(np.std(v)),
        'min': float(np.min(v)),
        'max': float(np.max(v)),
        'range': float(np.max(v) - np.min(v)),
    }
    d = np.abs(np.diff(v))
    s['rate_of_change'] = float(np.mean(d)) if len(d) > 0 else 0.0
    s['skewness'] = float(pd.Series(v).skew())
    s['auc'] = float(np.trapezoid(v))
    return s


# ===========================================================================
# MediaPipe Helpers
# ===========================================================================

def download_model_if_needed(model_path):
    """Download the PoseLandmarker .task model if not present."""
    if os.path.exists(model_path):
        return
    print("Downloading PoseLandmarker model...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ctx = ssl._create_unverified_context()
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    except Exception:
        with urllib.request.urlopen(POSE_MODEL_URL, context=ctx) as r, \
             open(model_path, 'wb') as f:
            f.write(r.read())


def create_landmarker(model_path):
    """Create a fresh PoseLandmarker instance (one per video for clean timestamps)."""
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


# ===========================================================================
# Model Loading
# ===========================================================================

def load_models(models_dir):
    """Load all trained models from the models/ directory.

    Returns:
        move_clf, quality_clfs (dict[move→clf]), iso_forests (dict[move→clf]),
        normal_avgs (dict)
    """
    def _load(name):
        with open(os.path.join(models_dir, name), 'rb') as f:
            return pickle.load(f)

    move_clf = _load('move_classifier.pkl')

    quality_clfs = {}
    iso_forests = {}
    for m in MOVE_NAMES:
        qfn = f'quality_{m}.pkl'
        if os.path.exists(os.path.join(models_dir, qfn)):
            quality_clfs[m] = _load(qfn)
        ifn = f'isolation_forest_{m}.pkl'
        if os.path.exists(os.path.join(models_dir, ifn)):
            iso_forests[m] = _load(ifn)

    normal_avgs = {}
    navg_path = os.path.join(models_dir, 'normal_averages.json')
    if os.path.exists(navg_path):
        with open(navg_path, 'r') as f:
            normal_avgs = json.load(f)

    return move_clf, quality_clfs, iso_forests, normal_avgs


# ===========================================================================
# Per-Frame Parameter Extraction
# ===========================================================================

def extract_param_series(video_path, model_path):
    """Process every frame of a video and return per-parameter value arrays.

    Returns:
        param_vals: dict[str, list[float]] — 12 keys, each a list of per-frame values
        n_proc: int — number of frames with a detected pose
        n_skip: int — number of frames without a detected pose
    """
    cap = cv2.VideoCapture(video_path)
    lmkr = create_landmarker(model_path)

    param_vals = {p: [] for p in PARAM_NAMES}
    n_proc, n_skip = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        res = lmkr.detect_for_video(mp_img, ts)
        if not res.pose_landmarks:
            n_skip += 1
            continue

        lm = res.pose_landmarks[0]
        h, w = frame.shape[:2]
        pixel_lm = [PixelLandmark(lm[i].x * w, lm[i].y * h) for i in range(33)]

        try:
            for pname, calc_fn in PARAM_CALCULATORS.items():
                param_vals[pname].append(calc_fn(pixel_lm))
        except Exception:
            n_skip += 1
            continue
        n_proc += 1

    cap.release()
    lmkr.close()
    return param_vals, n_proc, n_skip


# ===========================================================================
# Feature Aggregation & Prediction
# ===========================================================================

def aggregate_features(param_vals):
    """Convert per-frame param arrays → 96 aggregated features + model input X."""
    feat = {}
    for p in PARAM_NAMES:
        stats = compute_aggregated_stats(np.array(param_vals[p]))
        for sn in STAT_NAMES:
            feat[f'{p}_{sn}'] = stats[sn]
    X = np.array([list(feat.values())])
    return feat, X


def predict(feat, X, move_clf, quality_clfs, iso_forests):
    """Run all three model layers on aggregated features.

    Returns:
        move_pred, move_conf, quality_pred, quality_conf, anomaly_score,
        quality_score, move_probs, quality_probs
    """
    move_pred = move_clf.predict(X)[0]
    mpb = move_clf.predict_proba(X)[0]
    move_conf = float(max(mpb))
    move_probs = {name: float(p) for name, p in zip(move_clf.classes_, mpb)}

    quality_pred = 'Unknown'
    quality_conf = 0.5
    quality_probs = {}
    if move_pred in quality_clfs:
        quality_pred = quality_clfs[move_pred].predict(X)[0]
        qpb = quality_clfs[move_pred].predict_proba(X)[0]
        quality_conf = float(max(qpb))
        quality_probs = {
            name: float(p)
            for name, p in zip(quality_clfs[move_pred].classes_, qpb)
        }

    anomaly_score = 0.0
    if move_pred in iso_forests:
        try:
            df_val = iso_forests[move_pred].decision_function(X)[0]
            anomaly_score = max(0.0, min(1.0, 0.5 - float(df_val)))
        except ValueError:
            anomaly_score = 0.0

    quality_score = int((1.0 - anomaly_score) * 100)

    return (move_pred, move_conf, quality_pred, quality_conf,
            anomaly_score, quality_score, move_probs, quality_probs)


def compute_top_deviations(feat, move_pred, normal_avgs, top_n=3):
    """Return top-N parameter deviations from Normal baseline for a move.

    Returns:
        list of dicts: [{feature, value, normal_mean, normal_std, z_score, direction}, ...]
    """
    if move_pred not in normal_avgs:
        return []

    norms = normal_avgs[move_pred]
    deviations = []

    for fname, fval in feat.items():
        if fname.endswith(SKIP_DEV_STATS):
            continue
        if fname not in norms:
            continue

        n_mean = norms[fname]
        # Look up the corresponding std key (e.g. shoulder_tilt_mean → shoulder_tilt_std)
        std_key = fname
        for suffix in ['_mean', '_min', '_max', '_range', '_skewness']:
            if std_key.endswith(suffix):
                std_key = std_key[: -len(suffix)] + '_std'
                break
        n_std = norms.get(std_key, 1.0)

        if n_std > 0:
            z = abs(fval - n_mean) / n_std
        else:
            z = 0.0

        diff = fval - n_mean
        deviations.append({
            'feature': fname,
            'value': fval,
            'normal_mean': n_mean,
            'normal_std': n_std,
            'z_score': round(z, 1),
            'direction': 'above' if diff >= 0 else 'below',
            'diff': diff,
        })

    deviations.sort(key=lambda x: x['z_score'], reverse=True)
    return deviations[:top_n]


def format_deviations(deviations):
    """Format deviation dicts into a readable string for CSV output."""
    parts = []
    for d in deviations:
        parts.append(
            f"{d['feature']}: {d['value']:.2f} "
            f"({d['direction']} normal {d['normal_mean']:.2f}±{d['normal_std']:.2f}, "
            f"z={d['z_score']:.1f}σ)"
        )
    return ' | '.join(parts)


# ===========================================================================
# Top-Level: Analyze One Video
# ===========================================================================

def analyze_video(video_path, model_dir, pose_model_path=None):
    """Full pipeline: extract params → aggregate → predict → deviations.

    Args:
        video_path: path to the .mp4 file
        model_dir: directory containing trained .pkl models + normal_averages.json
        pose_model_path: path to PoseLandmarker .task model (auto-downloaded if None)

    Returns:
        dict with all prediction results
    """
    if pose_model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pose_model_path = os.path.join(base_dir, 'data', 'models', 'pose_landmarker.task')

    download_model_if_needed(pose_model_path)

    move_clf, quality_clfs, iso_forests, normal_avgs = load_models(model_dir)

    video_name = os.path.basename(video_path)
    param_vals, n_proc, n_skip = extract_param_series(video_path, pose_model_path)

    if n_proc == 0:
        return {
            'filename': video_name,
            'predicted_move': 'ERROR',
            'predicted_move_confidence': 0.0,
            'predicted_quality': 'ERROR',
            'predicted_quality_confidence': 0.0,
            'anomaly_score': 1.0,
            'quality_score': 0,
            'top_deviations': 'No pose detected',
            'move_probabilities': {},
            'quality_probabilities': {},
            'frames_processed': 0,
            'frames_skipped': n_skip,
        }

    feat, X = aggregate_features(param_vals)

    (move_pred, move_conf, quality_pred, quality_conf,
     anomaly_score, quality_score, move_probs, quality_probs) = \
        predict(feat, X, move_clf, quality_clfs, iso_forests)

    deviations = compute_top_deviations(feat, move_pred, normal_avgs)
    top_dev_str = format_deviations(deviations)

    return {
        'filename': video_name,
        'predicted_move': move_pred,
        'predicted_move_confidence': round(move_conf, 4),
        'predicted_quality': quality_pred,
        'predicted_quality_confidence': round(quality_conf, 4),
        'anomaly_score': round(anomaly_score, 4),
        'quality_score': quality_score,
        'top_deviations': top_dev_str,
        'top_deviations_detail': deviations,
        'move_probabilities': move_probs,
        'quality_probabilities': quality_probs,
        'frames_processed': n_proc,
        'frames_skipped': n_skip,
    }
