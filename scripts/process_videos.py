"""
process_videos.py

Processes all videos in data/raw_videos/ using MediaPipe Pose.
Extracts 33 3D landmarks from EVERY frame (no skipping) and computes
5 biomechanical parameters per frame:

  1. Shoulder Tilt   (0-90 deg)  — angle between shoulder vector & horizontal
  2. Pelvic Tilt      (0-90 deg)  — angle between hip vector & horizontal
  3. Trunk Shift      (0-180 deg) — angle between trunk vector & vertical
  4. Knee Curvature   (0-180 deg) — avg left & right knee angle
  5. Arm Misalignment (0-90 deg)  — avg wrist & elbow tilt vs horizontal

Also stores the raw 33 landmarks as JSON so future parameters can be
derived without re-processing videos.

Output: data/processed_features/training_data_frames.csv  (overwritten each run)
"""

# Standard library
import os
import csv
import json
import math
import ssl
import urllib.request
from collections import namedtuple
from pathlib import Path

# Third-party packages
import cv2              # OpenCV — video reading / frame extraction
import mediapipe as mp   # MediaPipe — pose landmark detection
import numpy as np       # NumPy — vector math
import pandas as pd      # Pandas — for skewness in aggregated stats

# MediaPipe Tasks API (0.10.x+)
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices
# ---------------------------------------------------------------------------
# MediaPipe Pose outputs 33 landmarks per detected person.
# Each landmark has .x, .y, .z (normalized coordinates).
# We only define the ones used by our five parameters below.
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Lightweight container for pixel-space landmark coordinates.
# Used to avoid aspect-ratio distortion when computing angles from
# normalized coords (x,y both in [0,1] but image may not be square).
PixelLandmark = namedtuple('PixelLandmark', ['x', 'y'])

# ---- Configuration ----

# Supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}

# URL for the PoseLandmarker model
POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task'
)


# ---------------------------------------------------------------------------
# Vector / angle helpers
# ---------------------------------------------------------------------------

def _vec(a, b):
    """Return 2D vector from landmark a to landmark b.

    Both a and b are MediaPipe NormalizedLandmark objects with .x, .y, .z.
    Uses only X and Y (screen-plane) — Z is ignored to avoid depth distortion.
    """
    return np.array([b.x - a.x, b.y - a.y])


def _norm(v):
    """Euclidean norm (magnitude) of a 2D vector."""
    return float(np.linalg.norm(v))


def _angle_between(v1, v2):
    """Angle in degrees between two 2D vectors (0—180).

    Uses the 2D dot-product formula:
        cos(theta) = (v1 · v2) / (|v1| * |v2|)

    Returns 0 if either vector has near-zero magnitude (degenerate case).
    """
    dot = float(np.dot(v1, v2))          # v1 · v2
    n1 = _norm(v1)                        # |v1|
    n2 = _norm(v2)                        # |v2|
    if n1 < 1e-9 or n2 < 1e-9:            # Avoid division by zero
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (n1 * n2)))  # Clamp for numerical safety
    return math.degrees(math.acos(cos_theta))          # Radians → degrees


def _tilt_from_horizontal(vec):
    """
    Angle (0–90 deg) between `vec` and the horizontal axis (1, 0).

    Takes min(angle, 180-angle) so a vector pointing down-left (e.g. 150 deg)
    is treated the same as 30 deg — both are 30 deg away from level.
    0 deg = perfectly horizontal / level.
    """
    horizontal = np.array([1.0, 0.0])
    raw = _angle_between(vec, horizontal)     # 0–180 deg
    return min(raw, 180.0 - raw)              # Fold into 0–90


# ---------------------------------------------------------------------------
# Parameter calculators  (all use X, Y, Z)
# ---------------------------------------------------------------------------

def calc_shoulder_tilt(lm):
    """Parameter 1: Shoulder Tilt (0–90 deg).  0 = level shoulders.

    Creates a 3D vector from left shoulder (11) → right shoulder (12),
    then measures its angle against the horizontal axis.
    """
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]))


def calc_pelvic_tilt(lm):
    """Parameter 2: Pelvic Tilt (0–90 deg).  0 = level hips.

    Creates a 3D vector from left hip (23) → right hip (24),
    then measures its angle against the horizontal axis.
    """
    return _tilt_from_horizontal(_vec(lm[LEFT_HIP], lm[RIGHT_HIP]))


def calc_trunk_shift(lm):
    """Parameter 3: Trunk Shift (0–90 deg).  0 = perfectly upright.

    Step 1: Compute the 2D midpoint of both shoulders and both hips.
    Step 2: Trunk vector = shoulder_midpoint - hip_midpoint (points upward).
    Step 3: Measure the 2D angle between the trunk vector and upward-vertical (0,-1).
    """
    # Midpoint of left & right shoulders (X, Y)
    shoulder_mid = np.array([
        (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2,
        (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2,
    ])
    # Midpoint of left & right hips (X, Y)
    hip_mid = np.array([
        (lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2,
        (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2,
    ])
    trunk = shoulder_mid - hip_mid          # Vector from hips → shoulders (points up)
    upward = np.array([0.0, -1.0])           # Screen coords: -Y = upward
    return _angle_between(trunk, upward)     # 0° = upright, ~90° = horizontal


def calc_left_knee_curvature(lm):
    """Left Knee Curvature — athlete's left leg (0–180 deg).  180 = fully straight.

    Athlete facing camera: their left = camera's right = MediaPipe RIGHT landmarks.
    Uses MediaPipe RIGHT_HIP(24) → RIGHT_KNEE(26) → RIGHT_ANKLE(28).
    Computes the bend angle and inverts so 180° = straight, 0° = fully bent.
    """
    bend = _angle_between(
        _vec(lm[RIGHT_HIP], lm[RIGHT_KNEE]),
        _vec(lm[RIGHT_KNEE], lm[RIGHT_ANKLE]),
    )
    return 180.0 - bend


def calc_right_knee_curvature(lm):
    """Right Knee Curvature — athlete's right leg (0–180 deg).  180 = fully straight.

    Athlete facing camera: their right = camera's left = MediaPipe LEFT landmarks.
    Uses MediaPipe LEFT_HIP(23) → LEFT_KNEE(25) → LEFT_ANKLE(27).
    """
    bend = _angle_between(
        _vec(lm[LEFT_HIP], lm[LEFT_KNEE]),
        _vec(lm[LEFT_KNEE], lm[LEFT_ANKLE]),
    )
    return 180.0 - bend


def calc_knee_curvature(lm):
    """Combined Knee Curvature — average of left and right (0–180 deg).  180 = straight legs."""
    return (calc_left_knee_curvature(lm) + calc_right_knee_curvature(lm)) / 2.0


def calc_arm_misalignment(lm):
    """Parameter 5: Arm Misalignment (0–90 deg).  0 = level arms.

    Creates two 3D vectors:
        - Wrist vector:  left wrist (15) → right wrist (16)
        - Elbow vector:  left elbow (13) → right elbow (14)

    Measures each against the horizontal axis and averages them.
    """
    # Tilt of the line connecting both wrists
    wrist_angle = _tilt_from_horizontal(_vec(lm[LEFT_WRIST], lm[RIGHT_WRIST]))
    # Tilt of the line connecting both elbows
    elbow_angle = _tilt_from_horizontal(_vec(lm[LEFT_ELBOW], lm[RIGHT_ELBOW]))
    # Average for a single arm-level metric
    return (wrist_angle + elbow_angle) / 2.0


def calc_distance_between_feet(lm):
    """Distance Between Feet — magnitude of vector between left and right ankles.

    Uses MediaPipe LEFT_ANKLE(27) and RIGHT_ANKLE(28).
    Returns normalized Euclidean distance (0+).
    """
    return _norm(_vec(lm[LEFT_ANKLE], lm[RIGHT_ANKLE]))


def calc_left_arm_curvature(lm):
    """Left Arm Curvature — athlete's left arm (0–180 deg).  180 = fully straight.

    Athlete facing camera: their left = camera's right = MediaPipe RIGHT landmarks.
    Angle at elbow: wrist→elbow vs elbow→shoulder.
    Uses MediaPipe RIGHT_WRIST(16) → RIGHT_ELBOW(14) → RIGHT_SHOULDER(12).
    """
    bend = _angle_between(
        _vec(lm[RIGHT_WRIST], lm[RIGHT_ELBOW]),
        _vec(lm[RIGHT_ELBOW], lm[RIGHT_SHOULDER]),
    )
    return 180.0 - bend


def calc_right_arm_curvature(lm):
    """Right Arm Curvature — athlete's right arm (0–180 deg).  180 = fully straight.

    Athlete facing camera: their right = camera's left = MediaPipe LEFT landmarks.
    Angle at elbow: wrist→elbow vs elbow→shoulder.
    Uses MediaPipe LEFT_WRIST(15) → LEFT_ELBOW(13) → LEFT_SHOULDER(11).
    """
    bend = _angle_between(
        _vec(lm[LEFT_WRIST], lm[LEFT_ELBOW]),
        _vec(lm[LEFT_ELBOW], lm[LEFT_SHOULDER]),
    )
    return 180.0 - bend


def calc_left_arm_angle(lm):
    """Left Arm Angle — athlete's left upper arm vs horizontal (0–90 deg).  0 = level.

    Athlete facing camera: their left = camera's right = MediaPipe RIGHT landmarks.
    Vector from shoulder to elbow, measured against horizontal.
    Uses MediaPipe RIGHT_SHOULDER(12) → RIGHT_ELBOW(14).
    """
    return _tilt_from_horizontal(_vec(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW]))


def calc_right_arm_angle(lm):
    """Right Arm Angle — athlete's right upper arm vs horizontal (0–90 deg).  0 = level.

    Athlete facing camera: their right = camera's left = MediaPipe LEFT landmarks.
    Vector from shoulder to elbow, measured against horizontal.
    Uses MediaPipe LEFT_SHOULDER(11) → LEFT_ELBOW(13).
    """
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[LEFT_ELBOW]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_model_if_needed(model_path: str) -> None:
    """Download the PoseLandmarker .task model file if it doesn't exist.

    Uses an unverified SSL context to work around macOS certificate issues
    with some Python installations.  The model is hosted on Google's CDN
    so the download itself is safe.
    """
    if os.path.exists(model_path):
        print(f"Model already downloaded: {model_path}")
        return

    print(f"Downloading PoseLandmarker model (~5 MB)...")
    print(f"  From: {POSE_MODEL_URL}")
    print(f"  To:   {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # macOS Python often has SSL cert issues — use unverified context as
    # a safe workaround for a known Google-hosted file.
    ssl_context = ssl._create_unverified_context()
    try:
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
    except Exception:
        # Fallback: if system certs are missing, skip verification
        print("  SSL verification failed, retrying with unverified context...")
        with urllib.request.urlopen(
            POSE_MODEL_URL, context=ssl_context
        ) as response, open(model_path, 'wb') as out_file:
            out_file.write(response.read())
    print("  Download complete.\n")


def severity_from_filename(filename: str) -> str:
    """Extract severity label from the video filename.

    Looks for the substring '_normal_' or '_bad_' in the filename.
    Examples:
        'standing_normal_01.mp4'  → 'Normal'
        'tjump_bad_05.mp4'        → 'Bad'
        'unknown_video.mp4'       → 'Unknown'
    """
    basename = os.path.basename(filename)
    if '_normal_' in basename:
        return 'Normal'
    if '_bad_' in basename:
        return 'Bad'
    return 'Unknown'


def landmarks_to_json(landmark_list) -> str:
    """Convert 33 MediaPipe NormalizedLandmark objects to a compact JSON string.

    Each landmark is stored as {"x": ..., "y": ..., "z": ...} with
    6 decimal places of precision. This preserves all raw coordinates
    so new parameters can be derived later without re-processing videos.
    """
    data = []
    for lm in landmark_list:
        data.append({
            'x': round(lm.x, 6),
            'y': round(lm.y, 6),
            'z': round(lm.z, 6),
        })
    return json.dumps(data)


# ---------------------------------------------------------------------------
# Aggregated feature computation
# ---------------------------------------------------------------------------

STAT_NAMES = ['mean', 'std', 'min', 'max', 'range', 'rate_of_change', 'skewness', 'auc']


def compute_aggregated_stats(values: np.ndarray) -> dict:
    """Compute 8 summary statistics from a sequence of per-frame values.

    Args:
        values: 1-D numpy array of a single parameter across all frames.

    Returns:
        dict with keys: mean, std, min, max, range, rate_of_change, skewness, auc
    """
    if len(values) < 2:
        # Not enough frames for meaningful stats — return NaN
        return {s: float('nan') for s in STAT_NAMES}

    vals = np.asarray(values, dtype=np.float64)
    stats = {}
    stats['mean'] = float(np.mean(vals))
    stats['std'] = float(np.std(vals))
    stats['min'] = float(np.min(vals))
    stats['max'] = float(np.max(vals))
    stats['range'] = stats['max'] - stats['min']

    # Rate of change: average absolute difference between consecutive frames
    diffs = np.abs(np.diff(vals))
    stats['rate_of_change'] = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    # Skewness via pandas (handles edge cases)
    stats['skewness'] = float(pd.Series(vals).skew())

    # Area under the curve (trapezoidal integral over frame index)
    stats['auc'] = float(np.trapezoid(vals))

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_videos():
    """Main entry point: scan raw_videos/, process every frame, write CSV."""

    # ---- Resolve paths ------------------------------------------------
    # base_dir = cheer-ai/  (two levels up from this script)
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / 'data' / 'raw_videos'
    out_dir = base_dir / 'data' / 'processed_features'
    out_dir.mkdir(parents=True, exist_ok=True)      # Create output folder if needed

    csv_path = out_dir / 'training_data_frames.csv'
    agg_csv_path = out_dir / 'training_data_aggregated.csv'

    # ---- Discover all video files ------------------------------------
    # Recursively find .mp4, .mov, .avi, .webm, .mkv in all subfolders
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(raw_dir.rglob(f'*{ext}'))
    video_files = sorted(video_files)                # Deterministic order

    total_videos = len(video_files)
    print(f"Found {total_videos} videos across all move folders.\n")

    # ---- Download MediaPipe Pose model (once) ------------------------
    # The new Tasks API (0.10.x+) uses a .task model file instead of the
    # old solutions API.  We auto-download it on first run.
    model_path = str(base_dir / 'data' / 'models' / 'pose_landmarker.task')
    download_model_if_needed(model_path)

    # Shared options — a fresh landmarker is created per video to avoid
    # VIDEO mode's monotonically-increasing timestamp requirement.
    landmarker_options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO,   # Temporal tracking across frames
        num_poses=1,                                 # Single-person detection
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ---- Open both CSVs for writing (overwrite existing) ------------
    param_names = ['shoulder_tilt', 'pelvic_tilt', 'trunk_shift',
                   'left_knee_curvature', 'right_knee_curvature',
                   'knee_curvature', 'arm_misalignment',
                   'distance_between_feet',
                   'left_arm_curvature', 'right_arm_curvature',
                   'left_arm_angle', 'right_arm_angle']

    # Per-frame CSV header
    frame_header = ['video_filename', 'move_name', 'severity_label',
                    'frame_number'] + param_names + ['landmarks_json']

    # Aggregated CSV header (one row per video)
    agg_header = ['video_filename', 'move_name', 'severity_label']
    for param in param_names:
        for stat in STAT_NAMES:
            agg_header.append(f'{param}_{stat}')

    with open(csv_path, 'w', newline='') as f, \
         open(agg_csv_path, 'w', newline='') as agg_f:

        frame_writer = csv.writer(f)
        agg_writer = csv.writer(agg_f)

        frame_writer.writerow(frame_header)
        agg_writer.writerow(agg_header)

        frames_written = 0   # Total frames successfully processed
        frames_skipped = 0   # Total frames where pose detection failed

        # ---- Process each video --------------------------------------
        for vid_idx, vid_path in enumerate(video_files, 1):
            video_filename = vid_path.name         # e.g. "standing_normal_01.mp4"
            move_name = vid_path.parent.name       # e.g. "standing" (folder name)
            severity = severity_from_filename(video_filename)  # "Normal" or "Bad"

            # Open video with OpenCV
            cap = cv2.VideoCapture(str(vid_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"[{vid_idx}/{total_videos}] {move_name}/{video_filename} "
                  f"({total_frames} frames) ... ", end='', flush=True)

            local_written = 0   # Frames written for THIS video
            local_skipped = 0   # Frames skipped (no pose detected) for THIS video
            frame_num = 0       # 1-based frame counter

            # Collect per-frame parameter values for aggregation
            param_values = {p: [] for p in param_names}

            # ---- Create a fresh PoseLandmarker for this video -------
            # Each video gets its own landmarker so timestamps start at 0.
            landmarker = vision.PoseLandmarker.create_from_options(landmarker_options)

            # ---- Process every frame (no skipping) -------------------
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break            # End of video

                frame_num += 1
                h, w = frame.shape[:2]  # frame dimensions for aspect-ratio correction

                # MediaPipe Tasks API: convert BGR → RGB, then wrap as mp.Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # Timestamp in milliseconds (required for VIDEO mode)
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                # Skip frame if no person/pose was detected
                if not results.pose_landmarks:
                    local_skipped += 1
                    continue

                # 33 landmarks for the first (and only) detected person
                lm = results.pose_landmarks[0]
                # Denormalize to pixel coordinates to avoid aspect-ratio distortion
                pixel_lm = [PixelLandmark(lm[i].x * w, lm[i].y * h) for i in range(33)]

                # ---- Calculate all 5 parameters ----------------------
                try:
                    st = calc_shoulder_tilt(pixel_lm)       # Parameter 1
                    pt = calc_pelvic_tilt(pixel_lm)          # Parameter 2
                    ts = calc_trunk_shift(pixel_lm)          # Parameter 3
                    lk = calc_left_knee_curvature(pixel_lm)  # Parameter 4
                    rk = calc_right_knee_curvature(pixel_lm) # Parameter 5
                    kc = calc_knee_curvature(pixel_lm)       # Parameter 6
                    am = calc_arm_misalignment(pixel_lm)     # Parameter 7
                    df_ = calc_distance_between_feet(pixel_lm)  # Parameter 8
                    la = calc_left_arm_curvature(pixel_lm)   # Parameter 9
                    ra = calc_right_arm_curvature(pixel_lm)  # Parameter 10
                    laa = calc_left_arm_angle(pixel_lm)      # Parameter 11
                    raa = calc_right_arm_angle(pixel_lm)     # Parameter 12
                    lm_json = landmarks_to_json(lm)    # Raw landmarks backup (normalized)
                except Exception:
                    # Safety net — skip frame if math fails for any reason
                    local_skipped += 1
                    continue

                # Write one row per frame (no aggregations)
                frame_writer.writerow([
                    video_filename,
                    move_name,
                    severity,
                    frame_num,
                    round(st, 4),       # Shoulder tilt
                    round(pt, 4),       # Pelvic tilt
                    round(ts, 4),       # Trunk shift
                    round(lk, 4),       # Left knee curvature
                    round(rk, 4),       # Right knee curvature
                    round(kc, 4),       # Knee curvature (combined)
                    round(am, 4),       # Arm misalignment
                    round(df_, 4),      # Distance between feet
                    round(la, 4),       # Left arm curvature
                    round(ra, 4),       # Right arm curvature
                    round(laa, 4),      # Left arm angle
                    round(raa, 4),      # Right arm angle
                    lm_json,            # Full 33-landmark JSON string
                ])
                local_written += 1

                # Collect values for per-video aggregation
                param_values['shoulder_tilt'].append(st)
                param_values['pelvic_tilt'].append(pt)
                param_values['trunk_shift'].append(ts)
                param_values['left_knee_curvature'].append(lk)
                param_values['right_knee_curvature'].append(rk)
                param_values['knee_curvature'].append(kc)
                param_values['arm_misalignment'].append(am)
                param_values['distance_between_feet'].append(df_)
                param_values['left_arm_curvature'].append(la)
                param_values['right_arm_curvature'].append(ra)
                param_values['left_arm_angle'].append(laa)
                param_values['right_arm_angle'].append(raa)

            # Release video file handle
            cap.release()

            # Release this video's landmarker
            landmarker.close()

            # ---- Compute per-video aggregated stats -----------------
            suffix = f", {local_skipped} skipped" if local_skipped else ""

            if local_written == 0:
                # No frames with detected pose — skip aggregated row
                print(f"{local_written} frames written{suffix}")
                continue

            agg_row = [video_filename, move_name, severity]
            for param in param_names:
                vals = np.array(param_values[param], dtype=np.float64)
                stats = compute_aggregated_stats(vals)
                for stat_name in STAT_NAMES:
                    agg_row.append(round(stats[stat_name], 6)
                                   if not np.isnan(stats[stat_name])
                                   else stats[stat_name])
            agg_writer.writerow(agg_row)

            # Accumulate totals
            frames_written += local_written
            frames_skipped += local_skipped

            # Per-video summary line
            print(f"{local_written} frames written{suffix}")

    # ---- Final summary -----------------------------------------------
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Per-frame CSV:   {csv_path}")
    print(f"  Aggregated CSV:  {agg_csv_path}")
    print(f"  Frames written:  {frames_written}")
    print(f"  Frames skipped:  {frames_skipped}")
    print(f"{'='*60}")


if __name__ == '__main__':
    process_videos()
