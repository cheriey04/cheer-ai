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
import sys
import urllib.request
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

# Shared pipeline module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cheer_ai.pipeline import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    POSE_MODEL_URL, PARAM_NAMES, STAT_NAMES, MOVE_NAMES,
    PixelLandmark, PARAM_CALCULATORS,
    _vec, _norm, _angle_between, _tilt_from_horizontal,
    compute_aggregated_stats, create_landmarker,
)

# Supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}


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
