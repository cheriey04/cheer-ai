"""
visualize_frames.py — Draw 10 random frames from a video with all 12
biomechanical parameters labeled: keypoints, vectors, and angle callouts.

Usage:
  python scripts/visualize_frames.py test_videos/test_video_020.mp4 [num_frames]
"""

import os, sys, math, random, ssl, urllib.request
from collections import namedtuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

# ── Landmark indices ──────────────────────────────────────────────────
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_WRIST = 15; RIGHT_WRIST = 16
LEFT_HIP = 23; RIGHT_HIP = 24
LEFT_KNEE = 25; RIGHT_KNEE = 26
LEFT_ANKLE = 27; RIGHT_ANKLE = 28

POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-assets/pose_landmarker.task'

PixelLandmark = namedtuple('PixelLandmark', ['x', 'y'])

# ── Math helpers ──────────────────────────────────────────────────────
def _vec(a, b):
    return np.array([b.x - a.x, b.y - a.y])

def _norm(v):
    return float(np.linalg.norm(v))

def _angle_between(v1, v2):
    dot = float(np.dot(v1, v2))
    n1, n2 = _norm(v1), _norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))

def _tilt_from_horizontal(vec):
    raw = _angle_between(vec, np.array([1.0, 0.0]))
    return min(raw, 180.0 - raw)

# ── Parameter calculators (same as process_videos.py) ─────────────────
def calc_shoulder_tilt(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]))

def calc_pelvic_tilt(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_HIP], lm[RIGHT_HIP]))

def calc_trunk_shift(lm):
    sm = np.array([(lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2,
                   (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2])
    hm = np.array([(lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2,
                   (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2])
    return _angle_between(sm - hm, np.array([0.0, -1.0]))

def calc_left_knee_curvature(lm):
    bend = _angle_between(_vec(lm[RIGHT_HIP], lm[RIGHT_KNEE]),
                          _vec(lm[RIGHT_KNEE], lm[RIGHT_ANKLE]))
    return 180.0 - bend

def calc_right_knee_curvature(lm):
    bend = _angle_between(_vec(lm[LEFT_HIP], lm[LEFT_KNEE]),
                          _vec(lm[LEFT_KNEE], lm[LEFT_ANKLE]))
    return 180.0 - bend

def calc_knee_curvature(lm):
    return (calc_left_knee_curvature(lm) + calc_right_knee_curvature(lm)) / 2.0

def calc_distance_between_feet(lm):
    return _norm(_vec(lm[LEFT_ANKLE], lm[RIGHT_ANKLE]))

def calc_left_arm_curvature(lm):
    bend = _angle_between(_vec(lm[RIGHT_WRIST], lm[RIGHT_ELBOW]),
                          _vec(lm[RIGHT_ELBOW], lm[RIGHT_SHOULDER]))
    return 180.0 - bend

def calc_right_arm_curvature(lm):
    bend = _angle_between(_vec(lm[LEFT_WRIST], lm[LEFT_ELBOW]),
                          _vec(lm[LEFT_ELBOW], lm[LEFT_SHOULDER]))
    return 180.0 - bend

def calc_left_arm_angle(lm):
    return _tilt_from_horizontal(_vec(lm[RIGHT_SHOULDER], lm[RIGHT_ELBOW]))

def calc_right_arm_angle(lm):
    return _tilt_from_horizontal(_vec(lm[LEFT_SHOULDER], lm[LEFT_ELBOW]))

def calc_arm_misalignment(lm):
    wa = _tilt_from_horizontal(_vec(lm[LEFT_WRIST], lm[RIGHT_WRIST]))
    ea = _tilt_from_horizontal(_vec(lm[LEFT_ELBOW], lm[RIGHT_ELBOW]))
    return (wa + ea) / 2.0

# ── Drawing helpers ───────────────────────────────────────────────────
def denorm(pt, w, h):
    """Convert normalized landmark to pixel coordinates."""
    return (int(pt.x * w), int(pt.y * h))

def draw_angle_arc(img, apex, p1, p2, color, label, radius=40):
    """Draw a small arc at apex showing the angle between apex→p1 and apex→p2."""
    v1 = np.array([p1[0] - apex[0], p1[1] - apex[1]])
    v2 = np.array([p2[0] - apex[0], p2[1] - apex[1]])
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    # Draw arc
    cv2.ellipse(img, apex, (radius, radius), 0,
                math.degrees(min(a1, a2)), math.degrees(max(a1, a2)),
                color, 2, cv2.LINE_AA)
    # Label midpoint of arc
    mid_a = (a1 + a2) / 2
    label_pt = (int(apex[0] + (radius + 25) * math.cos(mid_a)),
                int(apex[1] + (radius + 25) * math.sin(mid_a)))
    cv2.putText(img, label, label_pt, cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)

def draw_vector(img, p1, p2, color, thickness=2):
    """Draw an arrow from p1 to p2."""
    cv2.arrowedLine(img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=0.08)

# ── Main visualization ────────────────────────────────────────────────
def visualize_frames(video_path, num_frames=10):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'data', 'models', 'pose_landmarker.task')

    # Download model if needed
    if not os.path.exists(model_path):
        print("Downloading PoseLandmarker model...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        ctx = ssl._create_unverified_context()
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
        except Exception:
            with urllib.request.urlopen(POSE_MODEL_URL, context=ctx) as r, \
                 open(model_path, 'wb') as f:
                f.write(r.read())

    # Create landmarker
    o = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionTaskRunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    lmkr = vision.PoseLandmarker.create_from_options(o)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path)

    # First pass: find all frames with valid pose detections
    print(f"Scanning {video_name} ({total} frames) for poses...")
    valid_frames = []
    all_frames_data = []  # store (frame_idx, frame_bgr, landmarks)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        res = lmkr.detect_for_video(mp_img, ts)
        if res.pose_landmarks:
            valid_frames.append(frame_idx)
            all_frames_data.append((frame_idx, frame.copy(), res.pose_landmarks[0]))

    cap.release()
    lmkr.close()

    if len(valid_frames) < num_frames:
        print(f"Only {len(valid_frames)} valid frames found, using all.")
        num_frames = len(valid_frames)

    # Pick random frames
    random.seed(42)
    chosen = random.sample(range(len(all_frames_data)), num_frames)
    chosen.sort()

    out_dir = os.path.join(base_dir, 'output_frames')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nAnnotating {num_frames} random frames → {out_dir}/\n")

    # Landmark connections for skeleton (MediaPipe pose connections)
    CONNECTIONS = [
        (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST), (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]

    COLORS = {
        'shoulder_tilt': (0, 255, 255),       # yellow
        'pelvic_tilt': (255, 0, 255),          # magenta
        'trunk_shift': (255, 255, 0),          # cyan
        'knee': (0, 255, 0),                   # green
        'arm_curvature': (255, 165, 0),        # orange
        'arm_angle': (0, 165, 255),            # gold
        'arm_misalignment': (128, 0, 128),     # purple
        'feet': (255, 0, 0),                   # blue (BGR)
    }

    for i, idx in enumerate(chosen):
        frame_idx, frame, lm = all_frames_data[idx]
        h, w = frame.shape[:2]

        # Denormalize landmarks for drawing + pixel-space angle calc
        h, w = frame.shape[:2]
        pts = {}
        pixel_lm = []
        for lid in range(33):
            pts[lid] = denorm(lm[lid], w, h)
            pixel_lm.append(PixelLandmark(lm[lid].x * w, lm[lid].y * h))

        # Draw skeleton
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)

        # Draw all landmarks
        for lid in range(33):
            cv2.circle(frame, pts[lid], 3, (0, 255, 0), -1, cv2.LINE_AA)

        # ═══ PARAMETER 1: Shoulder Tilt ═══
        draw_vector(frame, pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER], COLORS['shoulder_tilt'], 3)
        # Show horizontal reference
        mid_sh = ((pts[LEFT_SHOULDER][0] + pts[RIGHT_SHOULDER][0]) // 2,
                  (pts[LEFT_SHOULDER][1] + pts[RIGHT_SHOULDER][1]) // 2)
        cv2.line(frame, (mid_sh[0] - 50, mid_sh[1]), (mid_sh[0] + 50, mid_sh[1]),
                 COLORS['shoulder_tilt'], 1, cv2.LINE_AA)
        val = calc_shoulder_tilt(pixel_lm)
        cv2.putText(frame, f"shoulder_tilt={val:.1f}", (mid_sh[0] - 100, mid_sh[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['shoulder_tilt'], 1, cv2.LINE_AA)

        # ═══ PARAMETER 2: Pelvic Tilt ═══
        draw_vector(frame, pts[LEFT_HIP], pts[RIGHT_HIP], COLORS['pelvic_tilt'], 3)
        mid_hip = ((pts[LEFT_HIP][0] + pts[RIGHT_HIP][0]) // 2,
                   (pts[LEFT_HIP][1] + pts[RIGHT_HIP][1]) // 2)
        cv2.line(frame, (mid_hip[0] - 50, mid_hip[1]), (mid_hip[0] + 50, mid_hip[1]),
                 COLORS['pelvic_tilt'], 1, cv2.LINE_AA)
        val = calc_pelvic_tilt(pixel_lm)
        cv2.putText(frame, f"pelvic_tilt={val:.1f}", (mid_hip[0] - 100, mid_hip[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['pelvic_tilt'], 1, cv2.LINE_AA)

        # ═══ PARAMETER 3: Trunk Shift ═══
        sm = ((pts[LEFT_SHOULDER][0] + pts[RIGHT_SHOULDER][0]) // 2,
              (pts[LEFT_SHOULDER][1] + pts[RIGHT_SHOULDER][1]) // 2)
        hm = ((pts[LEFT_HIP][0] + pts[RIGHT_HIP][0]) // 2,
              (pts[LEFT_HIP][1] + pts[RIGHT_HIP][1]) // 2)
        draw_vector(frame, hm, sm, COLORS['trunk_shift'], 3)
        # Vertical reference
        cv2.line(frame, (hm[0], hm[1] - 60), (hm[0], hm[1] + 60),
                 COLORS['trunk_shift'], 1, cv2.LINE_AA)
        val = calc_trunk_shift(pixel_lm)
        cv2.putText(frame, f"trunk_shift={val:.1f}", (hm[0] + 15, (hm[1] + sm[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['trunk_shift'], 1, cv2.LINE_AA)

        # ═══ PARAMETER 4+5: Knee Curvature (Left = RIGHT landmarks, Right = LEFT landmarks) ═══
        # Athlete's LEFT knee = MediaPipe RIGHT
        draw_angle_arc(frame, pts[RIGHT_KNEE], pts[RIGHT_HIP], pts[RIGHT_ANKLE],
                       COLORS['knee'], f"L knee={calc_left_knee_curvature(pixel_lm):.0f}")
        # Athlete's RIGHT knee = MediaPipe LEFT
        draw_angle_arc(frame, pts[LEFT_KNEE], pts[LEFT_HIP], pts[LEFT_ANKLE],
                       COLORS['knee'], f"R knee={calc_right_knee_curvature(pixel_lm):.0f}")

        # ═══ PARAMETER 6: Distance Between Feet ═══
        draw_vector(frame, pts[LEFT_ANKLE], pts[RIGHT_ANKLE], COLORS['feet'], 3)
        val = calc_distance_between_feet(pixel_lm)
        mid_ank = ((pts[LEFT_ANKLE][0] + pts[RIGHT_ANKLE][0]) // 2,
                   (pts[LEFT_ANKLE][1] + pts[RIGHT_ANKLE][1]) // 2)
        cv2.putText(frame, f"feet_dist={val:.2f}", (mid_ank[0] - 50, mid_ank[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['feet'], 1, cv2.LINE_AA)

        # ═══ PARAMETER 7+8: Arm Curvature (Left = RIGHT landmarks, Right = LEFT landmarks) ═══
        # Athlete's LEFT arm = MediaPipe RIGHT
        draw_angle_arc(frame, pts[RIGHT_ELBOW], pts[RIGHT_WRIST], pts[RIGHT_SHOULDER],
                       COLORS['arm_curvature'], f"L arm curv={calc_left_arm_curvature(pixel_lm):.0f}")
        # Athlete's RIGHT arm = MediaPipe LEFT
        draw_angle_arc(frame, pts[LEFT_ELBOW], pts[LEFT_WRIST], pts[LEFT_SHOULDER],
                       COLORS['arm_curvature'], f"R arm curv={calc_right_arm_curvature(pixel_lm):.0f}")

        # ═══ PARAMETER 9+10: Arm Angle vs Horizontal ═══
        # Athlete's LEFT upper arm = MediaPipe RIGHT shoulder→elbow
        cv2.line(frame, (pts[RIGHT_SHOULDER][0] - 30, pts[RIGHT_SHOULDER][1]),
                 (pts[RIGHT_SHOULDER][0] + 30, pts[RIGHT_SHOULDER][1]),
                 COLORS['arm_angle'], 1, cv2.LINE_AA)
        draw_vector(frame, pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW], COLORS['arm_angle'], 2)
        val = calc_left_arm_angle(pixel_lm)
        mid_l = ((pts[RIGHT_SHOULDER][0] + pts[RIGHT_ELBOW][0]) // 2,
                 (pts[RIGHT_SHOULDER][1] + pts[RIGHT_ELBOW][1]) // 2)
        cv2.putText(frame, f"L arm ang={val:.1f}", (mid_l[0] + 10, mid_l[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['arm_angle'], 1, cv2.LINE_AA)

        # Athlete's RIGHT upper arm = MediaPipe LEFT shoulder→elbow
        cv2.line(frame, (pts[LEFT_SHOULDER][0] - 30, pts[LEFT_SHOULDER][1]),
                 (pts[LEFT_SHOULDER][0] + 30, pts[LEFT_SHOULDER][1]),
                 COLORS['arm_angle'], 1, cv2.LINE_AA)
        draw_vector(frame, pts[LEFT_SHOULDER], pts[LEFT_ELBOW], COLORS['arm_angle'], 2)
        val = calc_right_arm_angle(pixel_lm)
        mid_r = ((pts[LEFT_SHOULDER][0] + pts[LEFT_ELBOW][0]) // 2,
                 (pts[LEFT_SHOULDER][1] + pts[LEFT_ELBOW][1]) // 2)
        cv2.putText(frame, f"R arm ang={val:.1f}", (mid_r[0] + 10, mid_r[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['arm_angle'], 1, cv2.LINE_AA)

        # ═══ PARAMETER 11+12: Arm Misalignment ═══
        draw_vector(frame, pts[LEFT_WRIST], pts[RIGHT_WRIST], COLORS['arm_misalignment'], 2)
        draw_vector(frame, pts[LEFT_ELBOW], pts[RIGHT_ELBOW], COLORS['arm_misalignment'], 2)
        val = calc_arm_misalignment(pixel_lm)
        # Place label near wrists
        mw = ((pts[LEFT_WRIST][0] + pts[RIGHT_WRIST][0]) // 2,
              (pts[LEFT_WRIST][1] + pts[RIGHT_WRIST][1]) // 2)
        cv2.putText(frame, f"arm_misalign={val:.1f}", (mw[0] - 70, mw[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['arm_misalignment'], 1, cv2.LINE_AA)

        # ── Frame info ──
        cv2.putText(frame, f"Frame {frame_idx}/{total}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"  [{i+1}/{num_frames}] frame {frame_idx} → {out_path}")

    print(f"\nDone. {num_frames} annotated frames in {out_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_frames.py <video_path> [num_frames]")
        sys.exit(1)
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    visualize_frames(sys.argv[1], nf)
