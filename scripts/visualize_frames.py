"""
visualize_frames.py — Draw 10 random frames from a video with all 12
biomechanical parameters labeled: keypoints, vectors, and angle callouts.

Usage:
  python scripts/visualize_frames.py test_videos/test_video_020.mp4 [num_frames]
"""

import os, sys, math, random, ssl, urllib.request

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cheer_ai.pipeline import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    POSE_MODEL_URL, PixelLandmark,
    calc_shoulder_tilt, calc_pelvic_tilt, calc_trunk_shift,
    calc_left_knee_curvature, calc_right_knee_curvature, calc_knee_curvature,
    calc_distance_between_feet,
    calc_left_arm_curvature, calc_right_arm_curvature,
    calc_left_arm_angle, calc_right_arm_angle,
    calc_arm_misalignment,
    create_landmarker, download_model_if_needed,
)

# ── Drawing helpers ───────────────────────────────────────────────────
def denorm(pt, w, h):
    return (int(pt.x * w), int(pt.y * h))

def draw_angle_arc(img, apex, p1, p2, color, label, radius=40):
    v1 = np.array([p1[0] - apex[0], p1[1] - apex[1]])
    v2 = np.array([p2[0] - apex[0], p2[1] - apex[1]])
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    cv2.ellipse(img, apex, (radius, radius), 0,
                math.degrees(min(a1, a2)), math.degrees(max(a1, a2)),
                color, 2, cv2.LINE_AA)
    mid_a = (a1 + a2) / 2
    label_pt = (int(apex[0] + (radius + 25) * math.cos(mid_a)),
                int(apex[1] + (radius + 25) * math.sin(mid_a)))
    cv2.putText(img, label, label_pt, cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA)

def draw_vector(img, p1, p2, color, thickness=2):
    cv2.arrowedLine(img, p1, p2, color, thickness, cv2.LINE_AA, tipLength=0.08)


def visualize_frames(video_path, num_frames=10):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'data', 'models', 'pose_landmarker.task')
    download_model_if_needed(model_path)

    lmkr = create_landmarker(model_path)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.basename(video_path)

    print(f"Scanning {video_name} ({total} frames) for poses...")
    all_frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        res = lmkr.detect_for_video(mp_img, ts)
        if res.pose_landmarks:
            all_frames_data.append((frame_idx, frame.copy(), res.pose_landmarks[0]))

    cap.release()
    lmkr.close()

    num_frames = min(num_frames, len(all_frames_data))
    random.seed(42)
    chosen = sorted(random.sample(range(len(all_frames_data)), num_frames))

    out_dir = os.path.join(base_dir, 'output_frames')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nAnnotating {num_frames} random frames → {out_dir}/\n")

    CONNECTIONS = [
        (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW),
        (RIGHT_ELBOW, RIGHT_WRIST), (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    COLORS = {
        'shoulder_tilt': (0, 255, 255), 'pelvic_tilt': (255, 0, 255),
        'trunk_shift': (255, 255, 0), 'knee': (0, 255, 0),
        'arm_curvature': (255, 165, 0), 'arm_angle': (0, 165, 255),
        'arm_misalignment': (128, 0, 128), 'feet': (255, 0, 0),
    }

    for i, idx in enumerate(chosen):
        frame_idx, frame, lm = all_frames_data[idx]
        h, w = frame.shape[:2]

        pts = {}
        pixel_lm = []
        for lid in range(33):
            pts[lid] = denorm(lm[lid], w, h)
            pixel_lm.append(PixelLandmark(lm[lid].x * w, lm[lid].y * h))

        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
        for lid in range(33):
            cv2.circle(frame, pts[lid], 3, (0, 255, 0), -1, cv2.LINE_AA)

        # Shoulder Tilt
        draw_vector(frame, pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER], COLORS['shoulder_tilt'], 3)
        mid_sh = ((pts[LEFT_SHOULDER][0] + pts[RIGHT_SHOULDER][0]) // 2,
                  (pts[LEFT_SHOULDER][1] + pts[RIGHT_SHOULDER][1]) // 2)
        cv2.line(frame, (mid_sh[0] - 50, mid_sh[1]), (mid_sh[0] + 50, mid_sh[1]),
                 COLORS['shoulder_tilt'], 1, cv2.LINE_AA)
        cv2.putText(frame, f"shoulder_tilt={calc_shoulder_tilt(pixel_lm):.1f}",
                    (mid_sh[0] - 100, mid_sh[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['shoulder_tilt'], 1, cv2.LINE_AA)

        # Pelvic Tilt
        draw_vector(frame, pts[LEFT_HIP], pts[RIGHT_HIP], COLORS['pelvic_tilt'], 3)
        mid_hip = ((pts[LEFT_HIP][0] + pts[RIGHT_HIP][0]) // 2,
                   (pts[LEFT_HIP][1] + pts[RIGHT_HIP][1]) // 2)
        cv2.line(frame, (mid_hip[0] - 50, mid_hip[1]), (mid_hip[0] + 50, mid_hip[1]),
                 COLORS['pelvic_tilt'], 1, cv2.LINE_AA)
        cv2.putText(frame, f"pelvic_tilt={calc_pelvic_tilt(pixel_lm):.1f}",
                    (mid_hip[0] - 100, mid_hip[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['pelvic_tilt'], 1, cv2.LINE_AA)

        # Trunk Shift
        sm = ((pts[LEFT_SHOULDER][0] + pts[RIGHT_SHOULDER][0]) // 2,
              (pts[LEFT_SHOULDER][1] + pts[RIGHT_SHOULDER][1]) // 2)
        hm = ((pts[LEFT_HIP][0] + pts[RIGHT_HIP][0]) // 2,
              (pts[LEFT_HIP][1] + pts[RIGHT_HIP][1]) // 2)
        draw_vector(frame, hm, sm, COLORS['trunk_shift'], 3)
        cv2.line(frame, (hm[0], hm[1] - 60), (hm[0], hm[1] + 60),
                 COLORS['trunk_shift'], 1, cv2.LINE_AA)
        cv2.putText(frame, f"trunk_shift={calc_trunk_shift(pixel_lm):.1f}",
                    (hm[0] + 15, (hm[1] + sm[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['trunk_shift'], 1, cv2.LINE_AA)

        # Knee Curvature
        draw_angle_arc(frame, pts[RIGHT_KNEE], pts[RIGHT_HIP], pts[RIGHT_ANKLE],
                       COLORS['knee'], f"L knee={calc_left_knee_curvature(pixel_lm):.0f}")
        draw_angle_arc(frame, pts[LEFT_KNEE], pts[LEFT_HIP], pts[LEFT_ANKLE],
                       COLORS['knee'], f"R knee={calc_right_knee_curvature(pixel_lm):.0f}")

        # Distance Between Feet
        draw_vector(frame, pts[LEFT_ANKLE], pts[RIGHT_ANKLE], COLORS['feet'], 3)
        mid_ank = ((pts[LEFT_ANKLE][0] + pts[RIGHT_ANKLE][0]) // 2,
                   (pts[LEFT_ANKLE][1] + pts[RIGHT_ANKLE][1]) // 2)
        cv2.putText(frame, f"feet_dist={calc_distance_between_feet(pixel_lm):.2f}",
                    (mid_ank[0] - 50, mid_ank[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['feet'], 1, cv2.LINE_AA)

        # Arm Curvature
        draw_angle_arc(frame, pts[RIGHT_ELBOW], pts[RIGHT_WRIST], pts[RIGHT_SHOULDER],
                       COLORS['arm_curvature'], f"L arm curv={calc_left_arm_curvature(pixel_lm):.0f}")
        draw_angle_arc(frame, pts[LEFT_ELBOW], pts[LEFT_WRIST], pts[LEFT_SHOULDER],
                       COLORS['arm_curvature'], f"R arm curv={calc_right_arm_curvature(pixel_lm):.0f}")

        # Arm Angle
        cv2.line(frame, (pts[RIGHT_SHOULDER][0] - 30, pts[RIGHT_SHOULDER][1]),
                 (pts[RIGHT_SHOULDER][0] + 30, pts[RIGHT_SHOULDER][1]),
                 COLORS['arm_angle'], 1, cv2.LINE_AA)
        draw_vector(frame, pts[RIGHT_SHOULDER], pts[RIGHT_ELBOW], COLORS['arm_angle'], 2)
        mid_l = ((pts[RIGHT_SHOULDER][0] + pts[RIGHT_ELBOW][0]) // 2,
                 (pts[RIGHT_SHOULDER][1] + pts[RIGHT_ELBOW][1]) // 2)
        cv2.putText(frame, f"L arm ang={calc_left_arm_angle(pixel_lm):.1f}",
                    (mid_l[0] + 10, mid_l[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['arm_angle'], 1, cv2.LINE_AA)

        cv2.line(frame, (pts[LEFT_SHOULDER][0] - 30, pts[LEFT_SHOULDER][1]),
                 (pts[LEFT_SHOULDER][0] + 30, pts[LEFT_SHOULDER][1]),
                 COLORS['arm_angle'], 1, cv2.LINE_AA)
        draw_vector(frame, pts[LEFT_SHOULDER], pts[LEFT_ELBOW], COLORS['arm_angle'], 2)
        mid_r = ((pts[LEFT_SHOULDER][0] + pts[LEFT_ELBOW][0]) // 2,
                 (pts[LEFT_SHOULDER][1] + pts[LEFT_ELBOW][1]) // 2)
        cv2.putText(frame, f"R arm ang={calc_right_arm_angle(pixel_lm):.1f}",
                    (mid_r[0] + 10, mid_r[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['arm_angle'], 1, cv2.LINE_AA)

        # Arm Misalignment
        draw_vector(frame, pts[LEFT_WRIST], pts[RIGHT_WRIST], COLORS['arm_misalignment'], 2)
        draw_vector(frame, pts[LEFT_ELBOW], pts[RIGHT_ELBOW], COLORS['arm_misalignment'], 2)
        mw = ((pts[LEFT_WRIST][0] + pts[RIGHT_WRIST][0]) // 2,
              (pts[LEFT_WRIST][1] + pts[RIGHT_WRIST][1]) // 2)
        cv2.putText(frame, f"arm_misalign={calc_arm_misalignment(pixel_lm):.1f}",
                    (mw[0] - 70, mw[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['arm_misalignment'], 1, cv2.LINE_AA)

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
