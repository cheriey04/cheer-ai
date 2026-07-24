"""
batch_predict.py

Run the trained models on all videos in test_videos/ and output
a CSV with predictions, confidences, anomaly scores, and top deviations.

Usage:
  python scripts/batch_predict.py
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheer_ai.pipeline import analyze_video


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'test_videos')
    model_dir = os.path.join(base_dir, 'models')

    video_files = sorted([
        f for f in os.listdir(test_dir) if f.endswith('.mp4')
    ])
    if not video_files:
        print(f"No .mp4 files found in {test_dir}")
        sys.exit(1)

    print(f"Found {len(video_files)} test videos\n")

    results = []
    for i, vf in enumerate(video_files):
        vpath = os.path.join(test_dir, vf)
        print(f"[{i+1}/{len(video_files)}] {vf} ...", end=' ', flush=True)
        result = analyze_video(vpath, model_dir)
        results.append(result)
        print(f"→ {result['predicted_move']} | "
              f"{result['predicted_quality']} | "
              f"anomaly={result['anomaly_score']:.2f}")

    csv_path = os.path.join(base_dir, 'test_videos_predictions.csv')
    fieldnames = [
        'filename', 'predicted_move', 'predicted_move_confidence',
        'predicted_quality', 'predicted_quality_confidence',
        'anomaly_score', 'top_deviations',
        'actual_move', 'actual_quality',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results written to {csv_path}")
    print(f"   {len(results)} videos processed")


if __name__ == '__main__':
    main()
