"""
predict_video.py

Run the trained models on a single video and print a detailed verdict.

Usage:
  python scripts/predict_video.py path/to/video.mp4
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cheer_ai.pipeline import (
    analyze_video, MOVE_NAMES, load_models, download_model_if_needed,
)


def predict_video(video_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models')

    result = analyze_video(video_path, model_dir)

    if result['predicted_move'] == 'ERROR':
        print(f"\nERROR: {result['top_deviations']}")
        sys.exit(1)

    video_name = result['filename']
    move_pred = result['predicted_move']
    quality_pred = result['predicted_quality']
    quality_score = result['quality_score']
    anomaly_score = result['anomaly_score']

    print("=" * 60)
    print(f"RESULTS: {video_name}")
    print("=" * 60)
    print(f"\n  PREDICTED MOVE:     {move_pred}  "
          f"(confidence: {result['predicted_move_confidence']:.1%})")
    print(f"  PREDICTED QUALITY:  {quality_pred}  "
          f"(confidence: {result['predicted_quality_confidence']:.1%})")
    score_bar = '🟢' if quality_score >= 70 else ('🟡' if quality_score >= 40 else '🔴')
    print(f"  QUALITY SCORE:      {quality_score}/100  {score_bar}  "
          f"(anomaly: {anomaly_score:.2f})")
    print(f"  FRAMES:             {result['frames_processed']} processed"
          + (f", {result['frames_skipped']} skipped" if result['frames_skipped'] else ""))

    print("\n  Move probabilities:")
    for name, prob in result['move_probabilities'].items():
        bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
        marker = ' ←' if name == move_pred else ''
        print(f"    {name:12s}  {prob:.1%}  {bar}{marker}")

    print("\n  Quality probabilities:")
    for name, prob in result['quality_probabilities'].items():
        bar = '█' * int(prob * 40) + '░' * (40 - int(prob * 40))
        marker = ' ←' if name == quality_pred else ''
        print(f"    {name:12s}  {prob:.1%}  {bar}{marker}")

    # ---- Top deviations from Normal ----------------------------------
    deviations = result.get('top_deviations_detail', [])
    if deviations:
        print(f"\n  🔴 Top Deviations from Normal ({move_pred}):")
        print(f"     {'Feature':<38s} {'Value':>8s}  {'Normal':>14s}  {'Z':>6s}")
        print(f"     {'─'*38}  {'─'*8}  {'─'*14}  {'─'*6}")
        for d in deviations[:5]:
            normal_str = f"{d['normal_mean']:.2f} ± {d['normal_std']:.2f}"
            marker = '🔴' if d['z_score'] >= 3 else ('🟡' if d['z_score'] >= 2 else '  ')
            print(f"  {marker}  {d['feature']:<36s} {d['value']:8.2f}  "
                  f"{normal_str:>14s}  {d['z_score']:5.1f}σ")

    print(f"\n{'='*60}")
    e = "✅" if quality_pred == "Normal" and anomaly_score < 0.5 else "⚠️"
    print(f"  {e} Move: {move_pred} | Quality: {quality_pred} | Score: {quality_score}/100")
    print("=" * 60)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_video.py <path_to_video>")
        sys.exit(1)
    predict_video(sys.argv[1])
