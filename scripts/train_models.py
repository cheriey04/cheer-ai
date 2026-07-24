"""
train_models.py

Trains three types of models from the aggregated pose features
using 5-fold cross-validation:

  1. Move Classifier       — Random Forest that predicts which cheer move
                             (standing, high_v, liberty, t_jump, tuck_jump)
                             from the aggregated biomechanical stats.
                             → models/move_classifier.pkl

  2. Isolation Forests     — One per move, trained on "Normal" videos only.
                             Flags outlier videos that may indicate bad form.
                             → models/isolation_forest_[move_name].pkl

  3. Per-Move Quality      — One Random Forest per move, predicts Normal vs Bad.
                             Uses feature selection (5 or 7 params) and
                             complexity control based on sample size.
                             → models/quality_[move_name].pkl

Uses training_data_aggregated.csv (one row per video).
Uses all columns derived from the 5 biomechanical parameters.
"""

import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = 'data/processed_features/training_data_aggregated.csv'
MODEL_DIR = 'models'

META_COLS = ['video_filename', 'move_name', 'severity_label']

MOVE_NAMES = ['standing', 'high_v', 'liberty', 't_jump', 'tuck_jump']

N_FOLDS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_model_dir():
    """Create the models/ directory if it doesn't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(model, filename: str):
    """Save a trained model to models/ using pickle."""
    path = os.path.join(MODEL_DIR, filename)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")


def cross_validate_rf(X, y, n_folds: int):
    """Run k-fold cross-validation on a RandomForestClassifier.

    Returns:
        accuracies  — list of per-fold accuracy scores
        y_trues_all — list of true label arrays per fold
        y_preds_all — list of predicted label arrays per fold
    """
    accuracies = []
    y_trues_all = []
    y_preds_all = []

    # Stratified splits when possible; fall back to regular KFold
    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = skf.split(X, y)
    except ValueError:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = kf.split(X)

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        y_trues_all.append(y_test)
        y_preds_all.append(y_pred)

    return accuracies, y_trues_all, y_preds_all


def print_confusion_matrix(y_true, y_pred, label_names: list, title: str):
    """Print a formatted confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    print(f"\n  Confusion Matrix ({title}):")
    header = " " * 12 + "".join(f"{n:>8s}" for n in label_names)
    print(header)
    for i, name in enumerate(label_names):
        row = "".join(f"{cm[i, j]:>8d}" for j in range(len(label_names)))
        print(f"  {name:10s}{row}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_model_dir()
    warnings.filterwarnings('ignore', category=UserWarning)

    # ---- Load data -------------------------------------------------------
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    df = pd.read_csv(CSV_PATH)
    print(f"  Rows (one per video): {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Moves: {df['move_name'].value_counts().to_dict()}")
    print(f"  Severity: {df['severity_label'].value_counts().to_dict()}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"  Features: {len(feature_cols)} (8 stats × 5 params)")
    print()

    X = df[feature_cols].to_numpy()
    y_move = df['move_name'].to_numpy()
    y_severity = df['severity_label'].to_numpy()

    # =====================================================================
    # MODEL 1 — Move Classifier (5-fold CV)
    # =====================================================================
    print("=" * 60)
    print("MODEL 1: MOVE CLASSIFIER — 5-Fold Cross-Validation")
    print("=" * 60)

    move_labels = sorted(set(y_move))
    accs, y_trues, y_preds = cross_validate_rf(X, y_move, N_FOLDS)

    print(f"  Per-fold accuracies: {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean accuracy:        {np.mean(accs):.4f} "
          f"({np.mean(accs) * 100:.2f}%)")
    print(f"  Std deviation:        {np.std(accs):.4f}")

    # Confusion matrix from the last fold
    print_confusion_matrix(y_trues[-1], y_preds[-1], move_labels,
                           title=f"Fold {len(accs)}")

    # Classification report across all folds
    print()
    print("  Classification Report (all folds):")
    y_true_all = np.concatenate(y_trues)
    y_pred_all = np.concatenate(y_preds)
    print(classification_report(y_true_all, y_pred_all, zero_division=0))

    # Train final model on ALL data
    move_clf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1,
    )
    move_clf.fit(X, y_move)

    print("  Top 10 feature importances (final model):")
    imps = sorted(zip(feature_cols, move_clf.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    for name, imp in imps[:10]:
        print(f"    {name:35s} importance = {imp:.4f}")

    save_model(move_clf, 'move_classifier.pkl')
    print()

    # =====================================================================
    # MODEL 2 — Isolation Forests (per-move, Normal-only)
    # =====================================================================
    print("=" * 60)
    print("MODEL 2: ISOLATION FORESTS (per-move, Normal-only)")
    print("=" * 60)
    print()

    for move in MOVE_NAMES:
        mask = (df['move_name'] == move) & (df['severity_label'] == 'Normal')
        X_normal = df.loc[mask, feature_cols].to_numpy()

        if len(X_normal) < 3:
            print(f"  {move:12s}  only {len(X_normal)} Normal videos — skipping")
            # Rename any old per-frame model so it doesn't conflict
            old = os.path.join(MODEL_DIR, f'isolation_forest_{move}.pkl')
            legacy = os.path.join(MODEL_DIR, f'isolation_forest_{move}_legacy_perframe.pkl')
            if os.path.exists(old):
                os.rename(old, legacy)
                print(f"    (renamed old 5-feature model → {legacy})")
            continue

        iso_forest = IsolationForest(
            n_estimators=100, contamination=0.05,
            random_state=42, n_jobs=-1,
        )
        iso_forest.fit(X_normal)

        preds = iso_forest.predict(X_normal)
        n_in = (preds == 1).sum()
        n_out = (preds == -1).sum()
        print(f"  {move:12s}  Normal videos: {len(X_normal):>5}  "
              f"inliers: {n_in}  outliers: {n_out}")

        save_model(iso_forest, f'isolation_forest_{move}.pkl')

    print()

    # =====================================================================
    # MODEL 3 — Per-Move Quality Classifiers
    # =====================================================================
    print("=" * 60)
    print("MODEL 3: PER-MOVE QUALITY CLASSIFIERS")
    print("=" * 60)
    print("  Trains a separate Normal-vs-Bad classifier for each move.")
    print("  Uses feature selection + complexity control per move.")
    print("  Also saves Normal averages for error localization.")
    print()

    # Core 5 params (40 features) — used for data-starved moves
    CORE_PARAMS = {'shoulder_tilt', 'pelvic_tilt', 'trunk_shift',
                   'knee_curvature', 'arm_misalignment'}
    core_feature_cols = [c for c in feature_cols
                         if c.split('_')[0] in CORE_PARAMS
                         or c.split('_')[0] + '_' + c.split('_')[1] in CORE_PARAMS
                         or any(c.startswith(p + '_') for p in CORE_PARAMS)]

    normal_averages = {}  # {move_name: {feature: mean_value}}

    for move in MOVE_NAMES:
        mask = df['move_name'] == move
        df_move = df[mask]
        n_videos = len(df_move)
        n_normal = (df_move['severity_label'] == 'Normal').sum()
        n_bad = (df_move['severity_label'] == 'Bad').sum()

        # All moves use all features (56 → 96 with new params)
        move_feature_cols = feature_cols
        n_feats = len(move_feature_cols)

        # Complexity control: fewer trees + depth limit for small datasets
        use_light_model = n_videos < 15
        n_est = 100 if use_light_model else 200
        max_d = 10 if use_light_model else None

        print(f"  {move:12s}  videos: {n_videos} (Normal: {n_normal}, Bad: {n_bad})"
              f"  |  {n_feats} features"
              f"  |  {'light' if use_light_model else 'full'} model")

        print(f"  {move:12s}  videos: {n_videos}  (Normal: {n_normal}, Bad: {n_bad})")

        if n_normal < 2 or n_bad < 2:
            print(f"    ⚠️  Need ≥2 Normal AND ≥2 Bad — skipping\n")
            continue

        X_move = df_move[move_feature_cols].to_numpy()
        y_move_sev = df_move['severity_label'].to_numpy()

        # Cross-validation
        folds_for_move = min(N_FOLDS, n_videos // 2, n_normal, n_bad)
        folds_for_move = max(2, folds_for_move)

        try:
            accs, y_trues, y_preds = cross_validate_rf(
                X_move, y_move_sev, folds_for_move
            )
        except ValueError:
            print(f"    ⚠️  Not enough data for cross-validation — skipping\n")
            continue

        print(f"    CV folds: {folds_for_move}")
        print(f"    Per-fold accuracies: {[f'{a:.4f}' for a in accs]}")
        print(f"    Mean accuracy:        {np.mean(accs):.4f} "
              f"({np.mean(accs) * 100:.2f}%)")
        print(f"    Std deviation:        {np.std(accs):.4f}")

        yt = np.concatenate(y_trues)
        yp = np.concatenate(y_preds)
        print(f"\n    Classification Report:")
        cr = classification_report(yt, yp, zero_division=0)
        for line in cr.split('\n'):
            if line.strip():
                print(f"    {line}")

        # Train final model with per-move complexity control
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_move, y_move_sev)
        save_model(clf, f'quality_{move}.pkl')

        # Normal averages
        mask_normal = df_move['severity_label'] == 'Normal'
        means = df_move.loc[mask_normal, move_feature_cols].mean().to_dict()
        normal_averages[move] = {k: round(v, 6) for k, v in means.items()}
        print()

    # Save Normal averages to JSON
    if normal_averages:
        avg_path = os.path.join(MODEL_DIR, 'normal_averages.json')
        with open(avg_path, 'w') as f:
            json.dump(normal_averages, f, indent=2)
        print(f"  Saved Normal averages → {avg_path}")
    print()

    # ---- Final summary --------------------------------------------------
    print("=" * 60)
    print("ALL MODELS TRAINED (5-fold CV)")
    print("=" * 60)
    print(f"  Data:           {CSV_PATH} ({len(df)} videos)")
    print(f"  Features:       {len(feature_cols)} aggregated stats")
    print(f"  CV folds:       {N_FOLDS}")
    print()
    print(f"  models/move_classifier.pkl")
    for move in MOVE_NAMES:
        for prefix in ['isolation_forest', 'quality']:
            path = os.path.join(MODEL_DIR, f'{prefix}_{move}.pkl')
            if os.path.exists(path):
                print(f"  models/{prefix}_{move}.pkl")
    if normal_averages:
        print(f"  models/normal_averages.json")
    print()


if __name__ == '__main__':
    main()
