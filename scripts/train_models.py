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

  3. Quality Classifier    — Random Forest that predicts severity_label
                             (Normal vs Bad) from the aggregated stats.
                             → models/quality_classifier.pkl

Uses training_data_aggregated.csv (one row per video).
Uses all columns derived from the 5 biomechanical parameters.
"""

import os
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
    # MODEL 3 — Quality Classifier (5-fold CV)
    # =====================================================================
    print("=" * 60)
    print("MODEL 3: QUALITY CLASSIFIER — 5-Fold Cross-Validation")
    print("=" * 60)

    quality_labels = sorted(set(y_severity))
    accs, y_trues, y_preds = cross_validate_rf(X, y_severity, N_FOLDS)

    print(f"  Per-fold accuracies: {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean accuracy:        {np.mean(accs):.4f} "
          f"({np.mean(accs) * 100:.2f}%)")
    print(f"  Std deviation:        {np.std(accs):.4f}")

    print_confusion_matrix(y_trues[-1], y_preds[-1], quality_labels,
                           title=f"Fold {len(accs)}")

    print()
    print("  Classification Report (all folds):")
    y_true_all = np.concatenate(y_trues)
    y_pred_all = np.concatenate(y_preds)
    print(classification_report(y_true_all, y_pred_all, zero_division=0))

    # Train final model on ALL data
    quality_clf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1,
    )
    quality_clf.fit(X, y_severity)

    print("  Top 10 feature importances (final model):")
    imps = sorted(zip(feature_cols, quality_clf.feature_importances_),
                  key=lambda x: x[1], reverse=True)
    for name, imp in imps[:10]:
        print(f"    {name:35s} importance = {imp:.4f}")

    save_model(quality_clf, 'quality_classifier.pkl')
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
        path = os.path.join(MODEL_DIR, f'isolation_forest_{move}.pkl')
        if os.path.exists(path):
            print(f"  models/isolation_forest_{move}.pkl")
    print(f"  models/quality_classifier.pkl")
    print()


if __name__ == '__main__':
    main()
