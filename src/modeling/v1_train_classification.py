"""
Production-Level Risk Scoring Training Script
src/modeling/train_classification.py

Strategy: Probability-based Risk Scoring (not hard classification)
1. Train XGBoost/RF with scale_pos_weight to handle imbalance (no SMOTE)
2. Calibrate probabilities with Isotonic Regression (makes scores meaningful)
3. Evaluate with Log Loss, Brier Score, AUC-ROC, PR-AUC (not accuracy)
4. Output: risk_score per segment (0.0–1.0) saved to parquet

Usage:
  python src/models/train.py              # v1: full features (with history)
  python src/models/train.py --version v2 # v2: no accident history
"""

import os
import argparse
import yaml
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

# Try to import MLFlow (Optional)
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Config Paths
DATA_CFG_PATH  = "configs/data_sources.yaml"
MODEL_CFG_PATH = "configs/model_params.yaml"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(y_test, y_prob_uncal, y_prob_cal):
    """
    Evaluate using probability-focused metrics only.
    Compares uncalibrated vs calibrated scores.

    Metrics:
      - AUC-ROC      : Ranking quality (higher = better)
      - PR-AUC       : Precision-Recall for rare class (higher = better)
      - Log Loss     : Penalizes confident wrong predictions (lower = better)
      - Brier Score  : Mean squared error of probabilities (lower = better)
    """
    metrics = {}

    print("\n" + "=" * 60)
    print("  RISK SCORING EVALUATION")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Uncalibrated':>14} {'Calibrated':>14}")
    print(f"  {'-'*48}")

    for label, y_prob in [("Uncalibrated", y_prob_uncal), ("Calibrated", y_prob_cal)]:
        roc  = roc_auc_score(y_test, y_prob)
        pr   = average_precision_score(y_test, y_prob)
        ll   = log_loss(y_test, y_prob)
        bs   = brier_score_loss(y_test, y_prob)
        if label == "Calibrated":
            metrics = {"roc_auc": roc, "pr_auc": pr,
                       "log_loss": ll, "brier_score": bs}

    # Print side by side
    roc_u  = roc_auc_score(y_test, y_prob_uncal)
    pr_u   = average_precision_score(y_test, y_prob_uncal)
    ll_u   = log_loss(y_test, y_prob_uncal)
    bs_u   = brier_score_loss(y_test, y_prob_uncal)

    roc_c  = roc_auc_score(y_test, y_prob_cal)
    pr_c   = average_precision_score(y_test, y_prob_cal)
    ll_c   = log_loss(y_test, y_prob_cal)
    bs_c   = brier_score_loss(y_test, y_prob_cal)

    print(f"  {'AUC-ROC':<20} {roc_u:>14.4f} {roc_c:>14.4f}  ↑ higher is better")
    print(f"  {'PR-AUC':<20} {pr_u:>14.4f} {pr_c:>14.4f}  ↑ higher is better")
    print(f"  {'Log Loss':<20} {ll_u:>14.4f} {ll_c:>14.4f}  ↓ lower is better")
    print(f"  {'Brier Score':<20} {bs_u:>14.4f} {bs_c:>14.4f}  ↓ lower is better")
    print("=" * 60)

    print("\n  Interpretation:")
    print(f"  • AUC-ROC {roc_c:.2f}: Model ranks risky roads above safe roads "
          f"{'well' if roc_c > 0.8 else 'moderately'}.")
    print(f"  • Brier Score {bs_c:.4f}: "
          f"{'Well-calibrated probabilities.' if bs_c < 0.05 else 'Scores may need refinement.'}")

    return metrics


def build_model(model_type, h_params, scale_pos_weight):
    """Instantiate the base model from config (before calibration)."""
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=h_params.get("n_estimators", 100),
            learning_rate=h_params.get("learning_rate", 0.1),
            max_depth=h_params.get("max_depth", 6),
            scale_pos_weight=scale_pos_weight,
            random_state=h_params.get("random_state", 42),
            eval_metric="logloss"
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=h_params.get("n_estimators", 100),
            max_depth=h_params.get("max_depth", 10),
            class_weight="balanced",
            random_state=h_params.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. "
                         "Choose 'xgboost' or 'random_forest'.")


def save_model(model, model_type, version_tag):
    """Save model as pickle to local /models/ directory."""
    os.makedirs("models", exist_ok=True)
    path = os.path.join("models", f"{model_type}_{version_tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {path}")
    return path


def save_risk_scores(model_cal, X, segment_ids, version_tag):
    """
    Generate risk scores (calibrated probabilities) for all 277k segments
    and save to parquet for the dashboard/map.
    """
    print("\nGenerating risk scores for all segments...")
    risk_scores = model_cal.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "segment_id": segment_ids.values,
        "risk_score": risk_scores,
        "risk_pct":   (risk_scores * 100).round(1),   # Human readable: 0–100%
    })
    results = results.sort_values("risk_score", ascending=False)

    os.makedirs("data/processed/results", exist_ok=True)
    out_path = f"data/processed/results/risk_scores_{version_tag}.parquet"
    results.to_parquet(out_path, index=False)
    print(f"Risk scores saved → {out_path}")

    # Quick sanity check
    print(f"\n  Top 5 highest-risk segments:")
    print(results.head(5)[["segment_id", "risk_pct"]].to_string(index=False))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2"], default="v2",
                        help="v1=full features (with accident history), "
                             "v2=no accident history (for new roads)")
    parser.add_argument("--calibration", choices=["isotonic", "sigmoid"], default="isotonic",
                        help="Calibration method: isotonic (default) or sigmoid (Platt Scaling)")
    args = parser.parse_args()

    # 1. Load Configurations
    data_cfg  = load_yaml(DATA_CFG_PATH)
    model_cfg = load_yaml(MODEL_CFG_PATH)

    m_params   = model_cfg["modeling"]
    h_params   = m_params["hyperparameters"]
    model_type = m_params.get("model_type", "xgboost")
    target     = m_params["target"]
    exp_name   = m_params.get("experiment_name", "Bangkok_Accident_Risk")

    # Select feature version
    feat_key = "features" if args.version == "v1" else "features_v2"
    features = m_params[feat_key]
    version_tag = f"{args.version}_{model_type}"

    print(f"\n{'='*60}")
    print(f"  H-Spot Bangkok — Risk Score Training")
    print(f"{'='*60}")
    print(f"  Model    : {model_type.upper()}")
    print(f"  Features : {args.version} ({len(features)} features)")
    print(f"  Calibration: {args.calibration}")
    print(f"{'='*60}")

    # 2. Load dataset
    output_dir = data_cfg["features"]["output_dir"]
    data_path  = os.path.join(output_dir, "model_dataset.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run preprocess.py first.")

    df = pd.read_parquet(data_path)
    features = [f for f in features if f in df.columns]
    missing  = [f for f in m_params[feat_key] if f not in df.columns]
    if missing:
        print(f"\n  ⚠️  Features not in dataset (skipped): {missing}")

    X = df[features]
    y = df[target]
    segment_ids = df["segment_id"] if "segment_id" in df.columns else pd.Series(df.index)

    # 3. Stratified Train/Test Split
    # Use 60/20/20: train / calibration / test
    X_train_cal, X_test, y_train_cal, y_test = train_test_split(
        X, y, test_size=0.2, random_state=h_params.get("random_state", 42), stratify=y
    )
    print(f"\n  Train+Cal: {len(X_train_cal):,} | Test: {len(X_test):,}")
    print(f"  Risky in test: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")

    # 4. Build base model
    ratio = (y == 0).sum() / (y == 1).sum()
    print(f"\n  Class imbalance: {ratio:.0f}:1 → scale_pos_weight={ratio:.0f}")
    base_model = build_model(model_type, h_params, scale_pos_weight=ratio)

    # 5. Calibrate using CalibratedClassifierCV
    # cv=3 means it does 3-fold internal cross-validation for calibration
    print(f"\n  Training base model + calibrating with {args.calibration}...")
    calibrated_model = CalibratedClassifierCV(
        base_model, method=args.calibration, cv=3
    )
    calibrated_model.fit(X_train_cal, y_train_cal)

    # Also train uncalibrated version for comparison
    base_model.fit(X_train_cal, y_train_cal)

    # 6. Predict on test set
    y_prob_uncal = base_model.predict_proba(X_test)[:, 1]
    y_prob_cal   = calibrated_model.predict_proba(X_test)[:, 1]

    # 7. Evaluate (probability metrics only)
    metrics = evaluate(y_test, y_prob_uncal, y_prob_cal)

    # 8. Log to MLFlow & Save
    if HAS_MLFLOW:
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name=f"{version_tag}_{args.calibration}"):
            mlflow.log_params(h_params)
            mlflow.log_param("model_type",       model_type)
            mlflow.log_param("feature_version",  args.version)
            mlflow.log_param("num_features",     len(features))
            mlflow.log_param("class_ratio",      round(ratio, 1))
            mlflow.log_param("calibration",      args.calibration)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(calibrated_model, "calibrated_model")
    else:
        print("\n  (MLFlow not detected — skipping experiment logging)")

    save_model(calibrated_model, model_type, version_tag)
    save_risk_scores(calibrated_model, X, segment_ids, version_tag)


if __name__ == "__main__":
    main()
