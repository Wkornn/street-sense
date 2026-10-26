"""Training entrypoint. One config file = one run.

    python -m src.modeling.train --config configs/train/xgb_severe.yaml

Flow: config -> data -> target -> features -> split -> model -> fit
      -> calibrate -> evaluate -> save model + metrics + risk scores
"""

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import yaml
from sklearn.calibration import CalibratedClassifierCV

from src.modeling import data
from src.modeling.config import load_config
from src.modeling.evaluate import classification_metrics
from src.modeling.models import build_model

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except ImportError:  # older scikit-learn
    FrozenEstimator = None


def calibrate(model, method, X_cal, y_cal):
    """Fit a calibrator on a held-out split, keeping the already-trained model fixed."""
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    else:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    return calibrated.fit(X_cal, y_cal)


def run(cfg):
    # 1. Data
    df = data.load_dataset(cfg["data_path"])
    df = data.add_exposure(df)
    X = data.select_features(df, cfg["features"])
    y = data.make_target(df, cfg["target"])
    X_train, X_cal, X_test, y_train, y_cal, y_test = data.split(X, y, cfg["split"])
    print(f"Run '{cfg['name']}': {X.shape[1]} features | "
          f"train {len(X_train):,} / cal {len(X_cal):,} / test {len(X_test):,}")

    # 2. Model
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = build_model(cfg["model"], scale_pos_weight=pos_weight)
    model.fit(X_train, y_train)

    # 3. Calibration
    if cfg.get("calibration"):
        model = calibrate(model, cfg["calibration"], X_cal, y_cal)

    # 4. Evaluation (test split only)
    metrics = classification_metrics(y_test, model.predict_proba(X_test)[:, 1])
    print(json.dumps(metrics, indent=2))

    # 5. Save
    out_dir = Path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:  # TODO: native format (step 7)
        pickle.dump(model, f)
    meta = {
        "name": cfg["name"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": cfg["features"],
        "metrics": metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 6. Score every segment
    scores_path = Path(cfg["output"]["risk_scores"])
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores = df[["segment_id"]].assign(risk_score=model.predict_proba(X)[:, 1])
    scores.to_parquet(scores_path, index=False)
    print(f"Saved model -> {out_dir}/  |  risk scores -> {scores_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train one model from a config file")
    parser.add_argument("--config", required=True, help="Path to configs/train/<run>.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
