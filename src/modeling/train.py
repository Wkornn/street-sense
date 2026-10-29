"""Training entrypoint. One config file = one run.

    python -m src.modeling.train --config configs/train/xgb_severe.yaml

Flow: config -> data -> target -> features -> split
      -> plugin.fit (model-specific) -> evaluate -> plugin.save + meta + risk scores

Everything a run produces goes into output.model_dir, so the app can switch
models by changing one line (model_dir) in configs/app.yaml.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.modeling import data
from src.modeling.config import load_config
from src.modeling.evaluate import evaluate
from src.modeling.models import get_plugin


def run(cfg):
    # 1. Data (shared by every approach)
    df = data.load_dataset(cfg["data_path"])
    df = data.add_exposure(df)
    X = data.select_features(df, cfg["features"])
    y = data.make_target(df, cfg["target"])
    X_train, X_cal, X_test, y_train, y_cal, y_test = data.split(X, y, cfg["split"])
    print(f"Run '{cfg['name']}': {X.shape[1]} features | "
          f"train {len(X_train):,} / cal {len(X_cal):,} / test {len(X_test):,}")

    # 2. Fit (everything model-specific happens inside the plugin)
    plugin = get_plugin(cfg["model"]["type"])
    model = plugin.fit(cfg["model"], X_train, y_train, X_cal, y_cal)

    # 3. Evaluate on the test split only
    target_type = cfg["target"].get("type", "binary")
    metrics = evaluate(target_type, y_test, model, X_test)
    print(json.dumps(metrics, indent=2))

    # 4. Save: model files (plugin) + run info (same for every approach)
    out_dir = Path(cfg["output"]["model_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plugin.save(model, out_dir)
    meta = {
        "name": cfg["name"],
        "model_type": cfg["model"]["type"],
        "target": cfg["target"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": cfg["features"],
        "metrics": metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 5. Score every segment (risk = probability of >= 1 event; counts too if the model has them)
    scores = df[["segment_id"]].assign(risk_score=model.predict_risk(X))
    if hasattr(model, "predict_count"):
        scores["expected_count"] = model.predict_count(X)
    scores.to_parquet(out_dir / "risk_scores.parquet", index=False)

    # Optional extra copy, e.g. for notebooks that read a fixed path
    extra = cfg["output"].get("risk_scores")
    if extra:
        Path(extra).parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(extra, index=False)
    print(f"Saved model + risk scores -> {out_dir}/" + (f"  (copy: {extra})" if extra else ""))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train one model from a config file")
    parser.add_argument("--config", required=True, help="Path to configs/train/<run>.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
