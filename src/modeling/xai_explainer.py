"""
Explainable AI (XAI) — Risk Interpretation
src/modeling/xai_explainer.py

Calculates SHAP values for road segments and optionally generates
natural language narratives using the LLM Narrator module.

Usage:
  python src/modeling/xai_explainer.py --segment_id 103470 --narrative
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Automatically add project root to sys.path to fix "ModuleNotFoundError: No module named 'src'"
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

# Local Imports
from src.modeling import data
from src.modeling.models import load_model
from src.modeling.narrator import generate_explanation

# Config Paths
DATA_CFG_PATH = "configs/data_sources.yaml"
APP_CFG_PATH = "configs/app.yaml"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def explain_segment(segment_id, df, model, features, top_k=5):
    """
    Risk score and the top_k SHAP factors for one segment.
    Used by both the CLI and the Streamlit Dashboard.

    df must already contain derived columns (data.add_exposure).
    """
    segment_data = df[df["segment_id"] == segment_id]
    if segment_data.empty:
        raise ValueError(f"Segment ID {segment_id} not found in dataset.")

    # Same preparation as training: missing values -> 0, error on missing columns
    X_segment = data.select_features(segment_data, features)
    risk_score = float(model.predict_risk(X_segment)[0])

    # Each model explains itself (XGBoost: TreeSHAP, NB: coefficient contributions)
    contributions = model.explain(X_segment)[0]

    impacts = sorted(zip(features, contributions), key=lambda kv: abs(kv[1]), reverse=True)
    top_factors = {name: float(value) for name, value in impacts[:top_k]}
    return risk_score, top_factors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_id", type=int, required=True, help="Road segment ID to explain")
    parser.add_argument("--narrative", action=argparse.BooleanOptionalAction,
                        help="Enable/disable LLM narrative (overrides config)")
    args = parser.parse_args()

    # 1. Load Configurations
    data_cfg = load_yaml(DATA_CFG_PATH)
    app_cfg = load_yaml(APP_CFG_PATH)

    # Narrative setting: Arg > Config > Default(False)
    expl_cfg = app_cfg.get("explanation", {})
    should_narrate = args.narrative if args.narrative is not None else expl_cfg.get("enable_narrative", False)
    top_k = expl_cfg.get("top_k_features", 5)
    llm_model = expl_cfg.get("llm_model", "gemini-2.5-flash-lite")

    # 2. Load Dataset (same derived columns as training)
    data_path = os.path.join(data_cfg["features"]["output_dir"], "model_dataset.parquet")
    print(f"Loading dataset from {data_path}...")
    df = data.add_exposure(pd.read_parquet(data_path))

    # 3. Load Model; features come from its meta.json, written at training time
    model_dir = Path(app_cfg["model_dir"])
    print(f"Loading model from {model_dir}...")
    model = load_model(model_dir)
    features = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))["features"]

    # 4. Risk + SHAP
    try:
        risk_score, top_factors = explain_segment(args.segment_id, df, model, features, top_k)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\n" + "=" * 40)
    print(f" RISK ANALYSIS FOR SEGMENT {args.segment_id}")
    print(f" Predicted Risk: {risk_score * 100:.1f}%")
    print("=" * 40)

    print(f"\nTop {top_k} Factors driving this prediction:")
    for i, (k, v) in enumerate(top_factors.items()):
        direction = "↑ Increases Risk" if v > 0 else "↓ Decreases Risk"
        print(f"  {i + 1}. {k:<25}: {v:>8.4f} ({direction})")

    # 5. Optional Narrative Generation
    if should_narrate:
        print(f"\nGenerating narrative using {llm_model}...")
        narrative = generate_explanation(args.segment_id, risk_score, top_factors, llm_model)
        if narrative:
            print("\n--- AI Narrative Explanation ---")
            print(narrative)
            print("-" * 32)
    else:
        print("\n[Note] Narrative generation is disabled (use --narrative to enable).")


if __name__ == "__main__":
    main()
