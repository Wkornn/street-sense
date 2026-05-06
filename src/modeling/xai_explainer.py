"""
Explainable AI (XAI) — Risk Interpretation
src/modeling/xai_explainer.py

Calculates SHAP values for road segments and optionally generates
natural language narratives using the LLM Narrator module.

Usage:
  python src/modeling/xai_explainer.py --segment_id 103470 --narrative
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
import shap
import yaml
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Automatically add project root to sys.path to fix "ModuleNotFoundError: No module named 'src'"
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

# Local Imports
from src.modeling.narrator import generate_explanation

# Config Paths
DATA_CFG_PATH  = "configs/data_sources.yaml"
MODEL_CFG_PATH = "configs/model_params.yaml"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def explain_segment(segment_id, df, calibrated_model, features, top_k=5):
    """
    Calculates SHAP values for a given segment and returns the top_k factors.
    Used by both the CLI and the Streamlit Dashboard.
    """
    segment_data = df[df['segment_id'] == segment_id]
    if segment_data.empty:
        raise ValueError(f"Segment ID {segment_id} not found in dataset.")
        
    X_segment = segment_data[features]
    risk_score = calibrated_model.predict_proba(X_segment)[0, 1]
    
    # Extract base estimator from CalibratedClassifierCV
    base_model = calibrated_model.calibrated_classifiers_[0].estimator
    
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X_segment)
    
    # Map features to impacts
    impacts = dict(zip(features, shap_values[0]))
    sorted_impacts = dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True))
    
    top_factors = {}
    for i, (k, v) in enumerate(sorted_impacts.items()):
        if i >= top_k: break
        top_factors[k] = v
        
    return risk_score, top_factors

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_id", type=int, required=True, help="Road segment ID to explain")
    parser.add_argument("--version", choices=["v1", "v2", "bi"], default="bi", help="Feature version to use (v1, v2, or bi for latest)")
    parser.add_argument("--narrative", action=argparse.BooleanOptionalAction, 
                        help="Enable/disable LLM narrative (overrides config)")
    args = parser.parse_args()

    # 1. Load Configurations
    data_cfg = load_yaml(DATA_CFG_PATH)
    model_cfg = load_yaml(MODEL_CFG_PATH)
    
    # Determine narrative setting: Arg > Config > Default(False)
    expl_cfg = model_cfg.get("explanation", {})
    should_narrate = args.narrative if args.narrative is not None else expl_cfg.get("enable_narrative", False)
    top_k = expl_cfg.get("top_k_features", 5)
    llm_model = expl_cfg.get("llm_model", "gemini-2.5-flash")

    # 2. Load Dataset
    features_dir = data_cfg["features"]["output_dir"]
    data_path = os.path.join(features_dir, "model_dataset.parquet")
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Re-create engineering features to match training pipeline
    if "exposure" not in df.columns:
        df["exposure"] = df["probe_count"].fillna(0) * df["length_m"]
        df["log_exposure"] = np.log1p(df["exposure"])

    if args.segment_id not in df['segment_id'].values:
        print(f"Error: Segment ID {args.segment_id} not found in dataset.")
        return
        
    segment_data = df[df['segment_id'] == args.segment_id]
    
    # 3. Load Model
    if args.version == "bi":
        model_path = "models/xgboost_bi_classification.pkl"
    else:
        model_path = f"models/xgboost_{args.version}_xgboost.pkl"

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        calibrated_model = pickle.load(f)
        
    # Get features used by model
    try:
        # Try to get features from the model first
        features = calibrated_model.calibrated_classifiers_[0].estimator.feature_names_in_.tolist()
    except:
        # Fallback to config
        feat_key = "features" if args.version == "v1" else "features_v2"
        features = [f for f in model_cfg["modeling"].get(feat_key, []) if f in df.columns]
    
    if not features:
        print("Error: Could not determine features for the model.")
        return
        
    X_segment = segment_data[features]
    
    # Get risk prediction
    risk_score = calibrated_model.predict_proba(X_segment)[0, 1]
    print(f"\n" + "="*40)
    print(f" RISK ANALYSIS FOR SEGMENT {args.segment_id}")
    print(f" Predicted Risk: {risk_score*100:.1f}%")
    print("="*40)

    # 4. Calculate SHAP Values
    print("\nCalculating SHAP values (local explanation)...")
    try:
        risk_score, top_factors = explain_segment(args.segment_id, df, calibrated_model, features, top_k)
    except ValueError as e:
        print(e)
        return
        
    print(f"\nTop {top_k} Factors driving this prediction:")
    for i, (k, v) in enumerate(top_factors.items()):
        direction = "↑ Increases Risk" if v > 0 else "↓ Decreases Risk"
        print(f"  {i+1}. {k:<25}: {v:>8.4f} ({direction})")

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
