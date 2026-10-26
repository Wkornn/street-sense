"""
Negative Binomial Regression — Accident Count Model
src/modeling/train_negbin_count.py

Predicts the EXPECTED NUMBER of accidents per road segment.
This is the approach used in the Highway Safety Manual (HSM).

Why Negative Binomial (not Poisson)?
  - Accident counts are overdispersed: Variance >> Mean
  - Poisson assumes Mean = Variance, which is violated in real crash data
  - NB adds a dispersion parameter (alpha) to absorb that extra variance

Why an Exposure Offset?
  - A 500m road has 5x more "opportunity" for accidents than a 100m road
  - Without an offset, longer roads are unfairly penalized
  - We include log(length_m / 100) as the offset so the model predicts
    "accidents per 100m of road" rather than "total accidents"

Output:
  - Expected accident count per segment (e.g., 2.3 accidents per 7 years)
  - Saved to data/processed/results/count_predictions.parquet

Usage:
  pip install statsmodels
  python src/modeling/train_negbin_count.py
"""

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_CFG_PATH  = "configs/data_sources.yaml"
MODEL_CFG_PATH = "configs/model_params.yaml"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def check_overdispersion(y):
    """
    Quick Cameron & Trivedi overdispersion check.
    If Variance >> Mean, Poisson is inappropriate — use Negative Binomial.
    """
    mean = y.mean()
    var  = y.var()
    print(f"\n  Overdispersion Check:")
    print(f"    Mean   : {mean:.4f}")
    print(f"    Variance: {var:.4f}")
    print(f"    Ratio (Var/Mean): {var/mean:.2f}")
    if var / mean > 2:
        print("    → ⚠️  Severely overdispersed. Negative Binomial is required.")
    elif var / mean > 1.2:
        print("    → ⚠️  Mildly overdispersed. NB preferred over Poisson.")
    else:
        print("    → ✅ Not overdispersed. Poisson may be sufficient.")
    return var / mean


def evaluate_count_model(model, X_test_sm, y_test, offset_test, y_train, X_train_sm, off_train):
    """
    Evaluate a count regression model.
    Metrics:
      - MAE / RMSE: How far off are our predicted counts?
      - AIC: Lower = better model fit vs complexity trade-off
      - Deviance: Goodness of fit (lower = better)
      - Pearson chi2/df: Should be ~1.0 for good fit
    """
    y_pred = model.predict(X_test_sm, offset=offset_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    # Compute Pearson chi2 manually (not available as attribute in statsmodels NB)
    y_pred_train = model.predict(X_train_sm, offset=off_train)
    pearson_resid = (y_train.values - y_pred_train) / np.sqrt(y_pred_train + 1e-8)
    pearson_chi2  = (pearson_resid ** 2).sum()
    df_resid      = len(y_train) - len(model.params)
    pearson_ratio = pearson_chi2 / df_resid if df_resid > 0 else float('nan')

    aic = model.aic if not np.isnan(model.aic) else float('nan')
    llf = model.llf if not np.isnan(model.llf) else float('nan')

    print("\n" + "=" * 55)
    print("  COUNT MODEL EVALUATION")
    print("=" * 55)
    print(f"  MAE         : {mae:.4f}  (avg absolute error in accident count)")
    print(f"  RMSE        : {rmse:.4f}  (penalizes large misses more)")
    print(f"  AIC         : {aic:.2f}  (lower = better)" if not np.isnan(aic) else "  AIC         : N/A (optimizer did not fully converge)")
    print(f"  Log-Lik     : {llf:.2f}" if not np.isnan(llf) else "  Log-Lik     : N/A")
    print(f"  Pearson χ²/df: {pearson_ratio:.3f}  (ideal ≈ 1.0)")
    print("=" * 55)

    # Segment-level preview
    preview = pd.DataFrame({
        "actual_acc":   y_test.values[:10],
        "predicted_acc": y_pred[:10].round(2),
    })
    print("\n  Sample Predictions (first 10 test segments):")
    print(preview.to_string(index=False))

    return {
        "mae": float(mae), "rmse": float(rmse),
        "pearson_chi2_df": float(pearson_ratio),
    }


def main():
    # 1. Load Config
    data_cfg  = load_yaml(DATA_CFG_PATH)
    model_cfg = load_yaml(MODEL_CFG_PATH)
    m_params  = model_cfg["modeling"]

    # Count model always uses the "no-history" feature set
    # (the target IS accident count — can't use acc_* as features)
    features = m_params["features_v2"]

    # 2. Load dataset
    output_dir = data_cfg["features"]["output_dir"]
    data_path  = os.path.join(output_dir, "model_dataset.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}. Run preprocess.py first.")

    df = pd.read_parquet(data_path)

    TARGET = "acc_total"   # Count target (0, 1, 2, 3, ...)
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found in dataset.")

    features = [f for f in features if f in df.columns]
    print(f"\n{'='*55}")
    print(f"  H-Spot Bangkok — Negative Binomial Count Model")
    print(f"{'='*55}")
    print(f"  Target   : {TARGET} (accident count per segment)")
    print(f"  Features : {len(features)}")
    print(f"  Segments : {len(df):,}")

    # 3. Overdispersion Check (validates using NB over Poisson)
    check_overdispersion(df[TARGET])

    # 4. Exposure Offset: log(length_m / 100)
    # Normalizes to "accidents per 100m segment"
    df["log_exposure"] = np.log(df["length_m"].clip(lower=1) / 100)

    # Drop features that cause multicollinearity in NB regression:
    # 1. length_m — already used as the exposure offset, can't be a feature too
    # 2. Raw versions when log version exists (poi_count, building_density, probe_count)
    COLLINEAR_DROP = [
        "length_m",          # captured by offset
        "poi_count_200m",    
        "building_density_200m",
        "probe_count",
        "dist_intersection_m",
        "dist_school_m",
        "dist_hospital_m",
        "dist_fuel_m",
        "dist_mall_m"
    ]
    features = [f for f in features if f not in COLLINEAR_DROP]
    print(f"  Features after collinearity removal: {len(features)}")

    X = df[features].fillna(0)
    y = df[TARGET]
    offset = df["log_exposure"]

    segment_ids = df["segment_id"] if "segment_id" in df.columns else pd.Series(df.index)

    # 5. Stratified Train/Test Split
    # We stratify by the BINARY flag (is_risky) to ensure rare accidents 
    # are evenly spread between train and test.
    is_risky = (df[TARGET] > 0).astype(int)
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=is_risky
    )

    X_train_raw = X.iloc[idx_train].reset_index(drop=True)
    X_test_raw  = X.iloc[idx_test].reset_index(drop=True)

    print(f"\n  Train: {len(X_train_raw):,} | Test: {len(X_test_raw):,}")

    # 6. Standardize features for numerical stability
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_raw)
    X_test_sc  = scaler.transform(X_test_raw)
    
    X_train_df = pd.DataFrame(X_train_sc, columns=features)
    X_test_df  = pd.DataFrame(X_test_sc,  columns=features)

    X_train_sm = sm.add_constant(X_train_df)
    X_test_sm  = sm.add_constant(X_test_df)

    off_train  = offset.iloc[idx_train].reset_index(drop=True)
    off_test   = offset.iloc[idx_test].reset_index(drop=True)
    y_train    = y.iloc[idx_train].reset_index(drop=True)
    y_test     = y.iloc[idx_test].reset_index(drop=True)

    # 7. Fit Count Model
    # We use GLM with Negative Binomial family as it's often more stable than the discrete version.
    print("\n  Fitting Negative Binomial regression (via GLM)...")
    
    # statsmodels GLM NegativeBinomial needs an alpha. 
    # We'll use a standard starting point or estimate it.
    try:
        # First fit a Poisson model to get good starting values
        print("    1. Fitting Poisson baseline for initialization...")
        poisson_model = sm.GLM(y_train, X_train_sm, family=sm.families.Poisson(), offset=off_train)
        poisson_results = poisson_model.fit()
        
        print("    2. Fitting Negative Binomial for overdispersion...")
        # Use the Poisson results as starting values (optional but helpful)
        nb_family = sm.families.NegativeBinomial(alpha=1.0) # alpha=1.0 is a common default
        nb_model = sm.GLM(y_train, X_train_sm, family=nb_family, offset=off_train)
        result = nb_model.fit()
        
    except Exception as e:
        print(f"  ⚠️  GLM fitting failed: {e}. Falling back to Poisson.")
        nb_family = sm.families.Poisson()
        nb_model = sm.GLM(y_train, X_train_sm, family=nb_family, offset=off_train)
        result = nb_model.fit()

    print("\n  Model Summary (Top coefficients):")
    print(result.summary().tables[1])

    # 8. Evaluate
    metrics = evaluate_count_model(result, X_test_sm, y_test, off_test, y_train, X_train_sm, off_train)

    # 8. Save predictions for all segments
    print("\n  Generating expected accident counts for all segments...")
    X_all_sc = scaler.transform(X[features].fillna(0))
    X_all_df = pd.DataFrame(X_all_sc, columns=features)
    X_all_sm = sm.add_constant(X_all_df)
    
    expected_counts = result.predict(X_all_sm, offset=offset.reset_index(drop=True))

    pred_df = pd.DataFrame({
        "segment_id":     segment_ids.values,
        "expected_acc":   expected_counts.round(3),
        "actual_acc":     y.values,
    }).sort_values("expected_acc", ascending=False)

    os.makedirs("data/processed/results", exist_ok=True)
    out_path = "data/processed/results/count_predictions.parquet"
    pred_df.to_parquet(out_path, index=False)
    print(f"  Count predictions saved → {out_path}")

    print(f"\n  Top 10 Predicted High-Accident Segments:")
    print(pred_df.head(10)[["segment_id", "expected_acc", "actual_acc"]].to_string(index=False))

    # 9. Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/negative_binomial.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n  Model saved → {model_path}")


if __name__ == "__main__":
    main()
