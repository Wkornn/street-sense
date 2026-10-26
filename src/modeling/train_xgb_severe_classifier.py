import os
import yaml
import numpy as np
import pandas as pd
import pickle

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import warnings
warnings.filterwarnings("ignore")

CONFIG_PATH = "configs/model_params.yaml"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # =========================
    # 1. LOAD DATA
    # =========================
    data_path = "data/processed/features/model_dataset.parquet"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} road segments.")

    # =========================
    # 2. CREATE TARGET
    # =========================
    df["target"] = (
        (df["acc_fatal"] > 0) |
        (df["acc_serious"] > 0)
    ).astype(int)

    # =========================
    # 3. CREATE EXPOSURE
    # =========================
    df["exposure"] = df["probe_count"].fillna(0) * df["length_m"]
    df["log_exposure"] = np.log1p(df["exposure"])

    # =========================
    # 4. FEATURE SELECTION
    # =========================
    FEATURES = [
        # geometry
        'highway_rank',
        'lanes',
        'lanes_known',
        'length_m',

        # spatial
        'dist_intersection_m',
        'poi_count_200m',
        'building_density_200m',
        'dist_school_m',
        'dist_hospital_m',
        'dist_fuel_m',
        'dist_mall_m',

        'log_dist_intersection_m',
        'log_poi_count_200m',
        'log_building_density_200m',
        'log_dist_school_m',
        'log_dist_hospital_m',
        'log_dist_fuel_m',
        'log_dist_mall_m',

        # traffic
        'speed_mean',
        'speed_mean_daytime',
        'speed_mean_morning_peak',
        'speed_mean_evening_peak',

        'pct_below_20kmh_daytime',
        'pct_below_20kmh_morning_peak',
        'pct_below_20kmh_evening_peak',

        'congestion_score',
        'speed_drop_morning',

        # exposure
        'probe_count',
        'probe_count_daytime',
        'probe_count_morning_peak',
        'probe_count_evening_peak',
        'probe_count_late_night',
        'probe_count_night',
        'has_probe_data',
        'log_probe_count',
        'exposure',
        'log_exposure',
    ]

    FEATURES = [f for f in FEATURES if f in df.columns]

    X = df[FEATURES].fillna(0)
    y = df["target"]
    segment_ids = df["segment_id"]

    print(f"Using {len(FEATURES)} features")

    # =========================
    # 5. SPLIT 60/20/20
    # =========================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

    # =========================
    # 6. HANDLE IMBALANCE
    # =========================
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {ratio:.2f}")

    # =========================
    # 7. TRAIN MODEL
    # =========================
    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    base_model.fit(X_train, y_train)

    # =========================
    # 8. CALIBRATION
    # =========================
    cal_model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv="prefit"
    )

    cal_model.fit(X_cal, y_cal)

    # =========================
    # 9. EVALUATION
    # =========================
    y_prob = cal_model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation ===")
    print("AUC-ROC :", roc_auc_score(y_test, y_prob))
    print("PR-AUC  :", average_precision_score(y_test, y_prob))
    print("Brier   :", brier_score_loss(y_test, y_prob))

    # =========================
    # 10. SAVE MODEL
    # =========================
    os.makedirs("models", exist_ok=True)
    with open("models/xgboost_bi_classification.pkl", "wb") as f:
        pickle.dump(cal_model, f)

    print("\nModel saved.")

    # =========================
    # 11. INFERENCE (ALL DATA)
    # =========================
    df["risk_score"] = cal_model.predict_proba(X)[:, 1]

    os.makedirs("data/processed/results", exist_ok=True)
    df[["segment_id", "risk_score"]].to_parquet(
        "data/processed/results/risk_scores.parquet",
        index=False
    )

    print("\nRisk scores saved.")

    # =========================
    # 12. INSIGHT: NO-ACCIDENT BUT HIGH RISK
    # =========================
    df_new = df[df["acc_total"] == 0]

    print("\nTop risky segments with NO accident history:")
    print(
        df_new.sort_values("risk_score", ascending=False)
        .head(10)[["segment_id", "risk_score"]]
    )


if __name__ == "__main__":
    main()