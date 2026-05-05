"""
V3 Pipeline Implementation based on Phases I-V
src/modeling/train_v3.py
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
import optuna
from sklearn.metrics import average_precision_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle

import warnings
warnings.filterwarnings("ignore")

CONFIG_PATH = "configs/model_params.yaml"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------
# PHASE I: Unsupervised Risk Grading (FCM)
# ---------------------------------------------------------
def phase_1_fcm_clustering(df, surrogate_cols, n_clusters=4):
    print(f"\n--- Phase I: Unsupervised Risk Grading (FCM) ---")
    print(f"Using surrogate metrics: {surrogate_cols}")
    
    X_surr = df[surrogate_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_surr)
    
    # Invert speed_mean if present so higher scaled value = higher risk
    if "speed_mean" in surrogate_cols:
        idx = surrogate_cols.index("speed_mean")
        X_scaled[:, idx] = -X_scaled[:, idx]
        
    # skfuzzy cmeans expects data in shape (features, samples)
    alldata = X_scaled.T
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, c=n_clusters, m=2.0, error=0.005, maxiter=1000, init=None
    )
    
    # Assign cluster labels based on max membership
    cluster_labels = np.argmax(u, axis=0)
    
    # Sort clusters by "riskiness" (sum of standardized centroid values)
    # Since we inverted speed, higher sum = higher risk
    centroid_risk_scores = np.sum(cntr, axis=1)
    sorted_idx = np.argsort(centroid_risk_scores)
    
    # Create mapping from original cluster ID to ordered risk level (0=Safe, 1=Low, 2=Moderate, 3=High)
    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_idx)}
    ordered_labels = np.vectorize(mapping.get)(cluster_labels)
    
    df["risk_label_fcm"] = ordered_labels
    
    # Validate with XGBoost (Label Identification)
    print("\nValidating FCM clusters with Label Identification Model...")
    xgb_val = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    # Use 5-fold cross val
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X_scaled, ordered_labels):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = ordered_labels[train_idx], ordered_labels[test_idx]
        xgb_val.fit(X_train, y_train)
        preds = xgb_val.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
        
    print(f"Label Identification Accuracy (3-Fold CV): {np.mean(scores):.4f}")
    
    # Print cluster distribution
    print("\nFCM Cluster Distribution:")
    dist = pd.Series(ordered_labels).value_counts().sort_index()
    dist.index = ["Safe (0)", "Low (1)", "Moderate (2)", "High (3)"][:n_clusters]
    print(dist)
    
    return df, mapping

# ---------------------------------------------------------
# PHASE II: Hybrid Feature Selection (RFE)
# ---------------------------------------------------------
def phase_2_feature_selection(df, features, target_col):
    print(f"\n--- Phase II: Hybrid Feature Selection ---")
    X = df[features].fillna(0)
    y = df[target_col]
    
    print(f"Starting features: {len(features)}")
    
    # Base estimator
    estimator = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    
    # RFECV for automatic optimal subset selection
    min_features = max(1, len(features) // 3)
    selector = RFECV(estimator, step=1, cv=StratifiedKFold(3), scoring='accuracy', min_features_to_select=min_features)
    selector = selector.fit(X, y)
    
    selected_features = [f for f, s in zip(features, selector.support_) if s]
    print(f"Optimal number of features: {selector.n_features_}")
    print(f"Selected features: {selected_features}")
    
    return selected_features

# ---------------------------------------------------------
# PHASE III: Handling Class Imbalance (RENN)
# ---------------------------------------------------------
def phase_3_resampling(X, y):
    print(f"\n--- Phase III: Handling Class Imbalance (RENN) ---")
    print("Original target distribution:")
    print(pd.Series(y).value_counts().sort_index())
    
    # RENN will under-sample the majority class (Safe - 0) to clean boundaries
    renn = RepeatedEditedNearestNeighbours()
    try:
        X_resampled, y_resampled = renn.fit_resample(X, y)
        print("\nResampled target distribution:")
        print(pd.Series(y_resampled).value_counts().sort_index())
        return X_resampled, y_resampled
    except Exception as e:
        print(f"RENN failed (likely due to extreme sparsity in minority classes). Error: {e}")
        print("Proceeding without RENN resampling.")
        return X, y

# ---------------------------------------------------------
# PHASE IV: AutoML Optimization (Optuna)
# ---------------------------------------------------------
def phase_4_automl_optuna(X_train, y_train, X_val, y_val, n_trials=20):
    print(f"\n--- Phase IV: Supervised Prediction & AutoML ---")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        model = XGBClassifier(**params)
        
        # Implement early stopping
        # Using 10% of n_estimators as early stopping rounds roughly
        early_stopping_rounds = max(5, int(params['n_estimators'] * 0.1))
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
        
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest Hyperparameters (TPE):")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    best_model = XGBClassifier(**study.best_params, eval_metric='mlogloss', use_label_encoder=False, random_state=42)
    best_model.fit(X_train, y_train)
    return best_model

# ---------------------------------------------------------
# PHASE V: Evaluation Framework (AUPRC & Mapping)
# ---------------------------------------------------------
def phase_5_evaluation(model, X_test, y_test, df_test, segment_ids_test, output_version="advanced"):
    print(f"\n--- Phase V: Evaluation Framework ---")
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate Macro AUPRC (since it's multiclass)
    # We binarize the labels for PR curve calculation
    n_classes = len(np.unique(y_test))
    auprc_scores = []
    
    # Handle case where some classes might not be in test set
    classes_present = np.unique(y_test)
    
    for i in range(n_classes):
        if i in classes_present:
            y_test_bin = (y_test == i).astype(int)
            ap = average_precision_score(y_test_bin, y_prob[:, i])
            auprc_scores.append(ap)
            print(f"Class {i} AUPRC: {ap:.4f}")
            
    print(f"\nMacro AUPRC (Gold Standard for Imbalanced Risk): {np.mean(auprc_scores):.4f}")
    
    # Risk Mapping Projection Data
    # For mapping, we can output the expected risk score: sum(prob * class_id) / max_class_id
    # This gives a continuous 0-1 risk score based on the multiclass probabilities.
    max_class = n_classes - 1
    expected_risk = np.sum(y_prob * np.arange(n_classes), axis=1) / max_class
    
    results = pd.DataFrame({
        "segment_id": segment_ids_test.values,
        "true_risk_level": y_test.values,
        "pred_risk_level": y_pred,
        "risk_score": expected_risk,
        "risk_pct": (expected_risk * 100).round(1)
    })
    
    os.makedirs("data/processed/results", exist_ok=True)
    out_path = f"data/processed/results/risk_scores_{output_version}.parquet"
    results.to_parquet(out_path, index=False)
    print(f"\nRisk Mapping Data saved to -> {out_path}")
    print("Project this onto PyDeck maps using a heat-map gradient.")

def main():
    print("=== H-SPOT V3 PIPELINE (Phases I-V) ===")
    
    # Load Data
    data_path = "data/processed/features/model_dataset.parquet"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} road segments.")
    
    # Available features (combining geometric, probe, spatial)
    # Exclude target/id columns
    exclude_cols = ['segment_id', 'is_risky', 'risk_level']
    features = [c for c in df.columns if c not in exclude_cols and not c.startswith('acc_')]
    
    # 1. Phase I: Unsupervised Labels
    # Surrogate measures focusing on macro traffic/accidents context
    # Note: Using log_acc_rate_per_100m as a surrogate for "accidents" since TIT/CPI are missing
    surrogate_cols = ['speed_mean', 'congestion_score', 'speed_drop_morning', 'acc_rate_per_100m']
    surrogate_cols = [c for c in surrogate_cols if c in df.columns]
    
    df, fcm_mapping = phase_1_fcm_clustering(df, surrogate_cols, n_clusters=4)
    TARGET = "risk_label_fcm"
    
    # 2. Phase II: Hybrid Feature Selection
    # Select from non-surrogate features to prevent leakage? Actually, the user says use V2 features.
    selected_features = phase_2_feature_selection(df, features, TARGET)
    
    # Train/Val/Test Split
    X = df[selected_features].fillna(0)
    y = df[TARGET]
    
    # 60/20/20 split
    X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
        X, y, df['segment_id'], test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    ) # 0.25 of 0.8 = 0.2
    
    # 3. Phase III: Handling Class Imbalance
    X_train_res, y_train_res = phase_3_resampling(X_train, y_train)
    
    # 4. Phase IV: Supervised Prediction & AutoML
    best_model = phase_4_automl_optuna(X_train_res, y_train_res, X_val, y_val, n_trials=15)
    
    # Save Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_v3_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nModel saved to -> {model_path}")
    
    # 5. Phase V: Evaluation
    phase_5_evaluation(best_model, X_test, y_test, df.loc[X_test.index], ids_test, output_version="v3")
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
