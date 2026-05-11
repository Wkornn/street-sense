# 🚦 Street Sense Bangkok
**Spatial Accident Risk and Causality Assessment in Bangkok Using Machine Learning and Urban Context Features**

Street Sense is a predictive analytics platform designed to identify, analyze, and explain traffic accident hotspots in Bangkok. By integrating historical accident data, OpenStreetMap (OSM) road networks, and dynamic traffic probe data, the system builds high-precision risk models calibrated for urban planning and safety interventions.

---

## 🏗️ Pipeline Architecture

The project follows a rigorous 5-phase GIS-centric pipeline that transforms raw spatial data into actionable risk probabilities.

### 1. Data Ingestion
*   **OSM Network:** Automated fetching of drivable road networks, building footprints, and POIs (Schools, Hospitals, Malls) via `osmnx`.
*   **MOT Accidents:** Historical accident logs with severity, weather, and precise timestamps.
*   **Traffic Probes:** High-frequency GPS pings providing real-time operational speeds and congestion patterns.

### 2. Preprocessing & CRS Standardization
*   **Spatial Filtering:** Data is clipped to the Bangkok Metropolitan boundary using Point-in-Polygon joins.
*   **UTM Projection:** All geometries are projected to **EPSG:32647 (UTM Zone 47N)** to ensure accurate metric distance and area calculations.

### 3. Spatial Operations (The GIS Engine)
*   **Road Segmentation:** Continuous road lines are split into fixed-length **100m segments**, creating high-resolution analytical units.
*   **Intersection Detection:** Topological analysis to identify conflict points where ≥3 road segments meet.
*   **Map Matching:** Snapping accident points and probe pings to segments using an R-Tree spatial index with a configurable search radius.

### 4. Feature Engineering
*   **Infrastructure:** Road rank, lane counts, and segment topology.
*   **Urban Context:** Building density (200m buffer) and proximity to key points of interest.
*   **Historical Risk:** Time-binned accident frequencies (Morning Peak, Monsoon, etc.).
*   **Dynamic Traffic:** Weighted congestion scores and speed variance from probe data.

### 5. Modeling & Risk Prediction
*   **Algorithm:** XGBoost gradient boosting for non-linear risk assessment.
*   **Calibration:** Raw model outputs are passed through **Isotonic Regression** to produce statistically valid risk probabilities (0.0 to 1.0).

---

## 🚀 Getting Started

### 1. Create an environment
Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate with:

```bash
.venv\Scripts\activate
```

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Configuration
Copy `.env.example` to `.env` and add your `GEMINI_API_KEY` if you want AI-generated narrative explanations in the dashboard.

```bash
cp .env.example .env
```

The main configuration files are:

*   `configs/data_sources.yaml` - source paths, Bangkok boundary, OSM layers, CRS, road segmentation, snapping, POI, and feature settings.
*   `configs/data_inventory.yaml` - annual MOT accident resource IDs used by the ingestion script.
*   `configs/pipeline.yaml` - default pipeline phases and whether to process probe data.
*   `configs/model_params.yaml` - preprocessing choices, model hyperparameters, feature sets, and narrative model settings.

### 4. Build pipeline artifacts
Run the full configured pipeline from the project root:

```bash
python scripts/run_pipeline.py
```

Useful variants:

```bash
python scripts/run_pipeline.py --all
python scripts/run_pipeline.py --all --force
python scripts/run_pipeline.py --features --matrix
python scripts/run_pipeline.py --all --probe
```

Probe processing can take a long time and download large archives. Disable it by setting `run_probe: false` in `configs/pipeline.yaml` unless traffic probe features are needed.

The pipeline produces the main artifacts used by modeling and the dashboard:

*   `data/processed/road_segments.gpkg`
*   `data/processed/accidents_snapped.parquet`
*   `data/processed/features/feature_matrix.parquet`
*   `data/processed/features/model_dataset.parquet`

### 5. Train risk models
Train the default calibrated binary classifier used by the Streamlit app:

```bash
python src/modeling/train.py
```

This writes the dashboard defaults:

*   `models/xgboost_bi_classification.pkl`
*   `data/processed/results/risk_scores.parquet`

For experiment variants, use:

```bash
python src/modeling/train_classification.py --version v2 --calibration isotonic
```

Use `--version v1` for the full historical-accident feature set, or `--version v2` for the no-history feature set intended for new or under-observed roads.

Variant risk score outputs are written to `data/processed/results/risk_scores_<version>.parquet` and variant model artifacts are written to `models/`. To use a variant in the dashboard, either update `MODEL_PATH` and `RISK_SCORES_PATH` in `app.py` or copy the chosen outputs to the default filenames above.

### 6. Visualization
Launch the interactive dashboard to explore the risk map and XAI narratives:

```bash
streamlit run app.py
```

The dashboard includes:

*   **Predictive Risk Map:** road segment risk scores, filtering, segment selection, SHAP-style factor display, and optional Gemini narrative explanations.
*   **Historical Map:** accident filtering by date, time of day, severity, heatmap/hexagon/point views, and road-segment aggregation.

If processed artifacts are missing, the app falls back to a small prototype map so the interface can still be explored. To create local prototype artifacts explicitly, run:

```bash
python scripts/create_prototype_data.py
```

---

## 📊 Data And Outputs

The repository uses the following data zones:

*   `data/raw/` - downloaded OSM, MOT accident, and iTIC/Longdo probe source files.
*   `data/processed/` - cleaned spatial files, snapped accident points, road segments, and feature tables.
*   `data/processed/features/` - feature-level parquet outputs from the feature engineering phase.
*   `data/processed/results/` - predicted risk scores for dashboard mapping.
*   `models/` - serialized model artifacts used by the app and explanation tools.

Large raw data downloads are expected to be regenerated from the configured providers rather than edited manually.

---

## 🧭 Modeling Notes

The main supervised target is `is_risky`, derived from whether a road segment has historical accident evidence after preprocessing. The classification workflow handles class imbalance with `scale_pos_weight`, calibrates probabilities with scikit-learn's `CalibratedClassifierCV`, and saves segment-level risk probabilities for mapping.

Two feature sets are maintained:

*   `features` (`v1`) includes historical accident aggregates and is useful for retrospective hotspot analysis.
*   `features_v2` (`v2`) excludes historical accident aggregates and is better suited to new roads or planning scenarios where prior crash counts are unavailable.

The app's narrative explanation path requires `GEMINI_API_KEY`; without it, the dashboard still works but skips AI-generated text.

---

## 🧪 Development Tips

Run scripts from the repository root so relative paths resolve correctly:

```bash
python scripts/run_pipeline.py
python src/modeling/train_classification.py --version v2
streamlit run app.py
```

When adding new feature columns, update both the generating script in `src/features/` and the selected feature list in `configs/model_params.yaml`.

---

## 📂 Project Structure

```text
street-sense/
├── app.py                # Streamlit Dashboard (Main Entrypoint)
├── scripts/              # Pipeline orchestration scripts
├── src/
│   ├── ingestion/        # Data loading (OSM, MOT, Probes)
│   ├── geospatial/       # Road segmentation and snapping logic
│   ├── features/         # Feature engineering and matrix building
│   └── modeling/         # XGBoost training, calibration, and XAI
├── configs/              # YAML configurations for all pipeline stages
├── data/                 # Data storage (Raw & Processed)
├── models/               # Serialized model artifacts (.pkl)
└── notebooks/            # EDA and prototyping environments
```

---

## 🛠️ Tech Stack
*   **Geospatial:** GeoPandas, OSMnx, Shapely, PyDeck
*   **Machine Learning:** XGBoost, Scikit-Learn (Calibration), MLflow
*   **Visualization:** Streamlit
*   **LLM Integration:** Google Gemini (for risk narratives)

---

## 🙏 Data Acknowledgements

Street Sense Bangkok is built on open data made available by public and community data providers:

*   **OpenStreetMap contributors** for road networks, building footprints, land use, and points of interest, accessed through OSMnx. OpenStreetMap data is available under the Open Data Commons Open Database License (ODbL). See <https://www.openstreetmap.org/copyright>.
*   **Ministry of Transport Data Catalog (datagov.mot.go.th)** for historical Bangkok accident records used in the risk modeling pipeline. See <https://datagov.mot.go.th/>.
*   **iTIC Foundation / Longdo Open Data Archives** for historical traffic probe data used to derive traffic speed, congestion, and variance features. See <https://itic.longdo.com/data/>.

These datasets remain the property and responsibility of their respective providers. This project transforms and analyzes them for academic road-safety research.

*Developed as a GIS Term Project for CPE494.*
