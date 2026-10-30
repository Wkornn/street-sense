# 🚦 Street Sense Bangkok
**Spatial Accident Risk and Causality Assessment in Bangkok Using Machine Learning and Urban Context Features**

Street Sense is a predictive analytics platform designed to identify, analyze, and explain traffic accident hotspots in Bangkok. By integrating historical accident data, OpenStreetMap (OSM) road networks, and dynamic traffic probe data, the system builds high-precision risk models calibrated for urban planning and safety interventions.

---

## 🏗️ Pipeline Architecture

The project follows a 5-phase GIS-centric pipeline that transforms raw spatial data into actionable risk probabilities.

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
*   **Config-driven training:** one YAML file per run, with interchangeable model plugins (see [How Training Works](#-how-training-works)).
*   **Default model:** XGBoost gradient boosting, calibrated with **Isotonic Regression** so scores are real probabilities (0.0 to 1.0).
*   **Baseline:** Negative Binomial count regression, the standard road-safety approach.

---

## 🚀 Getting Started

### 1. Create an environment
Python 3.11 is recommended (the version the project is developed with).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and add your `GEMINI_API_KEY` if you want AI-generated narrative explanations in the dashboard.

```bash
cp .env.example .env
```

Every setting lives in `configs/`. Each file has one job:

| File | What it controls |
|---|---|
| `data_sources.yaml` | Paths, Bangkok boundary, OSM layers, CRS, segment length (100 m), snap radius (50 m), POI and feature settings |
| `data_inventory.yaml` | Which years of MOT accident data (and probe months) to download |
| `pipeline.yaml` | Which pipeline phases run by default, and which training runs (`train_configs`) |
| `model_params.yaml` | Dataset preprocessing (sparse time bins, default POI distance, congestion weights) |
| `features.yaml` | Named feature sets used by training runs |
| `train/*.yaml` | One file per training run: target, features, model, split, output folder |
| `app.yaml` | Which trained model the dashboard serves (`model_dir`) and LLM narrative settings |

### 3. Get the data
See [Data](#-data) below for sources and licenses. The pipeline downloads OSM and MOT accident data itself; probe data is optional.

### 4. Build the dataset and train
Run the configured pipeline from the project root:

```bash
python scripts/run_pipeline.py
```

Phases run in order:

1. **Ingestion:** OSM layers, MOT accidents, and (if `run_probe: true`) probe archives
2. **Preprocessing:** clean accidents, split roads into 100 m segments, snap accidents to the nearest segment (≤ 50 m)
3. **Features:** road, accident, spatial and (optionally) probe features
4. **Feature matrix and model dataset**
5. **Training:** every config listed in `pipeline.yaml` → `train_configs`

Steps whose outputs already exist are skipped. The download steps also skip files they already have.

```bash
python scripts/run_pipeline.py --force          # rebuild everything, including retraining
```

Probe processing is slow and downloads large archives. Set `run_probe: false` in `configs/pipeline.yaml` unless traffic features are needed. The command-line flags can switch phases on but not off; to run fewer phases, set them to `false` in `pipeline.yaml`.

> **Rebuilding after changing segmentation or accident cleaning:** every later step must be rebuilt too, because the segment IDs and accident features depend on them. Use `--force`, or delete the affected outputs, so nothing mixes old and new data.

Main artifacts:

*   `data/processed/road_segments.gpkg`, `data/processed/accidents_snapped.parquet`
*   `data/processed/features/model_dataset.parquet`: one row per 100 m segment, the input to training
*   `models/<run>/`: one folder per trained model (see below)

### 5. Run the dashboard

```bash
streamlit run app.py
```

*   **Predictive Risk Map:** segment risk scores, filtering, per-segment "Top Factors" explanation, and optional Gemini narratives.
*   **Historical Map:** accidents by date, time of day and severity; heatmap, hexagon and point views.

If processed artifacts are missing, the app falls back to a small prototype map (`python scripts/create_prototype_data.py` creates local prototype artifacts).

---

## 🧠 How Training Works

### One config file = one training run

```bash
python -m src.modeling.train --config configs/train/xgb_severe.yaml
```

A run config states everything about the experiment; the code is only the machinery:

```yaml
name: xgb_severe
target:
  type: binary                        # binary (any event) or count (number of events)
  columns: [acc_fatal, acc_serious]
features: traffic_spatial_exposure    # a named set in configs/features.yaml
model:
  type: xgb_classifier                # which plugin (see below)
  params: {n_estimators: 300, max_depth: 6, learning_rate: 0.05}
  calibration: isotonic
split: {method: random, cal_size: 0.2, test_size: 0.2, seed: 42}
output:
  model_dir: models/xgb_severe
```

Changing parameters, features, target or split needs **only a new YAML file**. A new kind of model needs a plugin.

### Flow

```
config → load data → target → features → split (60/20/20)
       → plugin.fit → evaluate on test split → plugin.save + meta.json + risk_scores.parquet
```

| Module | Job (shared by every approach) |
|---|---|
| `src/modeling/train.py` | Entrypoint; runs the flow above |
| `src/modeling/config.py` | Loads a run config and resolves the feature-set name |
| `src/modeling/data.py` | Load, target, exposure, feature selection, split. **Also used by the app**, so training and serving prepare features identically |
| `src/modeling/evaluate.py` | Metrics. Every model is scored on "any fatal/serious accident" with the same test split, so approaches are comparable |
| `src/modeling/models/` | The plugin registry and one file per approach |

### Model plugins

Each approach is a class registered under a name and owns everything specific to it (imbalance handling, calibration, file format):

```python
@register("xgb_classifier")
class XGBClassifierPlugin:
    @staticmethod
    def fit(model_cfg, X_train, y_train, X_cal, y_cal): ...   # returns a fitted model
    @staticmethod
    def save(model, out_dir): ...
    @staticmethod
    def load(model_dir): ...
```

Every fitted model provides `predict_risk(X)` (probability of at least one event, 0–1) and `explain(X)` (per-feature contributions). Count models also provide `predict_count(X)`.

**To add an approach:** create `src/modeling/models/<name>.py`, import it at the bottom of `src/modeling/models/__init__.py`, and write a config with `model.type: <name>`.

| Plugin | Approach | Saved as |
|---|---|---|
| `xgb_classifier` | XGBoost classifier, `scale_pos_weight` for imbalance, isotonic calibration on a held-out split | `booster.json` + `calibrator.joblib` |
| `neg_binomial` | Negative Binomial count regression (GLM) with a road-length offset, the Highway Safety Manual approach | `glm.json` (coefficients) |
| `fcm_xgb` | *Experimental.* Fuzzy C-means risk grades on accident roads, then XGBoost on the grades (the earlier "V4" approach) | `booster.json` + `grades.json` |

Models are saved in native formats, not pickle, so they load across library versions.

### What a run produces

```
models/xgb_severe/
  booster.json, calibrator.joblib   ← the model (plugin-specific files)
  meta.json                         ← plugin name, target, feature list, test metrics, timestamp
  config.yaml                       ← the exact config used, for reproducing the run
  risk_scores.parquet               ← risk_score (and expected_count for count models) per segment
```

`load_model(model_dir)` reads `meta.json`, picks the right plugin and returns the model, so the app never needs to know which approach it is serving.

### Comparing approaches

Same target (fatal or serious accident), same features where possible, same test split:

| Run | ROC-AUC | PR-AUC | Top-decile lift | Notes |
|---|---|---|---|---|
| `xgb_severe` | **0.859** | **0.213** | **6.8×** | Default model for the app |
| `nb_severe` | 0.797 | 0.098 | 5.7× | Interpretable baseline; predicts expected accident counts |
| `fcm_xgb_v4` | 0.229 | 0.006 | 0.3× | Experimental; see below |

Each folder's `meta.json` holds the full metrics. Top-decile lift = accident rate in the 10% highest-scored segments divided by the overall rate. The split is random for now; a spatial split (planned) will give more honest, likely lower, scores, because neighbouring segments are very similar.

**Why V4 fails:** its clustering assumes slow, congested roads are risky. Fatal and serious accidents happen mostly on fast, major roads (the XGBoost model's top factors are `highway_rank` and `speed_mean`), so V4 ranks roads almost in reverse. An earlier version looked better only because accident-history columns leaked into its inputs.

### Switching the model the app serves

Change one line in `configs/app.yaml` and restart the app:

```yaml
model_dir: models/nb_severe     # was models/xgb_severe
```

The map, risk scores and "Top Factors" panel all come from that folder.

---

## 📊 Data

| Source | Used for | License / terms | How it is obtained |
|---|---|---|---|
| **OpenStreetMap** (via OSMnx) | Roads, buildings, POIs, Bangkok boundary | ODbL, see <https://www.openstreetmap.org/copyright> | Downloaded by `src/ingestion/load_osm_data.py` (layers in `data_sources.yaml`) |
| **Ministry of Transport accident records** ([datagov.mot.go.th](https://datagov.mot.go.th/)) | Accident locations, severity, time, weather (2019–2025) | Thai government open data; see the portal's terms | Downloaded through the portal's API by `src/ingestion/load_accident_data.py` (resource IDs in `data_inventory.yaml`) |
| **iTIC Foundation / Longdo probe data** ([itic.longdo.com](https://itic.longdo.com/data/)) | GPS speeds and congestion per segment | See the provider's terms | Optional. Monthly archives downloaded by `src/ingestion/load_probe_data.py`, clipped to Bangkok (~21 GB after extraction) |

Data zones:

*   `data/raw/`: downloaded source files
*   `data/processed/`: cleaned spatial files, snapped accidents, road segments, feature tables
*   `models/`: trained model folders

`data/` and `models/` are not tracked in git; they are rebuilt by the pipeline. Without probe data, the dataset still builds, but the traffic columns are missing; training then stops with a "missing feature columns" error, because both feature sets in `configs/features.yaml` use them. To train without probe data, add a feature set without the speed, congestion and probe columns.

---

## 🧪 Development Tips

Run everything from the repository root so relative paths resolve:

```bash
python scripts/run_pipeline.py
python -m src.modeling.train --config configs/train/nb_severe.yaml
python src/modeling/xai_explainer.py --segment_id 103470 --no-narrative
streamlit run app.py
```

When adding a feature column: generate it in `src/features/`, then add it to a feature set in `configs/features.yaml`.

---

## 📂 Project Structure

```text
street-sense/
├── app.py                # Streamlit dashboard
├── scripts/              # Pipeline orchestration (run_pipeline.py)
├── src/
│   ├── ingestion/        # Data loading (OSM, MOT, probes)
│   ├── geospatial/       # Road segmentation and snapping
│   ├── features/         # Feature engineering and matrix building
│   └── modeling/         # Training framework, model plugins, explanations
├── configs/              # YAML configuration (see table above)
│   └── train/            # One file per training run
├── data/                 # Raw and processed data (not in git)
├── models/               # One folder per trained model (not in git)
└── notebooks/            # EDA and result analysis
```

---

## 🛠️ Tech Stack
*   **Geospatial:** GeoPandas, OSMnx, Shapely, PyDeck
*   **Machine Learning:** XGBoost, Scikit-Learn (calibration), Statsmodels (count models)
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
