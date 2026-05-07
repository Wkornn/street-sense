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
*   **Map Matching:** Snapping accident points and probe pings to segments using an R-Tree spatial index with a 30m search radius.

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

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and add your `GEMINI_API_KEY` for narrative risk explanations.
Customize pipeline parameters in `configs/data_sources.yaml` and `configs/model_params.yaml`.

### 3. Visualization
Launch the interactive dashboard to explore the risk map and XAI narratives:
```bash
streamlit run app.py
```

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
*Developed as a GIS Term Project for CPE494.*
