# 🚦 H-Spot Bangkok
**Spatio-Temporal Accident Risk and Causality Assessment in Bangkok Using Machine Learning and Urban Context Features**

H-Spot is a predictive analytics platform designed to identify, analyze, and explain traffic accident hotspots in Bangkok. It combines historical accident data, OpenStreetMap (OSM) road networks, and probe-based traffic data to build high-precision risk models.

## 🏗️ Project Structure

```text
H-Spot/
├── app.py                # Streamlit Dashboard (Main Entrypoint)
├── scripts/              # Pipeline and utility scripts
│   └── run_pipeline.py   # Main data processing orchestrator
├── src/                  # Core source code
│   ├── ingestion/        # Data loading from OSM and MOT
│   ├── geospatial/       # Road segmentation and hotspot analysis
│   ├── features/         # Feature engineering and matrix building
│   └── modeling/         # Model training, evaluation, and XAI
├── configs/              # YAML configurations (Data sources, Model params)
├── data/                 # Data storage (Git ignored except for structure)
│   ├── raw/              # Original datasets
│   └── processed/        # Cleaned and engineered features
├── models/               # Serialized model artifacts (.pkl)
├── notebooks/            # Jupyter notebooks for EDA and prototyping
├── .streamlit/           # Streamlit configuration
├── .env                  # Environment variables (API Keys)
└── requirements.txt      # Project dependencies
```

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and add your `GEMINI_API_KEY` for narrative explanations.

### 3. Running the Pipeline
To process data and generate features:
```bash
python scripts/run_pipeline.py --all
```

### 4. Training Models
```bash
python src/modeling/v2_binary_classification.py
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

## 🧠 Core Components
- **Predictive Risk Scoring**: XGBoost-based models calibrated for probability-based risk assessment.
- **Explainable AI (XAI)**: SHAP-based feature importance integrated with Gemini LLM for human-readable risk narratives.

## 🛠️ Tech Stack
- **Backend**: Python, GeoPandas, XGBoost, Scikit-Learn
- **UI**: Streamlit, PyDeck
- **Tracking**: MLflow
- **LLM**: Google Gemini API
