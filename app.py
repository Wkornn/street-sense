import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import pickle
import yaml
import os
import sys

# Windows compatibility fix for shap/pyspark
if sys.platform == "win32":
    from unittest.mock import MagicMock
    sys.modules["pyspark"] = MagicMock()

# Set up local imports
from pathlib import Path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.modeling.xai_explainer import explain_segment
from src.modeling.narrator import generate_explanation

st.set_page_config(page_title="H-Spot Bangkok", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Configuration & Caching
# -----------------------------------------------------------------------------
DATA_CFG_PATH  = "configs/data_sources.yaml"
MODEL_CFG_PATH = "configs/model_params.yaml"

RISK_SCORES_PATH = Path("data/processed/results/risk_scores_v2_xgboost.parquet")
SEGMENTS_PATH = Path("data/processed/road_segments.gpkg")
MODEL_DATASET_PATH = Path("data/processed/features/model_dataset.parquet")
MODEL_PATH = Path("models/xgboost_v2_xgboost.pkl")
SNAPPED_ACCIDENTS_PATH = Path("data/processed/accidents_snapped.gpkg")


def missing_paths(*paths):
    return [str(path) for path in paths if not path.exists()]


def prototype_risk_data():
    """Small Bangkok sample so the dashboard can run before the data pipeline."""
    records = [
        {
            "segment_id": 103470,
            "risk_score": 0.68,
            "risk_pct": 68.0,
            "path": [[100.4982, 13.7528], [100.5058, 13.7562], [100.5129, 13.7589]],
            "color": [255, 0, 0, 255],
        },
        {
            "segment_id": 88421,
            "risk_score": 0.42,
            "risk_pct": 42.0,
            "path": [[100.5238, 13.7444], [100.5298, 13.7481], [100.5368, 13.7517]],
            "color": [255, 165, 0, 220],
        },
        {
            "segment_id": 45112,
            "risk_score": 0.31,
            "risk_pct": 31.0,
            "path": [[100.4867, 13.7655], [100.4937, 13.7678], [100.5011, 13.7702]],
            "color": [255, 165, 0, 210],
        },
        {
            "segment_id": 77005,
            "risk_score": 0.23,
            "risk_pct": 23.0,
            "path": [[100.5451, 13.7231], [100.5508, 13.7295], [100.5562, 13.7348]],
            "color": [255, 255, 0, 170],
        },
        {
            "segment_id": 25018,
            "risk_score": 0.18,
            "risk_pct": 18.0,
            "path": [[100.4699, 13.7356], [100.4787, 13.7395], [100.4874, 13.7427]],
            "color": [255, 255, 0, 150],
        },
    ]
    return pd.DataFrame(records)


def prototype_xai_result(segment_id):
    risk_score = 0.68 if segment_id == 103470 else 0.42
    top_factors = {
        "congestion_score": 0.184,
        "log_poi_count_200m": 0.121,
        "speed_drop_morning": 0.087,
        "log_dist_intersection_m": -0.052,
        "highway_rank": 0.038,
    }
    return risk_score, top_factors

@st.cache_resource
def load_config():
    with open(DATA_CFG_PATH, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with open(MODEL_CFG_PATH, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg

@st.cache_data
def load_risk_data(threshold=0.15):
    """Loads and merges geometries with risk scores. Filters by threshold to keep map fast."""
    if missing_paths(RISK_SCORES_PATH, SEGMENTS_PATH):
        demo = prototype_risk_data()
        return demo[demo["risk_score"] >= threshold].copy()
    
    scores = pd.read_parquet(RISK_SCORES_PATH)
    segments = gpd.read_file(SEGMENTS_PATH, columns=['segment_id', 'geometry'])
    
    # Merge and transform to WGS84 for PyDeck
    gdf = segments.merge(scores, on="segment_id", how="inner")
    gdf = gdf.to_crs("EPSG:4326")
    
    # Aggressive filtering to prevent MessageSizeError
    gdf_filtered = gdf[gdf['risk_score'] >= threshold].copy()
    
    # Convert Linestring to coordinate lists for PyDeck PathLayer
    gdf_filtered['path'] = gdf_filtered['geometry'].apply(lambda geom: [[c[0], c[1]] for c in geom.coords])
    
    # Define colors based on risk
    def get_color(risk):
        if risk > 0.5: return [255, 0, 0, 255]      # Red
        if risk > 0.3: return [255, 165, 0, 200]    # Orange
        return [255, 255, 0, 150]                   # Yellow
        
    gdf_filtered['color'] = gdf_filtered['risk_score'].apply(get_color)
    # Drop geometry object to save memory/payload size
    return gdf_filtered.drop(columns=['geometry'])

@st.cache_data
def load_segments():
    """Loads all road segments for mapping."""
    if not SEGMENTS_PATH.exists():
        return None
    return gpd.read_file(SEGMENTS_PATH, columns=['segment_id', 'geometry'])

@st.cache_data
def load_xai_data():
    if missing_paths(MODEL_DATASET_PATH, MODEL_PATH):
        return None, None, [], None

    data_cfg, model_cfg = load_config()
    features_dir = data_cfg["features"]["output_dir"]
    df = pd.read_parquet(os.path.join(features_dir, "model_dataset.parquet"))
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    features = [f for f in model_cfg["modeling"]["features_v2"] if f in df.columns]
    return df, model, features, model_cfg

@st.cache_data
def load_historical_accidents():
    """Loads pre-snapped accident data."""
    if not SNAPPED_ACCIDENTS_PATH.exists():
        return None
    
    # Load data
    gdf = gpd.read_file(SNAPPED_ACCIDENTS_PATH)
    
    # Ensure time columns are numeric
    gdf['year'] = pd.to_numeric(gdf['year'], errors='coerce')
    gdf['month'] = pd.to_numeric(gdf['month'], errors='coerce')
    gdf['hour'] = pd.to_numeric(gdf['hour'], errors='coerce')
    
    # Transform to WGS84 for visualization
    gdf = gdf.to_crs("EPSG:4326")
    return gdf

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🚦 H-Spot Bangkok")
    st.sidebar.markdown("Urban Traffic Risk Assessment")

    prototype_missing = missing_paths(RISK_SCORES_PATH, SEGMENTS_PATH, MODEL_DATASET_PATH, MODEL_PATH)
    if prototype_missing:
        st.sidebar.info("Prototype mode: using sample Bangkok data until pipeline artifacts are available.")
    
    mode = st.sidebar.radio("Navigation", [
        "1. Predictive Risk Map",
        "2. Historical Map",
        "3. Explainable AI (XAI)"
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Map Filters")
    risk_threshold = st.sidebar.slider(
        "Min Risk Threshold (%)", 
        min_value=5, max_value=90, value=20, step=5
    ) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System")
    if st.sidebar.button("Clear App Cache"):
        st.cache_data.clear()
        st.rerun()
    
    data_cfg, model_cfg = load_config()

    if mode == "1. Predictive Risk Map":
        st.header("🔮 Predictive Risk Map")
        st.markdown(f"Showing segments with predicted risk ≥ {risk_threshold*100:.0f}%.")

        if missing_paths(RISK_SCORES_PATH, SEGMENTS_PATH):
            st.info("Prototype map data is being shown because processed risk scores or road segments are missing.")
        
        with st.spinner("Filtering and loading Map Data..."):
            gdf = load_risk_data(risk_threshold)
            
            if gdf.empty:
                st.warning(f"No segments found with risk ≥ {risk_threshold*100:.0f}%. Try lowering the threshold.")
                return
            
            st.caption(f"Currently displaying {len(gdf):,} segments.")
            
            # Setup PyDeck Layer
            layer = pdk.Layer(
                "PathLayer",
                gdf,
                pickable=True,
                get_color="color",
                width_scale=20,
                width_min_pixels=2,
                get_path="path",
                get_width=5,
            )
            
            # Top-down 2D view centered on Bangkok
            view_state = pdk.ViewState(latitude=13.7563, longitude=100.5018, zoom=11, pitch=0, bearing=0)
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Segment ID: {segment_id}\nRisk Score: {risk_pct}%"}
            )
            
            st.pydeck_chart(r)

    elif mode == "2. Historical Map":
        st.header("📍 Historical Map Visualization")
        
        accidents = load_historical_accidents()
        if accidents is None:
            st.error(f"Historical accident data not found at {SNAPPED_ACCIDENTS_PATH}. Please run the pipeline first.")
            return

        # ---------------------------------------------------------
        # Filters
        # ---------------------------------------------------------
        with st.expander("🛠️ Advanced Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                available_years = sorted(accidents['year'].dropna().unique().astype(int))
                selected_years = st.multiselect("Select Year(s)", available_years, default=available_years)
            
            with col2:
                months = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 
                          7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
                selected_month_ids = st.multiselect("Select Month(s)", options=list(months.keys()), 
                                                   format_func=lambda x: months[x], default=list(months.keys()))
            
            with col3:
                hour_range = st.slider("Hour of Day (0-23)", 0, 23, (0, 23))

        # Apply Filters
        filtered_acc = accidents[
            (accidents['year'].isin(selected_years)) & 
            (accidents['month'].isin(selected_month_ids)) &
            (accidents['hour'] >= hour_range[0]) &
            (accidents['hour'] <= hour_range[1])
        ].copy()

        if filtered_acc.empty:
            st.warning("No accidents found matching the selected filters.")
            return

        # ---------------------------------------------------------
        # Metrics & Summary
        # ---------------------------------------------------------
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Accidents", f"{len(filtered_acc):,}")
        
        # Most frequent cause
        if 'บริเวณที่เกิดเหตุ' in filtered_acc.columns:
            top_loc = filtered_acc['บริเวณที่เกิดเหตุ'].mode()
            m2.metric("Common Area Type", top_loc[0] if not top_loc.empty else "N/A")
            
        # Peak Hour
        peak_h = filtered_acc['hour'].mode()
        m3.metric("Peak Hour", f"{int(peak_h[0])}:00" if not peak_h.empty else "N/A")

        # ---------------------------------------------------------
        # Map Visualization
        # ---------------------------------------------------------
        st.subheader("Accident Density Map")
        
        # Data preparation for PyDeck
        filtered_acc['lng'] = filtered_acc.geometry.x
        filtered_acc['lat'] = filtered_acc.geometry.y
        
        # Layer Selection
        map_style = st.radio("Visualization Style", ["Heatmap", "Hexagon Grid", "Individual Points", "Road Segments"], horizontal=True)
        
        layers = []
        if map_style == "Heatmap":
            layers.append(pdk.Layer(
                "HeatmapLayer",
                data=filtered_acc,
                get_position=["lng", "lat"],
                radius_pixels=30,
                intensity=1,
                threshold=0.1,
            ))
        elif map_style == "Hexagon Grid":
            layers.append(pdk.Layer(
                "HexagonLayer",
                data=filtered_acc,
                get_position=["lng", "lat"],
                radius=150,
                elevation_scale=10,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ))
        elif map_style == "Road Segments":
            with st.spinner("Aggregating accidents to road segments..."):
                # 1. Aggregate filtered accidents by segment_id
                acc_counts = filtered_acc.groupby('segment_id').size().reset_index(name='acc_count')
                
                # 2. Load road segments (cached)
                segments = load_segments()
                if segments is None:
                    st.error("Road segments data not found.")
                    return
                
                gdf_merged = segments.merge(acc_counts, on='segment_id', how='inner')
                gdf_merged = gdf_merged.to_crs("EPSG:4326")
                
                # Convert to path for PyDeck
                gdf_merged['path'] = gdf_merged['geometry'].apply(lambda geom: [[c[0], c[1]] for c in geom.coords])
                
                # Color scale (Yellow to Red)
                max_c = gdf_merged['acc_count'].max()
                def get_agg_color(count):
                    if count > max_c * 0.7: return [255, 0, 0, 255]
                    if count > max_c * 0.3: return [255, 165, 0, 220]
                    return [255, 255, 0, 180]
                
                gdf_merged['color'] = gdf_merged['acc_count'].apply(get_agg_color)
                
                layers.append(pdk.Layer(
                    "PathLayer",
                    gdf_merged,
                    pickable=True,
                    get_color="color",
                    width_scale=20,
                    width_min_pixels=2,
                    get_path="path",
                    get_width=5,
                ))
        else:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=filtered_acc,
                get_position=["lng", "lat"],
                get_color=[255, 60, 60, 160],
                get_radius=20,
                pickable=True,
            ))

        view_state = pdk.ViewState(latitude=13.7563, longitude=100.5018, zoom=11, pitch=45 if map_style=="Hexagon Grid" else 0)
        
        # Tooltip handling
        if map_style == "Road Segments":
            tooltip = {"text": "Segment ID: {segment_id}\nFiltered Accidents: {acc_count}"}
        elif map_style == "Individual Points":
            tooltip = {"text": "Accident Info\nYear: {year}\nSeverity: {severity_label}"}
        else:
            tooltip = True

        r = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip
        )
        st.pydeck_chart(r)

    elif mode == "3. Explainable AI (XAI)":
        st.header("🧠 Explainable AI (XAI)")
        st.markdown("Enter a Road Segment ID to understand *why* it is considered risky.")
        
        df, model, features, xai_model_cfg = load_xai_data()
        is_prototype_xai = df is None or model is None

        if is_prototype_xai:
            st.info("Prototype XAI is being shown because the trained model or model dataset is missing.")
        
        segment_id = st.number_input("Enter Segment ID", min_value=0, value=103470, step=1)
        
        if st.button("Analyze Risk"):
            with st.spinner("Calculating SHAP values..."):
                try:
                    if is_prototype_xai:
                        risk_score, top_factors = prototype_xai_result(segment_id)
                    else:
                        risk_score, top_factors = explain_segment(segment_id, df, model, features, top_k=5)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Predicted Risk Score", f"{risk_score*100:.1f}%")
                        st.markdown("### Top Factors Driving Risk")
                        
                        # Prepare data for chart
                        plot_df = pd.DataFrame(list(top_factors.items()), columns=['Feature', 'Impact'])
                        plot_df['Direction'] = plot_df['Impact'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
                        plot_df['Absolute Impact'] = plot_df['Impact'].abs()
                        plot_df = plot_df.sort_values('Absolute Impact', ascending=True)
                        
                        st.bar_chart(plot_df.set_index('Feature')['Impact'])
                        
                    with col2:
                        st.markdown("### AI Narrative Explanation")
                        should_narrate = False if is_prototype_xai else xai_model_cfg.get("explanation", {}).get("enable_narrative", False)
                        
                        if is_prototype_xai:
                            st.info(
                                "Prototype narrative: this segment is flagged mainly because sample congestion, nearby activity density, "
                                "and morning speed drop increase the synthetic risk score. A practical countermeasure would be to review "
                                "signal timing and targeted speed enforcement during peak periods."
                            )
                        elif should_narrate:
                            if os.environ.get("GEMINI_API_KEY"):
                                with st.spinner("Asking Gemini..."):
                                    llm_model = xai_model_cfg.get("explanation", {}).get("llm_model", "gemini-1.5-flash")
                                    narrative = generate_explanation(segment_id, risk_score, top_factors, llm_model)
                                    if narrative:
                                        st.info(narrative)
                                    else:
                                        st.error("Failed to generate narrative. Check terminal for errors.")
                            else:
                                st.warning("GEMINI_API_KEY not found in environment. Narrative generation skipped. Check your .env file.")
                        else:
                            st.info("Narrative generation is disabled in configs/model_params.yaml.")
                except ValueError as e:
                    st.error(str(e))

if __name__ == "__main__":
    main()
