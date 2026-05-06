import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import pickle
import numpy as np
import yaml
import os
import sys
from dotenv import load_dotenv

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

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

RISK_SCORES_PATH = Path("data/processed/results/risk_scores.parquet")
SEGMENTS_PATH = Path("data/processed/road_segments.gpkg")
MODEL_DATASET_PATH = Path("data/processed/features/model_dataset.parquet")
MODEL_PATH = Path("models/xgboost_bi_classification.pkl")
SNAPPED_ACCIDENTS_PATH = Path("data/processed/accidents_snapped.parquet")


def missing_paths(*paths):
    return [str(path) for path in paths if not path.exists()]


def prototype_risk_data():
    """Small Bangkok sample so the dashboard can run before the data pipeline."""
    records = [
        {
            "segment_id": 103470,
            "risk_score": 0.68,
            "risk_pct": 68.0,
            "historical_accidents": 5,
            "path": [[100.4982, 13.7528], [100.5058, 13.7562], [100.5129, 13.7589]],
        },
        {
            "segment_id": 88421,
            "risk_score": 0.42,
            "risk_pct": 42.0,
            "historical_accidents": 0,
            "path": [[100.5238, 13.7444], [100.5298, 13.7481], [100.5368, 13.7517]],
        },
        {
            "segment_id": 45112,
            "risk_score": 0.31,
            "risk_pct": 31.0,
            "historical_accidents": 2,
            "path": [[100.4867, 13.7655], [100.4937, 13.7678], [100.5011, 13.7702]],
        },
        {
            "segment_id": 77005,
            "risk_score": 0.23,
            "risk_pct": 23.0,
            "historical_accidents": 0,
            "path": [[100.5451, 13.7231], [100.5508, 13.7295], [100.5562, 13.7348]],
        },
        {
            "segment_id": 25018,
            "risk_score": 0.18,
            "risk_pct": 18.0,
            "historical_accidents": 0,
            "path": [[100.4699, 13.7356], [100.4787, 13.7395], [100.4874, 13.7427]],
        },
    ]
    df = pd.DataFrame(records)
    
    def get_color(row):
        if row['historical_accidents'] > 0: return [255, 75, 75, 255] # Red
        return [0, 150, 255, 200] # Blue
        
    df['color'] = df.apply(get_color, axis=1)
    return df


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

@st.cache_data
def load_config():
    with open(DATA_CFG_PATH, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    with open(MODEL_CFG_PATH, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg

@st.cache_data
def load_risk_data():
    """Loads and merges geometries with risk scores. Returns all segments."""
    if missing_paths(RISK_SCORES_PATH, SEGMENTS_PATH):
        return prototype_risk_data()
    
    scores = pd.read_parquet(RISK_SCORES_PATH)
    if 'risk_pct' not in scores.columns:
        scores['risk_pct'] = (scores['risk_score'] * 100).round(1)
        
    segments = gpd.read_file(SEGMENTS_PATH, columns=['segment_id', 'geometry'])
    
    # Merge and transform to WGS84 for PyDeck
    gdf = segments.merge(scores, on="segment_id", how="inner")
    
    # Join with accident history (exclude unsnapped accidents: segment_id == -1)
    accidents = load_historical_accidents()
    if accidents is not None:
        snapped_only = accidents[accidents['segment_id'] != -1]
        acc_counts = snapped_only.groupby('segment_id').size().reset_index(name='historical_accidents')
        gdf = gdf.merge(acc_counts, on='segment_id', how='left')
        gdf['historical_accidents'] = gdf['historical_accidents'].fillna(0)
    else:
        gdf['historical_accidents'] = 0

    gdf = gdf.to_crs("EPSG:4326")
    
    # Convert Linestring to coordinate lists for PyDeck PathLayer
    gdf['path'] = gdf['geometry'].apply(lambda geom: [[c[0], c[1]] for c in geom.coords])
    
    # Define colors based on historical accidents (2 categories)
    def get_color(row):
        if row['historical_accidents'] > 0:
            return [255, 75, 75, 220]      # Red (Historical)
        return [0, 150, 255, 180]          # Blue (No History)
        
    gdf['color'] = gdf.apply(get_color, axis=1)
    
    # Drop geometry object to save memory/payload size
    return gdf.drop(columns=['geometry'])

@st.cache_data
def load_segments():
    """Loads all road segments for mapping. Returns a copy to protect the cache."""
    if not SEGMENTS_PATH.exists():
        return None
    return gpd.read_file(SEGMENTS_PATH, columns=['segment_id', 'geometry']).copy()

@st.cache_data
def load_xai_data():
    if missing_paths(MODEL_DATASET_PATH, MODEL_PATH):
        return None, None, [], None

    data_cfg, model_cfg = load_config()
    features_dir = data_cfg["features"]["output_dir"]
    df = pd.read_parquet(os.path.join(features_dir, "model_dataset.parquet"))
    
    # Re-create engineering features to match training pipeline
    if "exposure" not in df.columns:
        df["exposure"] = df["probe_count"].fillna(0) * df["length_m"]
        df["log_exposure"] = np.log1p(df["exposure"])
    
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # Get features from the model if available, otherwise from config
    try:
        # For CalibratedClassifierCV, reach into the base estimator
        features = model.calibrated_classifiers_[0].estimator.feature_names_in_.tolist()
    except:
        # Fallback to features_v2 in config
        features = [f for f in model_cfg["modeling"]["features_v2"] if f in df.columns]
        
    return df, model, features, model_cfg

@st.cache_data
def load_historical_accidents():
    """Loads pre-snapped accident data."""
    if not SNAPPED_ACCIDENTS_PATH.exists():
        return None
    
    # Load data
    gdf = gpd.read_parquet(SNAPPED_ACCIDENTS_PATH)
    
    # Ensure time columns are numeric
    gdf['year'] = pd.to_numeric(gdf['year'], errors='coerce')
    gdf['month'] = pd.to_numeric(gdf['month'], errors='coerce')
    gdf['hour'] = pd.to_numeric(gdf['hour'], errors='coerce')
    
    if 'วันที่และเวลาที่เกิดเหตุ' in gdf.columns:
        gdf['วันที่และเวลาที่เกิดเหตุ'] = pd.to_datetime(gdf['วันที่และเวลาที่เกิดเหตุ'])
    
    # Transform to WGS84 for visualization
    gdf = gdf.to_crs("EPSG:4326")
    return gdf

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🚦 Street-Sense")
    st.sidebar.markdown("Bangkok Road Risk Assessment")

    # Load configuration at the start
    data_cfg, model_cfg = load_config()

    # Handle map selection from session state (prevents multiple refreshes)
    if "risk_map" in st.session_state:
        event = st.session_state.risk_map
        if event and event.get("selection") and event["selection"].get("objects"):
            selection_objs = event["selection"]["objects"]
            if "risk-paths" in selection_objs and selection_objs["risk-paths"]:
                st.session_state.selected_segment_id = selection_objs["risk-paths"][0]["segment_id"]

    prototype_missing = missing_paths(RISK_SCORES_PATH, SEGMENTS_PATH, MODEL_DATASET_PATH, MODEL_PATH)
    if prototype_missing:
        st.sidebar.info("Prototype mode: using sample Bangkok data until pipeline artifacts are available.")
    
    # Updated Navigation: Removed XAI standalone page
    mode = st.sidebar.radio("Navigation", [
        "1. Predictive Risk Map",
        "2. Historical Map"
    ])

    # Mode change detection to clear selection
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = mode
    if st.session_state.current_mode != mode:
        st.session_state.current_mode = mode
        if "selected_segment_id" in st.session_state:
            del st.session_state.selected_segment_id

    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Map Filters")
    risk_threshold = st.sidebar.slider(
        "Min Risk Level (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        help="Filter road segments by their predicted risk percentage."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Legend")
    st.sidebar.markdown("🔴 **Has Historical Accidents**")
    st.sidebar.markdown("🔵 **No Historical Accidents**")

    # Sidebar XAI Panel (appears when a segment is selected)
    if "selected_segment_id" in st.session_state:
        st.sidebar.markdown("---")
        sid = st.session_state.selected_segment_id
        st.sidebar.subheader(f"🔍 Segment Analysis: {sid}")
        
        with st.sidebar:
            df, model, features, xai_model_cfg = load_xai_data()
            is_prototype_xai = df is None or model is None
            
            with st.spinner("Analyzing segment..."):
                try:
                    if is_prototype_xai:
                        risk_score, top_factors = prototype_xai_result(sid)
                    else:
                        risk_score, top_factors = explain_segment(sid, df, model, features, top_k=5)
                    
                    st.metric("Predicted Risk Score", f"{risk_score*100:.1f}%")
                    
                    # Feature Importance
                    st.markdown("### Top Factors Driving Risk")
                    plot_df = pd.DataFrame(list(top_factors.items()), columns=['Feature', 'Impact'])
                    plot_df['Absolute Impact'] = plot_df['Impact'].abs()
                    plot_df = plot_df.sort_values('Absolute Impact', ascending=True)
                    st.sidebar.bar_chart(plot_df.set_index('Feature')['Impact'])
                    
                    # AI Narrative
                    st.markdown("### 🤖 AI Narrative Explanation")
                    
                    if os.environ.get("GEMINI_API_KEY"):
                        llm_model = model_cfg.get("explanation", {}).get("llm_model", "gemini-1.5-flash")
                        
                        if st.button("✨ Generate AI Narrative"):
                            with st.spinner("Consulting AI..."):
                                stream_gen = generate_explanation(sid, risk_score, top_factors, llm_model, stream=True)
                                if stream_gen:
                                    st.write_stream(stream_gen)
                                else:
                                    st.error("Could not initialize AI Narrative stream.")
                    else:
                        st.warning("GEMINI_API_KEY not found. AI Narrative is disabled.")
                        if is_prototype_xai:
                            st.info(
                                "Prototype narrative (Static): this segment is flagged mainly because sample congestion, nearby activity density, "
                                "and morning speed drop increase the synthetic risk score."
                            )
                except Exception as e:
                    st.sidebar.error(f"Analysis error: {e}")
            
            if st.button("Close Analysis"):
                del st.session_state.selected_segment_id
                st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System")
    if st.sidebar.button("Clear App Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    data_cfg, model_cfg = load_config()

    if mode == "1. Predictive Risk Map":
        st.header("🔮 Predictive Risk Map")
        st.markdown(f"Showing segments with risk ≥ **{risk_threshold}%**. Click a segment to analyze.")

        if missing_paths(RISK_SCORES_PATH, SEGMENTS_PATH):
            st.info("Prototype map data is being shown because processed risk scores or road segments are missing.")
        
        with st.spinner("Loading Map Data..."):
            gdf = load_risk_data()
            
            # Apply Filter
            gdf = gdf[gdf['risk_pct'] >= risk_threshold]
            
            if gdf.empty:
                st.warning(f"No road segments found with risk ≥ {risk_threshold}%.")
                return
            
            st.caption(f"Currently displaying {len(gdf):,} segments.")
            
            # Setup PyDeck Layer
            layer = pdk.Layer(
                "PathLayer",
                gdf,
                id="risk-paths",
                pickable=True,
                get_color="color",
                width_scale=1,
                width_min_pixels=2,
                get_path="path",
                get_width=3,
            )
            
            # Top-down 2D view centered on Bangkok
            view_state = pdk.ViewState(latitude=13.7563, longitude=100.5018, zoom=11, pitch=0, bearing=0)
            
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Segment ID: {segment_id}\nRisk Score: {risk_pct}%\nAccidents: {historical_accidents}"}
            )
            
            # Selection is now handled at the top of main() via 'key="risk_map"'
            st.pydeck_chart(r, on_select="rerun", selection_mode="single-object", use_container_width=True, key="risk_map")

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
                # Date Range (Removed strict min/max constraints to avoid Streamlit range errors)
                min_data_date = accidents['วันที่และเวลาที่เกิดเหตุ'].min().date()
                max_data_date = accidents['วันที่และเวลาที่เกิดเหตุ'].max().date()
                
                date_selection = st.date_input(
                    "📅 Date Range",
                    value=(min_data_date, max_data_date)
                )
            
            with col2:
                # Time of Day Filter
                time_preset = st.selectbox(
                    "⏰ Time of Day", 
                    [
                        "All Day (00:00 - 23:59)", 
                        "Morning Peak (07:00 - 09:59)", 
                        "Evening Peak (16:00 - 19:59)", 
                        "Night (22:00 - 03:59)", 
                        "Custom Hours"
                    ]
                )
                
                if time_preset == "Custom Hours":
                    hour_range = st.slider("Select Hours", 0, 23, (0, 23))
                elif time_preset == "Morning Peak (07:00 - 09:59)":
                    hour_range = (7, 9)
                elif time_preset == "Evening Peak (16:00 - 19:59)":
                    hour_range = (16, 19)
                elif time_preset == "Night (22:00 - 03:59)":
                    hour_range = (22, 3) # Special case handled below
                else:
                    hour_range = (0, 23)
                    
            with col3:
                # Severity filter
                if 'severity_label' in accidents.columns:
                    severities = accidents['severity_label'].dropna().unique().tolist()
                    selected_severity = st.multiselect("🚑 Severity", severities, default=severities)
                else:
                    selected_severity = None

        # Apply Filters
        # 1. Handle date_selection
        if isinstance(date_selection, tuple) and len(date_selection) == 2:
            start_date, end_date = date_selection
        else:
            start_date = date_selection[0] if isinstance(date_selection, tuple) and len(date_selection) > 0 else date_selection
            end_date = start_date

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        mask = (accidents['วันที่และเวลาที่เกิดเหตุ'] >= start_dt) & (accidents['วันที่และเวลาที่เกิดเหตุ'] <= end_dt)
        
        # 2. Handle Hour Range
        if time_preset == "Night (22:00 - 03:59)":
            hour_mask = (accidents['hour'] >= 22) | (accidents['hour'] <= 3)
        else:
            hour_mask = (accidents['hour'] >= hour_range[0]) & (accidents['hour'] <= hour_range[1])
            
        mask = mask & hour_mask
        
        # 3. Handle Severity
        if selected_severity is not None:
            mask = mask & (accidents['severity_label'].isin(selected_severity))

        filtered_acc = accidents[mask].copy()

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
                # 1. Exclude unsnapped accidents (segment_id == -1 means GPS couldn't be matched to a road)
                snapped_acc = filtered_acc[filtered_acc['segment_id'] != -1]
                unsnapped_count = len(filtered_acc) - len(snapped_acc)
                if unsnapped_count > 0:
                    st.info(
                        f"ℹ️ **{unsnapped_count:,} accident(s)** could not be matched to a road segment "
                        f"(GPS too far from any road, >50m) and are excluded from this view. "
                        f"They still appear in the Heatmap."
                    )

                # 2. Aggregate matched accidents by segment_id
                acc_counts = snapped_acc.groupby('segment_id').size().reset_index(name='acc_count')
                
                # 3. Load road segments (cached)
                segments = load_segments()
                if segments is None:
                    st.error("Road segments data not found.")
                    return
                
                gdf_merged = segments.merge(acc_counts, on='segment_id', how='inner')

                if gdf_merged.empty:
                    st.warning("No road segments matched the filtered accidents. Try widening the date or time filters.")
                    return

                gdf_merged = gdf_merged.to_crs("EPSG:4326")

                # Convert to path for PyDeck — must extract coords BEFORE passing to PyDeck
                # because PyDeck cannot serialize Shapely geometry objects
                gdf_merged['path'] = gdf_merged['geometry'].apply(
                    lambda geom: [[c[0], c[1]] for c in geom.coords]
                )

                # Percentile-based color scale so mid-density areas are also visible
                p70 = float(gdf_merged['acc_count'].quantile(0.70))
                p30 = float(gdf_merged['acc_count'].quantile(0.30))
                max_c = int(gdf_merged['acc_count'].max())

                def get_agg_color(count):
                    if count >= p70: return [220, 30, 30, 255]    # Red   — top 30%
                    if count >= p30: return [255, 140, 0, 230]    # Orange — middle 40%
                    return [255, 230, 50, 180]                    # Yellow — bottom 30%

                def get_width(count):
                    return max(3, min(12, int(3 + 9 * (count / max_c))))

                gdf_merged['color'] = gdf_merged['acc_count'].apply(get_agg_color)
                gdf_merged['line_width'] = gdf_merged['acc_count'].apply(get_width)

                st.caption(
                    f"Showing **{len(gdf_merged):,} road segments** with accidents. "
                    f"Red ≥ {p70:.0f} · Orange ≥ {p30:.0f} · Yellow = 1+"
                )

                # ⚠️ Drop geometry column — PyDeck cannot serialize Shapely objects
                plot_df = gdf_merged.drop(columns=['geometry'])

                layers.append(pdk.Layer(
                    "PathLayer",
                    plot_df,
                    pickable=True,
                    get_color="color",
                    width_scale=1,
                    width_min_pixels=2,
                    get_path="path",
                    get_width="line_width",
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

if __name__ == "__main__":
    main()
