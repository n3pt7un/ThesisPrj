import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import fastf1
import fastf1.plotting
from datetime import datetime
import pickle
import os
from typing import List, Dict, Optional, Tuple
import logging

# Import backend functions
from backend import (
    load_session_data,
    get_available_sessions,
    extract_driver_corner_data,
    get_driver_lap_aggregates,
    get_session_drivers,
    get_corner_classifications
)
from visualizations import (
    create_corner_comparison_plot,
    create_lap_performance_plot,
    create_acceleration_heatmap,
    create_speed_trace_comparison,
    create_corner_radar_chart,
    create_throttle_brake_plot
)

# Page configuration
st.set_page_config(
    page_title="F1 Telemetry Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stPlotlyChart {
        background-color: #0E1117;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'session_data' not in st.session_state:
    st.session_state.session_data = None
if 'telemetry_data' not in st.session_state:
    st.session_state.telemetry_data = {}
if 'corner_data' not in st.session_state:
    st.session_state.corner_data = {}

# Title and description
st.title("🏎️ F1 Telemetry Analysis Dashboard")
st.markdown("### Interactive visualization of aggregated F1 telemetry data")

# Sidebar controls
with st.sidebar:
    st.header("📊 Data Selection")
    
    # Year selection
    year = st.selectbox(
        "Select Year",
        options=[2025, 2024, 2023, 2022, 2021],
        index=0
    )
    
    # Get available sessions for the year
    available_sessions = get_available_sessions(year)
    
    # Race selection
    race_name = st.selectbox(
        "Select Race",
        options=available_sessions,
        index=0 if available_sessions else None
    )
    
    # Session type selection
    session_type = st.selectbox(
        "Session Type",
        options=['R', 'Q', 'FP1', 'FP2', 'FP3'],
        index=0
    )
    
    # Load session button
    if st.button("Load Session Data", type="primary"):
        with st.spinner("Loading session data..."):
            try:
                session_data = load_session_data(year, race_name, session_type)
                st.session_state.session_data = session_data
                st.success(f"Loaded {race_name} {year} - {session_type}")
            except Exception as e:
                st.error(f"Error loading session: {str(e)}")
    
    st.divider()
    
    # Driver selection
    if st.session_state.session_data is not None:
        st.header("🏁 Driver Selection")
        
        available_drivers = get_session_drivers(st.session_state.session_data)
        
        # Multi-select for drivers
        selected_drivers = st.multiselect(
            "Select Drivers to Compare",
            options=available_drivers,
            default=available_drivers[:2] if len(available_drivers) >= 2 else available_drivers,
            max_selections=6
        )
        
        st.divider()
        
        # Visualization options
        st.header("📈 Visualization Options")
        
        show_corner_analysis = st.checkbox("Corner Analysis", value=True)
        show_lap_performance = st.checkbox("Lap Performance", value=True)
        show_acceleration_patterns = st.checkbox("Acceleration Patterns", value=True)
        show_speed_traces = st.checkbox("Speed Traces", value=False)
        show_throttle_brake = st.checkbox("Throttle/Brake Analysis", value=False)
        
        # Corner selection for detailed analysis
        if show_corner_analysis:
            st.subheader("Corner Selection")
            corner_selection_method = st.radio(
                "Corner Selection Method",
                options=["Default (Fast/Medium/Slow)", "All Corners", "Custom Selection"],
                index=0
            )
            
            if corner_selection_method == "Custom Selection":
                corner_classifications = get_corner_classifications(st.session_state.session_data)
                selected_corners = st.multiselect(
                    "Select Corners",
                    options=list(range(len(corner_classifications))),
                    format_func=lambda x: f"Corner {x+1} - {corner_classifications.iloc[x]['Class']}"
                )

# Main content area
if st.session_state.session_data is not None and selected_drivers:
    
    # Load telemetry data for selected drivers
    with st.spinner("Processing telemetry data..."):
        for driver in selected_drivers:
            if driver not in st.session_state.telemetry_data:
                try:
                    corner_data = extract_driver_corner_data(
                        st.session_state.session_data, 
                        driver
                    )
                    st.session_state.corner_data[driver] = corner_data
                    
                    lap_data = get_driver_lap_aggregates(
                        st.session_state.session_data,
                        driver
                    )
                    st.session_state.telemetry_data[driver] = lap_data
                except Exception as e:
                    st.error(f"Error loading data for {driver}: {str(e)}")
    
    # Create visualizations
    if show_corner_analysis and st.session_state.corner_data:
        st.header("🏁 Corner Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Corner Performance Comparison")
            corner_comparison_fig = create_corner_comparison_plot(
                st.session_state.corner_data,
                selected_drivers,
                st.session_state.session_data
            )
            st.plotly_chart(corner_comparison_fig, use_container_width=True)
        
        with col2:
            st.subheader("Corner Performance Radar")
            radar_fig = create_corner_radar_chart(
                st.session_state.corner_data,
                selected_drivers,
                st.session_state.session_data
            )
            st.plotly_chart(radar_fig, use_container_width=True)
    
    if show_lap_performance and st.session_state.telemetry_data:
        st.header("📊 Lap Performance Analysis")
        
        lap_perf_fig = create_lap_performance_plot(
            st.session_state.telemetry_data,
            selected_drivers,
            st.session_state.session_data
        )
        st.plotly_chart(lap_perf_fig, use_container_width=True)
    
    if show_acceleration_patterns and st.session_state.corner_data:
        st.header("🚀 Acceleration Patterns")
        
        accel_fig = create_acceleration_heatmap(
            st.session_state.corner_data,
            selected_drivers,
            st.session_state.session_data
        )
        st.plotly_chart(accel_fig, use_container_width=True)
    
    if show_speed_traces:
        st.header("📈 Speed Trace Comparison")
        
        # Lap selection for speed traces
        col1, col2 = st.columns(2)
        with col1:
            max_lap = int(st.session_state.session_data.laps['LapNumber'].max())
            default_lap = min(10, max_lap)  # Ensure default doesn't exceed max
            selected_lap = st.number_input(
                "Select Lap Number",
                min_value=1,
                max_value=max_lap,
                value=default_lap,
                step=1
            )
        
        speed_trace_fig = create_speed_trace_comparison(
            st.session_state.session_data,
            selected_drivers,
            selected_lap
        )
        st.plotly_chart(speed_trace_fig, use_container_width=True)
    
    if show_throttle_brake and st.session_state.telemetry_data:
        st.header("🎮 Throttle/Brake Analysis")
        
        throttle_brake_fig = create_throttle_brake_plot(
            st.session_state.telemetry_data,
            selected_drivers,
            st.session_state.session_data
        )
        st.plotly_chart(throttle_brake_fig, use_container_width=True)
    
    # Summary statistics
    st.header("📊 Summary Statistics")
    
    summary_data = []
    for driver in selected_drivers:
        if driver in st.session_state.telemetry_data:
            driver_data = st.session_state.telemetry_data[driver]
            summary_data.append({
                'Driver': driver,
                'Avg Speed': f"{driver_data['AvgSpeed'].mean():.1f} km/h",
                'Top Speed': f"{driver_data['TopSpeed'].max():.1f} km/h",
                'Avg Gear Changes': f"{driver_data['GearChanges'].mean():.1f}",
                'Avg Throttle': f"{driver_data['AvgThrottleIntensity'].mean():.1f}%",
                'Best Position': int(driver_data['Position'].min())
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

else:
    # Instructions when no data is loaded
    st.info("👈 Please select a session and drivers from the sidebar to begin analysis")
    
    st.markdown("""
    ### How to use this dashboard:
    
    1. **Select Year and Race**: Choose the season and specific race you want to analyze
    2. **Load Session Data**: Click the button to load telemetry data
    3. **Choose Drivers**: Select up to 6 drivers to compare
    4. **Customize Visualizations**: Toggle different analysis views
    5. **Explore the Data**: Interact with the plots to gain insights
    
    ### Features:
    - **Corner Analysis**: Compare driver performance through different corner types
    - **Lap Performance**: Track lap times and consistency
    - **Acceleration Patterns**: Analyze braking and acceleration zones
    - **Speed Traces**: Compare speed profiles across the lap
    - **Throttle/Brake Analysis**: Understand driver inputs
    """)

# Footer
st.markdown("---")
st.markdown("Built with FastF1 and Streamlit | Data source: F1 Telemetry API") 