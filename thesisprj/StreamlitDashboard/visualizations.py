"""
Visualization module for F1 telemetry dashboard.
Uses Plotly for interactive plots with proper team color handling.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import fastf1
import fastf1.plotting
from typing import List, Dict, Optional, Tuple
import logging

from backend import (
    get_team_info,
    compare_driver_corner_performance,
    get_driver_telemetry_for_lap,
    calculate_sector_times
)

logger = logging.getLogger(__name__)

# Define line styles for distinguishing teammates
LINE_STYLES = ['solid', 'dash', 'dot', 'dashdot']
MARKER_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up']

PLOTLY_TEMPLATE = "plotly_dark"
DEFAULT_LINE_WIDTH = 2

def set_plotly_template(template: str):
    global PLOTLY_TEMPLATE
    PLOTLY_TEMPLATE = template

def set_default_line_width(width: int):
    global DEFAULT_LINE_WIDTH
    DEFAULT_LINE_WIDTH = width



def get_driver_style(session, driver: str, driver_index: int, team_colors_used: Dict[str, int]) -> Dict:
    """
    Get styling for a driver, ensuring teammates have different line styles.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
    driver : str
        Driver abbreviation
    driver_index : int
        Index of the driver in the selection
    team_colors_used : Dict[str, int]
        Dictionary tracking how many times each team color has been used
        
    Returns:
    --------
    Dict
        Styling dictionary with color, line style, and marker
    """
    team_info = get_team_info(session, driver)
    team_color = team_info['team_color']
    
    # Track how many times this team color has been used
    if team_color not in team_colors_used:
        team_colors_used[team_color] = 0
    
    style_index = team_colors_used[team_color]
    team_colors_used[team_color] += 1
    
    return {
        'color': team_color,
        'line_style': LINE_STYLES[style_index % len(LINE_STYLES)],
        'marker_symbol': MARKER_SYMBOLS[style_index % len(MARKER_SYMBOLS)],
        'team_name': team_info['team_name'],
        'driver_number': team_info['driver_number']
    }


def create_corner_comparison_plot(
    corner_data_dict: Dict[str, Dict],
    drivers: List[str],
    session
) -> go.Figure:
    """
    Create an interactive corner performance comparison plot.
    
    Parameters:
    -----------
    corner_data_dict : Dict[str, Dict]
        Dictionary with driver corner data
    drivers : List[str]
        List of drivers to compare
    session : fastf1.core.Session
        The session object
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Get comparison data
    comparison_df = compare_driver_corner_performance(corner_data_dict, drivers)
    
    if comparison_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No corner data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Entry vs Exit Speed Comparison', 'Acceleration Patterns',
                       'Throttle Application Strategy', 'Speed by Corner Type'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    team_colors_used = {}
    
    for i, driver in enumerate(drivers):
        driver_data = comparison_df[comparison_df['Driver'] == driver]
        if driver_data.empty:
            continue
            
        style = get_driver_style(session, driver, i, team_colors_used)
        
        # Entry vs Exit Speed - Bar Chart
        into_data = driver_data[driver_data['Section'] == 'into_turn']
        out_data = driver_data[driver_data['Section'] == 'out_of_turn']
        
        if not into_data.empty and not out_data.empty:
            avg_entry_speed = into_data['EntrySpeed'].mean()
            avg_exit_speed = out_data['ExitSpeed'].mean()
            
            # Add entry speed bar
            fig.add_trace(
                go.Bar(
                    x=[f"{driver} Entry"],
                    y=[avg_entry_speed],
                    name=f"{driver} Entry" if i == 0 else None,
                    marker_color=style['color'],
                    opacity=0.7,
                    showlegend=(i == 0),
                    legendgroup="entry"
                ),
                row=1, col=1
            )
            
            # Add exit speed bar
            fig.add_trace(
                go.Bar(
                    x=[f"{driver} Exit"],
                    y=[avg_exit_speed],
                    name=f"{driver} Exit" if i == 0 else None,
                    marker_color=style['color'],
                    opacity=1.0,
                    showlegend=(i == 0),
                    legendgroup="exit"
                ),
                row=1, col=1
            )
        
        # Acceleration patterns
        fig.add_trace(
            go.Box(
                y=driver_data['MaxAcceleration'],
                x=driver_data['Section'],
                name=driver,
                marker_color=style['color'],
                line=dict(width=DEFAULT_LINE_WIDTH),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Throttle application - Scatter Plot
        avg_throttle_into = driver_data[driver_data['Section'] == 'into_turn']['AvgThrottleIntensity'].mean()
        avg_throttle_out = driver_data[driver_data['Section'] == 'out_of_turn']['AvgThrottleIntensity'].mean()
        
        if not pd.isna(avg_throttle_into) and not pd.isna(avg_throttle_out):
            fig.add_trace(
                go.Scatter(
                    x=[avg_throttle_into],
                    y=[avg_throttle_out],
                    mode='markers',
                    name=f"{driver} ({style['team_name']})",
                    marker=dict(
                        color=style['color'],
                        size=12,
                        symbol=style['marker_symbol']
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Speed by corner type
        for corner_type in ['Fast', 'Medium', 'Slow']:
            corner_type_data = driver_data[driver_data['SpeedClass'] == corner_type]
            if not corner_type_data.empty:
                avg_speed = corner_type_data['AvgSpeed'].mean()
                fig.add_trace(
                    go.Scatter(
                        x=[corner_type],
                        y=[avg_speed],
                        mode='markers+lines',
                        name=driver if corner_type == 'Fast' else None,
                        marker=dict(
                            color=style['color'],
                            size=12,
                            symbol=style['marker_symbol']
                        ),
                        line=dict(
                            color=style['color'],
                            dash=style['line_style']
                        ),
                        showlegend=(corner_type == 'Fast')
                    ),
                    row=2, col=2
                )
    
    # Add reference line for throttle plot (diagonal line where x=y)
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode='lines',
            name='Equal Throttle Line',
            line=dict(color='gray', dash='dash', width=1),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Driver - Speed Type", row=1, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
    
    fig.update_xaxes(title_text="Section", row=1, col=2)
    fig.update_yaxes(title_text="Max Acceleration (m/s²)", row=1, col=2)
    
    fig.update_xaxes(title_text="Throttle Into Corner (%)", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Throttle Out of Corner (%)", row=2, col=1, range=[0, 100])
    
    fig.update_xaxes(title_text="Corner Type", row=2, col=2)
    fig.update_yaxes(title_text="Average Speed (km/h)", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Corner Performance Analysis",
        template=PLOTLY_TEMPLATE
    )
    
    return fig


def create_lap_performance_plot(
    telemetry_data_dict: Dict[str, pd.DataFrame],
    drivers: List[str],
    session
) -> go.Figure:
    """
    Create lap time and performance progression plot.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Lap Times', 'Average Speed Progression',
                       'Gear Changes per Lap', 'Tire Degradation'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": True}]]
    )
    
    team_colors_used = {}
    
    for i, driver in enumerate(drivers):
        if driver not in telemetry_data_dict:
            continue
            
        driver_data = telemetry_data_dict[driver]
        if driver_data.empty:
            continue
            
        style = get_driver_style(session, driver, i, team_colors_used)
        
        # Lap times
        valid_laps = driver_data[driver_data['LapTime'].notna()]
        if not valid_laps.empty:
            lap_times_seconds = valid_laps['LapTime'].dt.total_seconds()
            
            fig.add_trace(
                go.Scatter(
                    x=valid_laps['LapNumber'],
                    y=lap_times_seconds,
                    mode='lines+markers',
                    name=f"{driver} ({style['team_name']})",
                    line=dict(
                        color=style['color'],
                        dash=style['line_style'],
                        width=DEFAULT_LINE_WIDTH
                    ),
                    marker=dict(
                        color=style['color'],
                        size=6,
                        symbol=style['marker_symbol']
                    )
                ),
                row=1, col=1
            )
        
        # Average speed progression
        fig.add_trace(
            go.Scatter(
                x=driver_data['LapNumber'],
                y=driver_data['AvgSpeed'],
                mode='lines',
                name=driver,
                line=dict(
                    color=style['color'],
                    dash=style['line_style'],
                    width=DEFAULT_LINE_WIDTH
                ),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Gear changes
        fig.add_trace(
            go.Scatter(
                x=driver_data['LapNumber'],
                y=driver_data['GearChanges'],
                mode='lines+markers',
                name=driver,
                line=dict(
                    color=style['color'],
                    dash=style['line_style']
                ),
                marker=dict(
                    color=style['color'],
                    size=4,
                    symbol=style['marker_symbol']
                ),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Tire degradation (if available)
        if 'TyreLife' in driver_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=driver_data['LapNumber'],
                    y=driver_data['TyreLife'],
                    mode='lines',
                    name=driver,
                    line=dict(
                        color=style['color'],
                        dash=style['line_style']
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Update axes
    fig.update_xaxes(title_text="Lap Number", row=1, col=1)
    fig.update_yaxes(title_text="Lap Time (seconds)", row=1, col=1)
    
    fig.update_xaxes(title_text="Lap Number", row=1, col=2)
    fig.update_yaxes(title_text="Average Speed (km/h)", row=1, col=2)
    
    fig.update_xaxes(title_text="Lap Number", row=2, col=1)
    fig.update_yaxes(title_text="Gear Changes", row=2, col=1)
    
    fig.update_xaxes(title_text="Lap Number", row=2, col=2)
    fig.update_yaxes(title_text="Tyre Life (laps)", row=2, col=2)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Lap Performance Analysis",
        template=PLOTLY_TEMPLATE
    )
    
    return fig


def create_acceleration_heatmap(
    corner_data_dict: Dict[str, Dict],
    drivers: List[str],
    session
) -> go.Figure:
    """
    Create acceleration/deceleration heatmap for corner sections.
    """
    # Prepare data for heatmap
    comparison_df = compare_driver_corner_performance(corner_data_dict, drivers)
    
    if comparison_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No acceleration data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Create pivot tables for acceleration data
    accel_pivot = comparison_df.pivot_table(
        values='MaxAcceleration',
        index='Driver',
        columns=['SpeedClass', 'Section'],
        aggfunc='mean'
    )
    
    decel_pivot = comparison_df.pivot_table(
        values='MaxDeceleration',
        index='Driver',
        columns=['SpeedClass', 'Section'],
        aggfunc='mean'
    )
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Maximum Acceleration', 'Maximum Deceleration'),
        horizontal_spacing=0.15
    )
    
    # Acceleration heatmap
    fig.add_trace(
        go.Heatmap(
            z=accel_pivot.values,
            x=[f"{col[0]} - {col[1]}" for col in accel_pivot.columns],
            y=accel_pivot.index,
            colorscale='RdYlGn',
            text=np.round(accel_pivot.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(title="m/s²", x=0.45)
        ),
        row=1, col=1
    )
    
    # Deceleration heatmap (absolute values)
    fig.add_trace(
        go.Heatmap(
            z=np.abs(decel_pivot.values),
            x=[f"{col[0]} - {col[1]}" for col in decel_pivot.columns],
            y=decel_pivot.index,
            colorscale='Reds',
            text=np.round(np.abs(decel_pivot.values), 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True,
            colorbar=dict(title="m/s²", x=1.02)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Corner Type - Section", row=1, col=1)
    fig.update_xaxes(title_text="Corner Type - Section", row=1, col=2)
    fig.update_yaxes(title_text="Driver", row=1, col=1)
    
    fig.update_layout(
        height=500,
        title_text="Acceleration/Deceleration Patterns",
        template=PLOTLY_TEMPLATE
    )
    
    return fig


def create_speed_trace_comparison(
    session,
    drivers: List[str],
    lap_number: int
) -> go.Figure:
    """
    Create speed trace comparison for specific lap.
    """
    fig = go.Figure()
    team_colors_used = {}
    
    max_distance = 0
    
    for i, driver in enumerate(drivers):
        try:
            telemetry = get_driver_telemetry_for_lap(session, driver, lap_number)
            
            if telemetry.empty:
                continue
                
            style = get_driver_style(session, driver, i, team_colors_used)
            
            # Update max distance
            max_distance = max(max_distance, telemetry['Distance'].max())
            
            # Add speed trace
            fig.add_trace(
                go.Scatter(
                    x=telemetry['Distance'],
                    y=telemetry['Speed'],
                    mode='lines',
                    name=f"{driver} ({style['team_name']})",
                    line=dict(
                        color=style['color'],
                        dash=style['line_style'],
                        width=DEFAULT_LINE_WIDTH
                    )
                )
            )
            
        except Exception as e:
            logger.error(f"Error getting speed trace for {driver}: {e}")
    
    # Add corner markers
    try:
        circuit_info = session.get_circuit_info()
        corners = circuit_info.corners
        
        for _, corner in corners.iterrows():
            fig.add_vline(
                x=corner['Distance'],
                line_dash="dot",
                line_color="gray",
                opacity=0.5
            )
            fig.add_annotation(
                x=corner['Distance'],
                y=300,
                text=f"T{corner['Number']}",
                showarrow=False,
                yshift=10
            )
    except:
        pass
    
    fig.update_xaxes(title_text="Distance (m)")
    fig.update_yaxes(title_text="Speed (km/h)")
    
    fig.update_layout(
        height=500,
        title_text=f"Speed Trace Comparison - Lap {lap_number}",
        template=PLOTLY_TEMPLATE,
        hovermode='x unified'
    )
    
    return fig


def create_corner_radar_chart(
    corner_data_dict: Dict[str, Dict],
    drivers: List[str],
    session
) -> go.Figure:
    """
    Create radar chart comparing driver performance metrics.
    """
    # Get comparison data
    comparison_df = compare_driver_corner_performance(corner_data_dict, drivers)
    
    if comparison_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for radar chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Define metrics for radar chart
    metrics = ['AvgSpeed', 'MaxAcceleration', 'AvgThrottleIntensity', 'EntrySpeed', 'ExitSpeed']
    metric_labels = ['Avg Speed', 'Max Acceleration', 'Avg Throttle', 'Entry Speed', 'Exit Speed']
    
    fig = go.Figure()
    team_colors_used = {}
    
    for i, driver in enumerate(drivers):
        driver_data = comparison_df[comparison_df['Driver'] == driver]
        
        if driver_data.empty:
            continue
            
        style = get_driver_style(session, driver, i, team_colors_used)
        
        # Calculate normalized values
        values = []
        for metric in metrics:
            all_values = comparison_df[metric].dropna()
            if not all_values.empty and len(all_values) > 1:
                driver_value = driver_data[metric].mean()
                # Normalize to 0-100 scale
                min_val = all_values.min()
                max_val = all_values.max()
                if max_val > min_val:
                    normalized = ((driver_value - min_val) / (max_val - min_val)) * 100
                else:
                    normalized = 50
                values.append(normalized)
            else:
                values.append(50)
        
        # Close the radar chart
        values += values[:1]
        
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metric_labels + metric_labels[:1],
                fill='toself',
                fillcolor=style['color'],
                opacity=0.3,
                line=dict(
                    color=style['color'],
                    dash=style['line_style'],
                    width=DEFAULT_LINE_WIDTH
                ),
                name=f"{driver} ({style['team_name']})"
            )
        )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Driver Performance Radar",
        template=PLOTLY_TEMPLATE,
        height=500
    )
    
    return fig


def create_throttle_brake_plot(
    telemetry_data_dict: Dict[str, pd.DataFrame],
    drivers: List[str],
    session
) -> go.Figure:
    """
    Create throttle and brake application analysis.
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Throttle Application by Lap', 'Throttle Intensity Distribution'),
        row_heights=[0.6, 0.4]
    )
    
    team_colors_used = {}
    
    for i, driver in enumerate(drivers):
        if driver not in telemetry_data_dict:
            continue
            
        driver_data = telemetry_data_dict[driver]
        if driver_data.empty:
            continue
            
        style = get_driver_style(session, driver, i, team_colors_used)
        
        # Throttle progression
        fig.add_trace(
            go.Scatter(
                x=driver_data['LapNumber'],
                y=driver_data['AvgThrottleIntensity'],
                mode='lines+markers',
                name=f"{driver} ({style['team_name']})",
                line=dict(
                    color=style['color'],
                    dash=style['line_style'],
                    width=DEFAULT_LINE_WIDTH
                ),
                marker=dict(
                    color=style['color'],
                    size=6,
                    symbol=style['marker_symbol']
                )
            ),
            row=1, col=1
        )
        
        # Throttle distribution
        fig.add_trace(
            go.Box(
                y=driver_data['AvgThrottleIntensity'],
                name=driver,
                marker_color=style['color'],
                boxmean='sd',
                showlegend=False
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Lap Number", row=1, col=1)
    fig.update_yaxes(title_text="Average Throttle (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="Driver", row=2, col=1)
    fig.update_yaxes(title_text="Throttle Intensity (%)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        title_text="Throttle Application Analysis",
        template=PLOTLY_TEMPLATE
    )
    
    return fig


def create_sector_times_plot(
    session,
    drivers: List[str]
) -> go.Figure:
    """
    Create sector times comparison plot.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Sector 1', 'Sector 2', 'Sector 3'),
        shared_yaxes=True
    )
    
    team_colors_used = {}
    
    for i, driver in enumerate(drivers):
        sector_data = calculate_sector_times(session, driver)
        
        if sector_data.empty:
            continue
            
        style = get_driver_style(session, driver, i, team_colors_used)
        
        for sector_num in range(1, 4):
            sector_col = f'Sector{sector_num}'
            
            fig.add_trace(
                go.Scatter(
                    x=sector_data['LapNumber'],
                    y=sector_data[sector_col],
                    mode='lines+markers',
                    name=f"{driver} ({style['team_name']})" if sector_num == 1 else None,
                    line=dict(
                        color=style['color'],
                        dash=style['line_style'],
                        width=DEFAULT_LINE_WIDTH
                    ),
                    marker=dict(
                        color=style['color'],
                        size=4,
                        symbol=style['marker_symbol']
                    ),
                    showlegend=(sector_num == 1)
                ),
                row=1, col=sector_num
            )
    
    for col in range(1, 4):
        fig.update_xaxes(title_text="Lap Number", row=1, col=col)
    
    fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
    
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="Sector Times Comparison",
        template=PLOTLY_TEMPLATE
    )
    
    return fig
