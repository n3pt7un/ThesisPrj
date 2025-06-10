# F1 Telemetry Analysis Dashboard

An interactive Streamlit dashboard for visualizing aggregated F1 telemetry data with proper team color support and driver distinction.

## Features

- **Interactive Data Selection**: Choose year, race, session type, and drivers
- **Multiple Visualization Types**:
  - Corner Performance Analysis
  - Lap Performance Tracking
  - Acceleration/Deceleration Patterns
  - Speed Trace Comparisons
  - Throttle/Brake Analysis
  - Sector Times Comparison
- **Team Color Support**: Uses official F1 team colors from FastF1
- **Teammate Distinction**: Different line styles and markers for drivers from the same team
- **Modular Backend**: Easy to modify data extraction and aggregation functions

## Installation

1. Ensure you have Poetry installed
2. Install dependencies:
```bash
poetry install
```

3. The dashboard requires the following dependencies (already added to pyproject.toml):
- streamlit
- plotly
- fastf1
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage

1. Navigate to the StreamlitDashboard directory:
```bash
cd thesisprj/StreamlitDashboard
```

2. Run the dashboard:
```bash
poetry run streamlit run dashboard.py
```

3. The dashboard will open in your default web browser

## Dashboard Structure

### Files

- **dashboard.py**: Main Streamlit application with UI components
- **backend.py**: Data extraction and processing functions
- **visualizations.py**: Plotly-based visualization functions
- **aggTele.py**: Core telemetry aggregation functions

### Key Components

#### Backend Module
The backend module provides modular functions for:
- Loading session data
- Extracting driver telemetry
- Corner classification
- Lap aggregation
- Team information retrieval

Key functions:
- `load_session_data()`: Load F1 session with telemetry
- `extract_driver_corner_data()`: Extract corner performance metrics
- `get_driver_lap_aggregates()`: Aggregate lap-level metrics
- `get_team_info()`: Get team colors and information

#### Visualizations Module
Creates interactive Plotly charts with:
- Proper team color handling
- Distinction between teammates (different line styles/markers)
- Dark theme for better visibility

Key functions:
- `create_corner_comparison_plot()`: Multi-metric corner analysis
- `create_lap_performance_plot()`: Lap progression analysis
- `create_acceleration_heatmap()`: Acceleration patterns heatmap
- `create_speed_trace_comparison()`: Speed profiles with corner markers

## Customization

### Modifying Data Extraction

Edit functions in `backend.py`:
```python
# Example: Add custom metric to lap aggregation
def get_driver_lap_aggregates(session, driver):
    # ... existing code ...
    # Add your custom metric
    lap_aggregates['CustomMetric'] = calculate_custom_metric(car_data)
    return lap_aggregates
```

### Adding New Visualizations

1. Create visualization function in `visualizations.py`
2. Import in `dashboard.py`
3. Add UI controls and display logic

### Changing Visual Styles

Modify style definitions in `visualizations.py`:
```python
LINE_STYLES = ['solid', 'dash', 'dot', 'dashdot']
MARKER_SYMBOLS = ['circle', 'square', 'diamond', 'cross']
```

You can also adjust the default Plotly template and line widths:
```python
from visualizations import set_plotly_template, set_default_line_width
set_plotly_template("plotly")  # switch to light theme
set_default_line_width(3)
```

## Team Color Handling

The dashboard automatically:
1. Retrieves official team colors from FastF1
2. Assigns different line styles to teammates
3. Uses different markers for additional distinction

Example:
- VER (Red Bull): Solid line, circle marker
- PER (Red Bull): Dashed line, square marker

## Performance Considerations

- Data is cached in session state to avoid reloading
- FastF1 cache is enabled for faster data access
- Large sessions may take time to load initially

## Troubleshooting

1. **No data showing**: Ensure the session has telemetry data available
2. **Slow loading**: First-time session loading downloads data; subsequent loads use cache
3. **Missing drivers**: Some sessions may have incomplete driver data

## Future Enhancements

- Export functionality for plots
- Comparison across multiple sessions
- Machine learning insights integration
- Real-time session analysis 

