import sys
from pathlib import Path
import types
import pandas as pd
import plotly.graph_objects as go

# Make StreamlitDashboard modules importable
DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "thesisprj" / "StreamlitDashboard"
sys.path.append(str(DASHBOARD_DIR))

import backend
import visualizations


def test_imports():
    assert isinstance(backend, types.ModuleType)
    assert isinstance(visualizations, types.ModuleType)


def test_get_available_sessions():
    sessions = backend.get_available_sessions(2024)
    assert isinstance(sessions, list)
    assert len(sessions) > 0


def test_create_corner_plot(monkeypatch):
    sample_df = pd.DataFrame({
        "Driver": ["VER", "VER", "HAM", "HAM"],
        "Section": ["into_turn", "out_of_turn", "into_turn", "out_of_turn"],
        "EntrySpeed": [100, None, 98, None],
        "ExitSpeed": [None, 120, None, 118],
        "MaxAcceleration": [2.5, 2.3, 2.4, 2.1],
        "AvgThrottleIntensity": [80, 90, 78, 88],
        "SpeedClass": ["Fast", "Fast", "Slow", "Slow"],
    })

    def fake_compare(corner_data_dict, drivers):
        return sample_df[sample_df["Driver"].isin(drivers)]

    monkeypatch.setattr(backend, "compare_driver_corner_performance", fake_compare)
    monkeypatch.setattr(visualizations, "get_team_info", lambda session, driver: {"team_name": "Test", "team_color": "#123456", "driver_number": "01"})

    fig = visualizations.create_corner_comparison_plot({}, ["VER", "HAM"], session=None)
    assert isinstance(fig, go.Figure)
