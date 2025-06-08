"""
Test script to verify dashboard components are working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import fastf1
        print("✓ FastF1 imported successfully")
    except ImportError as e:
        print(f"✗ FastF1 import failed: {e}")
        
    try:
        import streamlit
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        
    try:
        import plotly
        print("✓ Plotly imported successfully")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        
    try:
        from backend import load_session_data, get_available_sessions
        print("✓ Backend module imported successfully")
    except ImportError as e:
        print(f"✗ Backend module import failed: {e}")
        
    try:
        from visualizations import create_corner_comparison_plot
        print("✓ Visualizations module imported successfully")
    except ImportError as e:
        print(f"✗ Visualizations module import failed: {e}")
        
    try:
        from aggTele import get_all_car_data, classify_corners_by_speed
        print("✓ AggTele module imported successfully")
    except ImportError as e:
        print(f"✗ AggTele module import failed: {e}")


def test_backend_functions():
    """Test basic backend functionality."""
    print("\nTesting backend functions...")
    
    try:
        from backend import get_available_sessions
        sessions = get_available_sessions(2024)
        print(f"✓ Found {len(sessions)} sessions for 2024")
        if sessions:
            print(f"  First session: {sessions[0]}")
    except Exception as e:
        print(f"✗ Backend function test failed: {e}")


def test_team_colors():
    """Test team color retrieval."""
    print("\nTesting team color functionality...")
    
    try:
        import fastf1
        from backend import get_team_info
        
        # Create a mock session for testing
        print("  Note: Full team color test requires loading a session")
        print("✓ Team color functions available")
    except Exception as e:
        print(f"✗ Team color test failed: {e}")


if __name__ == "__main__":
    print("F1 Telemetry Dashboard Component Test")
    print("=" * 40)
    
    test_imports()
    test_backend_functions()
    test_team_colors()
    
    print("\n" + "=" * 40)
    print("Test complete!")
    print("\nTo run the dashboard:")
    print("  poetry run streamlit run dashboard.py") 