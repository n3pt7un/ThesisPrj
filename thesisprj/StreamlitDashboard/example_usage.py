"""
Example usage of the F1 Telemetry Dashboard components.
This script demonstrates how to use the backend functions programmatically.
"""

import pandas as pd
from backend import (
    load_session_data,
    get_available_sessions,
    extract_driver_corner_data,
    get_driver_lap_aggregates,
    get_team_info,
    get_corner_classifications
)
from visualizations import (
    create_corner_comparison_plot,
    create_lap_performance_plot
)

def main():
    # Example 1: Get available sessions
    print("Example 1: Getting available sessions for 2024")
    sessions_2024 = get_available_sessions(2024)
    print(f"Found {len(sessions_2024)} sessions in 2024")
    print(f"First 5 sessions: {sessions_2024[:5]}")
    print()
    
    # Example 2: Load a specific session
    print("Example 2: Loading a session")
    try:
        # Load the first available session
        if sessions_2024:
            session = load_session_data(2024, sessions_2024[0], 'R')
            print(f"Loaded: {session.event.EventName} - {session.name}")
            print(f"Number of drivers: {len(session.drivers)}")
            print()
    except Exception as e:
        print(f"Could not load session: {e}")
        return
    
    # Example 3: Get driver information
    print("Example 3: Getting driver and team information")
    drivers = session.drivers[:3]  # First 3 drivers
    for driver in drivers:
        team_info = get_team_info(session, driver)
        print(f"{driver}: {team_info['team_name']} - Color: {team_info['team_color']}")
    print()
    
    # Example 4: Get corner classifications
    print("Example 4: Corner classifications")
    corners = get_corner_classifications(session)
    if not corners.empty:
        print(f"Number of corners: {len(corners)}")
        print("\nCorner types:")
        print(corners['Class'].value_counts())
    print()
    
    # Example 5: Extract driver lap aggregates
    print("Example 5: Driver lap aggregates")
    driver = drivers[0] if drivers else None
    if driver:
        lap_data = get_driver_lap_aggregates(session, driver)
        if not lap_data.empty:
            print(f"Lap data for {driver}:")
            print(f"  Number of laps: {len(lap_data)}")
            print(f"  Average speed: {lap_data['AvgSpeed'].mean():.1f} km/h")
            print(f"  Top speed: {lap_data['TopSpeed'].max():.1f} km/h")
            print(f"  Average gear changes: {lap_data['GearChanges'].mean():.1f}")
    print()
    
    # Example 6: Extract corner telemetry
    print("Example 6: Corner telemetry analysis")
    if driver:
        corner_data = extract_driver_corner_data(
            session, driver,
            distance_before=100,
            distance_after=100,
            corner_selection_method='default'
        )
        if corner_data:
            print(f"Analyzed {len(corner_data)} corners for {driver}")
            for corner_name, data in list(corner_data.items())[:1]:  # First corner
                print(f"\n{corner_name}:")
                print(f"  Apex distance: {data['corner_info']['apex_distance']:.1f}m")
                print(f"  Min speed: {data['corner_info']['min_speed']:.1f} km/h")
                print(f"  Speed class: {data['corner_info']['speed_class']}")
    
    print("\n" + "="*50)
    print("To run the full dashboard with UI:")
    print("  poetry run streamlit run dashboard.py")


if __name__ == "__main__":
    main() 