"""
Backend module for F1 telemetry data extraction and processing.
This module provides modular functions that can be easily modified for different analysis needs.
"""

import fastf1
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from functools import lru_cache
import os

# Import the aggregation functions from aggTele
from aggTele import (
    get_all_car_data,
    get_track_data,
    classify_corners_by_speed,
    aggregation_function,
    extract_corner_telemetry_sections,
    compare_corner_sections
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable FastF1 cache
CACHE_DIR = '../cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)


@lru_cache(maxsize=32)
def get_available_sessions(year: int) -> List[str]:
    """
    Get list of available sessions for a given year.
    
    Parameters:
    -----------
    year : int
        The year to get sessions for
        
    Returns:
    --------
    List[str]
        List of race names
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter out testing sessions
        races = schedule[schedule['EventFormat'] != 'testing']
        return races['EventName'].tolist()
    except Exception as e:
        logger.error(f"Error getting sessions for {year}: {e}")
        return []


def load_session_data(year: int, race_name: str, session_type: str) -> fastf1.core.Session:
    """
    Load session data with telemetry.
    
    Parameters:
    -----------
    year : int
        Year of the session
    race_name : str
        Name of the race
    session_type : str
        Type of session (R, Q, FP1, etc.)
        
    Returns:
    --------
    fastf1.core.Session
        Loaded session object
    """
    try:
        session = fastf1.get_session(year, race_name, session_type)
        session.load(telemetry=True)
        return session
    except Exception as e:
        logger.error(f"Error loading session {race_name} {year} {session_type}: {e}")
        raise


def get_session_drivers(session: fastf1.core.Session) -> List[str]:
    """
    Get list of drivers who participated in the session.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
        
    Returns:
    --------
    List[str]
        List of driver abbreviations
    """
    try:
        drivers = session.laps['Driver'].unique().tolist()
        # Sort by best lap time
        driver_times = []
        for driver in drivers:
            driver_laps = session.laps.pick_drivers(driver)
            if not driver_laps.empty:
                best_time = driver_laps['LapTime'].min()
                driver_times.append((driver, best_time))
        
        # Sort by lap time
        driver_times.sort(key=lambda x: x[1] if pd.notna(x[1]) else pd.Timedelta(hours=1))
        return [driver for driver, _ in driver_times]
    except Exception as e:
        logger.error(f"Error getting session drivers: {e}")
        return []


def extract_driver_corner_data(
    session: fastf1.core.Session,
    driver: str,
    distance_before: float = 100.0,
    distance_after: float = 100.0,
    corner_selection_method: str = 'default'
) -> Dict:
    """
    Extract corner telemetry data for a specific driver.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
    driver : str
        Driver abbreviation
    distance_before : float
        Distance before corner apex to analyze
    distance_after : float
        Distance after corner apex to analyze
    corner_selection_method : str
        Method for selecting corners
        
    Returns:
    --------
    Dict
        Corner analysis results
    """
    try:
        return extract_corner_telemetry_sections(
            session, driver, distance_before, distance_after,
            corner_selection_method=corner_selection_method
        )
    except Exception as e:
        logger.error(f"Error extracting corner data for {driver}: {e}")
        return {}


def get_driver_lap_aggregates(
    session: fastf1.core.Session,
    driver: str
) -> pd.DataFrame:
    """
    Get aggregated lap data for a driver.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
    driver : str
        Driver abbreviation
        
    Returns:
    --------
    pd.DataFrame
        Aggregated lap metrics
    """
    try:
        # Get all car data
        car_data = get_all_car_data(session, driver)
        
        # Apply aggregation function
        lap_aggregates = aggregation_function(car_data)
        
        # Add additional metrics
        lap_aggregates['Driver'] = driver
        lap_aggregates['Session'] = f"{session.event.EventName} - {session.name}"
        
        # Add lap time information
        driver_laps = session.laps.pick_drivers(driver)
        lap_aggregates = lap_aggregates.merge(
            driver_laps[['LapNumber', 'LapTime', 'Compound', 'TyreLife']],
            on='LapNumber',
            how='left'
        )
        
        return lap_aggregates
    except Exception as e:
        logger.error(f"Error getting lap aggregates for {driver}: {e}")
        return pd.DataFrame()


def get_corner_classifications(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Get corner classifications for the circuit.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
        
    Returns:
    --------
    pd.DataFrame
        Corner classifications
    """
    try:
        return classify_corners_by_speed(session)
    except Exception as e:
        logger.error(f"Error classifying corners: {e}")
        return pd.DataFrame()


def get_driver_telemetry_for_lap(
    session: fastf1.core.Session,
    driver: str,
    lap_number: int
) -> pd.DataFrame:
    """
    Get detailed telemetry data for a specific lap.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
    driver : str
        Driver abbreviation
    lap_number : int
        Lap number
        
    Returns:
    --------
    pd.DataFrame
        Telemetry data for the lap
    """
    try:
        driver_laps = session.laps.pick_drivers(driver)
        lap = driver_laps[driver_laps['LapNumber'] == lap_number].iloc[0]
        
        # Get car data
        car_data = lap.get_car_data()
        car_data = car_data.add_distance()
        
        # Get position data
        pos_data = lap.get_pos_data()
        
        # Merge data
        merged_data = car_data.merge_channels(pos_data)
        
        # Add additional information
        merged_data['Driver'] = driver
        merged_data['LapNumber'] = lap_number
        merged_data['LapTime'] = lap.LapTime
        merged_data['Compound'] = lap.Compound
        
        return merged_data
    except Exception as e:
        logger.error(f"Error getting telemetry for {driver} lap {lap_number}: {e}")
        return pd.DataFrame()


def compare_driver_corner_performance(
    corner_data_dict: Dict[str, Dict],
    drivers: List[str]
) -> pd.DataFrame:
    """
    Compare corner performance across multiple drivers.
    
    Parameters:
    -----------
    corner_data_dict : Dict[str, Dict]
        Dictionary with driver corner data
    drivers : List[str]
        List of drivers to compare
        
    Returns:
    --------
    pd.DataFrame
        Comparison DataFrame
    """
    comparison_data = []
    
    for driver in drivers:
        if driver in corner_data_dict and corner_data_dict[driver]:
            driver_comparison = compare_corner_sections(corner_data_dict[driver])
            if not driver_comparison.empty:
                driver_comparison['Driver'] = driver
                comparison_data.append(driver_comparison)
    
    if comparison_data:
        return pd.concat(comparison_data, ignore_index=True)
    else:
        return pd.DataFrame()


def get_team_info(session: fastf1.core.Session, driver: str) -> Dict:
    """
    Get team information for a driver.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
    driver : str
        Driver abbreviation
        
    Returns:
    --------
    Dict
        Team information including color
    """
    try:
        driver_info = session.get_driver(driver)
        team_name = driver_info['TeamName']
        team_color = driver_info['TeamColor']
        
        # Ensure color has # prefix
        if not team_color.startswith('#'):
            team_color = f'#{team_color}'
            
        return {
            'team_name': team_name,
            'team_color': team_color,
            'driver_number': driver_info['DriverNumber']
        }
    except Exception as e:
        logger.error(f"Error getting team info for {driver}: {e}")
        return {
            'team_name': 'Unknown',
            'team_color': '#999999',
            'driver_number': '00'
        }


def calculate_sector_times(
    session: fastf1.core.Session,
    driver: str
) -> pd.DataFrame:
    """
    Calculate sector times for all laps of a driver.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        The session object
    driver : str
        Driver abbreviation
        
    Returns:
    --------
    pd.DataFrame
        Sector times per lap
    """
    try:
        driver_laps = session.laps.pick_drivers(driver)
        
        sector_data = []
        for _, lap in driver_laps.iterrows():
            if pd.notna(lap.Sector1Time) and pd.notna(lap.Sector2Time) and pd.notna(lap.Sector3Time):
                sector_data.append({
                    'LapNumber': lap.LapNumber,
                    'Sector1': lap.Sector1Time.total_seconds(),
                    'Sector2': lap.Sector2Time.total_seconds(),
                    'Sector3': lap.Sector3Time.total_seconds(),
                    'LapTime': lap.LapTime.total_seconds() if pd.notna(lap.LapTime) else None,
                    'Compound': lap.Compound
                })
        
        return pd.DataFrame(sector_data)
    except Exception as e:
        logger.error(f"Error calculating sector times for {driver}: {e}")
        return pd.DataFrame()


# Configuration functions for easy backend modification
def set_cache_directory(directory: str):
    """Set custom cache directory for FastF1."""
    global CACHE_DIR
    CACHE_DIR = directory
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    fastf1.Cache.enable_cache(CACHE_DIR)


def set_logging_level(level: str):
    """Set logging level for the backend."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logger.setLevel(numeric_level) 