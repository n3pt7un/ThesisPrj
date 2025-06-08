# %%
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

# Import tqdm for progress bar
from tqdm import tqdm

import logging
import pickle

# %% [markdown]
# The get_all_car_data function takes a FastF1 session object and driver identifier as input, and returns a pandas DataFrame containing detailed telemetry data for all laps driven by that driver. It combines car telemetry (speed, RPM, gear, etc), lap information, pit status, tire data and calculates acceleration for each data point.
# 

# %%
def get_all_car_data(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    driver_laps = session.laps.pick_drivers(driver).copy()
    
    driver_laps.loc[:, 'InPit'] = driver_laps['PitOutTime'].notna()
    driver_laps.loc[:, 'OutPit'] = driver_laps['PitInTime'].notna()
    
    
    driver_laps = driver_laps.pick_accurate()
    
    race_car_data = pd.DataFrame()
    for _, lap in driver_laps.iterrows():
        
        # !!! This includes interpolated data since pos and car data are not available at the same time
        # Ideally we should merge the pos and car data to get the actual data using resampling ??
        # ------------------------------
        car_data = lap.get_car_data()   
        # ------------------------------   
        
        car_data = car_data.add_distance()
        
        # if 'X' not in car_data.columns or 'Y' not in car_data.columns:
        #     print(f"Warning: X,Y coordinates missing for lap {lap.LapNumber}")
        #     continue
        #     
        # # Add coordinate-based calculations
        # car_data['X_coord'] = car_data['X']  # Explicit naming
        # car_data['Y_coord'] = car_data['Y']  # Explicit naming
        # 
        # # Calculate coordinate-based distance between consecutive points
        # car_data['Coord_Distance_Delta'] = np.sqrt(
        #     (car_data['X_coord'].diff())**2 + (car_data['Y_coord'].diff())**2
        # ).fillna(0)
        # 
        # car_data['Distance_Driven'] = car_data['Distance']
        # 
        # # Cumulative coordinate-based distance (alternative to add_distance())
        # car_data['Distance'] = car_data['Coord_Distance_Delta'].cumsum()/ 10 # X Y measured as 10cm per unit - Raw distance is in meters
        
        car_data['LapNumber'] = lap.LapNumber
        car_data['LapNumber'] = car_data['LapNumber'].astype(int)
        car_data['Position'] = lap.Position
        car_data['OutPit'] = car_data['LapNumber'].isin(driver_laps[driver_laps['OutPit'] == True]['LapNumber'])
        car_data['InPit'] = car_data['LapNumber'].isin(driver_laps[driver_laps['InPit'] == True]['LapNumber'])
        car_data['Driver'] = driver
        car_data['Session'] = session.name
        car_data['TyreCompound'] = lap.Compound
        car_data['TyreLife'] = lap.TyreLife
        # car_data['Acceleration'] = np.gradient(car_data.Speed/3.6)  incorrect implementation, 
        # because speed is not linear in time
        time_seconds = car_data['Time'].dt.total_seconds()
        car_data['Acceleration(m/s^2)'] = np.gradient(car_data.Speed/3.6, time_seconds)
        race_car_data = pd.concat([race_car_data, car_data])
    return race_car_data

# %%
def get_all_car_data(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    # ----------------------------------------------------------------
    # 1) Build reference spline once, from the session's fastest lap
    # ----------------------------------------------------------------
    # Pick the fastest lap and get the car and position data
    fastest = session.laps.pick_fastest()
    car_data = fastest.get_car_data()   
    pos_data = fastest.get_pos_data()
    # Merge the car and position data
    merged_data = car_data.merge_channels(pos_data)
    # Add distance to the merged data
    merged_data = merged_data.add_distance()
    
    # Interpolate missing data
    merged_data = merged_data.set_index('Time')
    merged_data[['X','Y']] = merged_data[['X','Y']].interpolate(method='time').ffill().bfill()
    merged_data = merged_data.reset_index()

    # Get the distance, x, and y values
    u_ref = merged_data['Distance'].values
    x_ref = merged_data['X'].values
    y_ref = merged_data['Y'].values

    # fit a degree-3 spline (s=0 means interpolate exactly)
    tck, _ = splprep([x_ref, y_ref], u=u_ref, s=0, k=3)

    # sample every 0.1 m along-track
    u_fine = np.linspace(0, u_ref.max(), int(u_ref.max() / 0.1))
    x_fine, y_fine = splev(u_fine, tck)

    # build a KD–Tree so we can snap points in bulk
    track_tree = cKDTree(np.column_stack((x_fine, y_fine)))

    # ----------------------------------------------------------------
    # 2) Now loop over your driver’s laps, merge pos+car, and snap
    # ----------------------------------------------------------------
    driver_laps = (
        session.laps
               .pick_drivers(driver)
               .assign(InPit=lambda df: df['PitOutTime'].notna(),
                       OutPit=lambda df: df['PitInTime'].notna())
               .pick_accurate()
    )

    all_data = []
    for _, lap in driver_laps.iterrows():
        car_data = lap.get_car_data()
        pos_data = lap.get_pos_data()

        merged = car_data.merge_channels(pos_data)
        merged = merged.add_distance()
        merged['raw_distance'] = merged['Distance']

        t = merged['Time'].dt.total_seconds()
        merged['X'] = merged['X'].interpolate(method='linear', x=t).ffill().bfill()
        merged['Y'] = merged['Y'].interpolate(method='linear', x=t).ffill().bfill()

        
        # vectorized snap: find nearest spline-sample to each (X,Y)
        pts = np.column_stack((merged['X'].values, merged['Y'].values))
        _, idx = track_tree.query(pts, k=1)

        # assign new columns
        merged['X_snap'] = x_fine[idx]
        merged['Y_snap'] = y_fine[idx]

        merged['Distance'] = u_fine[idx]
        
        merged['LapNumber']   = int(lap.LapNumber)
        merged['Position']    = lap.Position
        merged['OutPit']      = merged['LapNumber'].isin(
                                   driver_laps.loc[driver_laps['OutPit'], 'LapNumber']
                               )
        merged['InPit']       = merged['LapNumber'].isin(
                                   driver_laps.loc[driver_laps['InPit'],  'LapNumber']
                               )
        merged['Driver']      = driver
        merged['Session']     = session.name
        merged['TyreCompound']= lap.Compound
        merged['TyreLife']    = lap.TyreLife

        # proper acceleration
        tsec = merged['Time'].dt.total_seconds().values
        merged['Acceleration(m/s^2)'] = np.gradient(merged['Speed'].values/3.6, tsec)

        all_data.append(merged)

    return pd.concat(all_data, ignore_index=True)

# %%
def get_track_data(session: fastf1.core.Session) -> pd.DataFrame:
    info = session.get_circuit_info()
    corners = info.corners
    return corners

# %%
# Example usage - commented out for module import
# fastf1.Cache.enable_cache('./cache')
# china25 = fastf1.get_session(2025, 'Chinese Grand Prix', 'R')
# china25.load(telemetry=True)

# %%
# max_china = get_all_car_data(china25, 'VER')

# %%
# max_china.to_csv('max_china.csv')

# %% [markdown]
# # Exploring aggregation metrics for telemetry data 
# 
# 

# %%
def calculate_lap_gear_changes(gear_series):
    """
    Calculates the number of gear changes for a given pandas Series
    (representing one lap's gear data).
    Assumes the series is ordered chronologically for the lap.
    """
    # Ensure we're working with a clean Series index within the group
    gear_series = gear_series.reset_index(drop=True)

    if gear_series.shape[0] < 2:
        # Need at least two data points to potentially have a change
        return 0

    # Identify points where gear is different from the previous point
    is_change = gear_series.ne(gear_series.shift())

    # Assign a group ID to each block of consecutive gears
    # The first block starts with ID 1
    gear_block_groups = is_change.cumsum()

    # Count the number of distinct gear blocks
    num_blocks = gear_block_groups.nunique()

    # The number of changes is the number of blocks minus 1
    # If num_blocks is 1, gear was constant (or only one data point), so 0 changes.
    num_changes = max(0, num_blocks - 1)

    return num_changes

# %%
def classify_corners_by_speed(
    session: fastf1.core.Session,
    distance_col: str = "Distance",
    angle_col: str = "Angle",
    speed_col: str = "Speed", 
    window: float = 25.0
) -> pd.DataFrame:
    """
    Classify corners into fast, medium, or slow based on both angle and speed analysis.
    
    This function takes a FastF1 session object and automatically:
    1. Extracts circuit corner information
    2. Classifies corners by angle (Fast/Medium/Slow based on turning angle)
    3. Gets fastest lap telemetry data
    4. Classifies corners by speed (based on minimum speed quantiles)
    
    Parameters:
    - session: fastf1.core.Session
        FastF1 session object with loaded telemetry data
    - distance_col: str
        Column name for apex distance (default: "Distance")
    - angle_col: str  
        Column name for corner angle (default: "Angle")
    - speed_col: str
        Column name for speed data (default: "Speed")
    - window: float
        Distance in meters before and after apex to search for min speed (default: 25.0)
        
    Returns:
    - pd.DataFrame with columns: 
        - 'ApexDistance': Distance along track where corner apex occurs
        - 'Angle': Original turning angle 
        - 'AbsAngle': Absolute value of turning angle
        - 'CornerAngle': Classification based on angle (Fast/Medium/Slow)
        - 'MinSpeed': Minimum speed found in the corner window
        - 'Class': Classification based on speed quantiles (Fast/Medium/Slow)
    """
    
    # Extract circuit information and corners
    info = session.get_circuit_info()
    corners = info.corners.copy()
    
    # Add absolute angle and angle-based classification
    corners['AbsAngle'] = corners[angle_col].abs()
    corners['CornerAngle'] = corners['AbsAngle'].apply(
        lambda x: 'Fast' if x < 50 else 'Medium' if (x > 50 and x < 100) else 'Slow'
    )
    
    # Get fastest lap telemetry data
    fastest_lap_driver = session.laps.pick_fastest().Driver
    telemetry_df = get_all_car_data(session, fastest_lap_driver)
    
    # Perform speed-based corner classification
    results = []
    
    for i, row in corners.iterrows():
        apex_dist = row[distance_col]
        angle = row[angle_col]
        abs_angle = row['AbsAngle']
        corner_angle_class = row['CornerAngle']
        
        # Find telemetry data within the window around the apex
        mask = (telemetry_df['Distance'] >= apex_dist - window) & \
               (telemetry_df['Distance'] <= apex_dist + window)
        segment = telemetry_df[mask]
        
        if not segment.empty:
            min_speed = segment[speed_col].min()
        else:
            min_speed = np.nan
            
        results.append({
            'ApexDistance': apex_dist,
            'Angle': angle,
            'AbsAngle': abs_angle,
            'CornerAngle': corner_angle_class,
            'MinSpeed': min_speed
        })
    
    df_results = pd.DataFrame(results)
    
    # Classify corners by speed using quantiles
    valid_speeds = df_results['MinSpeed'].dropna()
    
    if not valid_speeds.empty:
        q1, q2 = np.percentile(valid_speeds, [33, 66])
    else:
        q1, q2 = 0, 0  # fallback for edge cases
        
    def classify_by_speed(speed):
        if pd.isna(speed):
            return None
        elif speed < q1:
            return 'Slow'
        elif speed < q2:
            return 'Medium'
        else:
            return 'Fast'
    
    df_results['Class'] = df_results['MinSpeed'].apply(classify_by_speed)
    
    return df_results

# %%
# Example usage - commented out for module import
# classified_china = classify_corners_by_speed(china25)
# classified_china

# %%
# import matplotlib.pyplot as plt
# 
# plt.scatter(classified_china["ApexDistance"], classified_china["MinSpeed"], c=classified_china["Class"].map({"Fast": "green", "Medium": "orange", "Slow": "red"}))
# plt.xlabel("Distance Along Track")
# plt.ylabel("Turn Angle (deg)") 
# plt.title("Corner Classification by Angle")
# plt.grid(True)
# plt.show()

# %%
def aggregation_function(telemetry):
    telemetry_sorted = telemetry.sort_values(by=['LapNumber', 'Time']) 
    return telemetry_sorted.groupby('LapNumber').agg(
        AvgSpeed = ('Speed', 'mean'),
        TopSpeed = ('Speed', 'max'),
        AvgGear = ('nGear', 'mean'),
        GearChanges = ('nGear', calculate_lap_gear_changes),
        AvgThrottleIntensity = ('Throttle', 'mean'),
        MaxDeceleration = ('Acceleration(m/s^2)', 'min'),
        MaxAcceleration = ('Acceleration(m/s^2)', 'max'), 
        AvgAcceleration = ('Acceleration(m/s^2)', 'mean'),
        Position = ('Position', 'min')
    ).reset_index()

# %%
def extract_corner_telemetry_sections(
    session: fastf1.core.Session,
    driver: str,
    distance_before: float = 100.0,
    distance_after: float = 100.0,
    selected_corners: list = None,
    corner_selection_method: str = 'default'
) -> dict:
    """
    Extract aggregated telemetry data for sections before and after corners.
    
    This function analyzes telemetry data around corner apexes, separating the data into
    "into_turn" (before apex) and "out_of_turn" (after apex) sections to compare
    acceleration patterns and driving behavior.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        FastF1 session object with loaded telemetry data
    driver : str
        Driver identifier (e.g., 'VER', 'HAM')
    distance_before : float, default=100.0
        Distance in meters before corner apex to include in analysis
    distance_after : float, default=100.0
        Distance in meters after corner apex to include in analysis
    selected_corners : list, optional
        List of corner indices to analyze manually. If None, uses corner_selection_method
    corner_selection_method : str, default='default'
        Method for automatic corner selection:
        - 'default': selects fastest, slowest, and median corners by speed
        - 'all': analyzes all corners
    
    Returns:
    --------
    dict
        Dictionary with corner analysis results:
        {
            'corner_X': {
                'corner_info': {corner classification data},
                'into_turn': {aggregated telemetry metrics},
                'out_of_turn': {aggregated telemetry metrics}
            },
            ...
        }
    """
    
    # Get telemetry data and corner classifications
    telemetry_data = get_all_car_data(session, driver)
    corner_classifications = classify_corners_by_speed(session)
    
    # Select corners to analyze
    if selected_corners is None:
        if corner_selection_method == 'default':
            # Select fastest, slowest, and median corners
            valid_corners = corner_classifications.dropna(subset=['MinSpeed'])
            if len(valid_corners) >= 3:
                sorted_corners = valid_corners.sort_values('MinSpeed')
                selected_indices = [
                    sorted_corners.index[0],  # Slowest
                    sorted_corners.index[len(sorted_corners)//2],  # Median
                    sorted_corners.index[-1]  # Fastest
                ]
            else:
                selected_indices = valid_corners.index.tolist()
        elif corner_selection_method == 'all':
            selected_indices = corner_classifications.index.tolist()
        else:
            raise ValueError("corner_selection_method must be 'default' or 'all'")
    else:
        selected_indices = selected_corners
    
    results = {}
    
    for corner_idx in selected_indices:
        corner_info = corner_classifications.iloc[corner_idx]
        apex_distance = corner_info['ApexDistance']
        
        # Extract telemetry sections for this corner across all laps
        into_turn_data = []
        out_of_turn_data = []
        
        for lap_num in telemetry_data['LapNumber'].unique():
            lap_data = telemetry_data[telemetry_data['LapNumber'] == lap_num].copy()
            
            if lap_data.empty:
                continue
                
            # Define distance ranges
            before_start = apex_distance - distance_before
            before_end = apex_distance
            after_start = apex_distance
            after_end = apex_distance + distance_after
            
            # Extract "into turn" section (before apex)
            into_mask = (lap_data['Distance'] >= before_start) & (lap_data['Distance'] < before_end)
            into_section = lap_data[into_mask].copy()
            
            # Extract "out of turn" section (after apex)  
            out_mask = (lap_data['Distance'] >= after_start) & (lap_data['Distance'] <= after_end)
            out_section = lap_data[out_mask].copy()
            
            if not into_section.empty:
                into_section['Section'] = 'into_turn'
                into_section['CornerIndex'] = corner_idx
                into_turn_data.append(into_section)
                
            if not out_section.empty:
                out_section['Section'] = 'out_of_turn'
                out_section['CornerIndex'] = corner_idx
                out_of_turn_data.append(out_section)
        
        # Combine data from all laps
        if into_turn_data:
            combined_into = pd.concat(into_turn_data, ignore_index=True)
            into_aggregated = aggregate_section_data(combined_into)
        else:
            into_aggregated = None
            
        if out_of_turn_data:
            combined_out = pd.concat(out_of_turn_data, ignore_index=True)
            out_aggregated = aggregate_section_data(combined_out)
        else:
            out_aggregated = None
        
        # Store results
        corner_name = f"corner_{corner_idx}"
        results[corner_name] = {
            'corner_info': {
                'apex_distance': apex_distance,
                'angle': corner_info['Angle'],
                'min_speed': corner_info['MinSpeed'],
                'speed_class': corner_info['Class'],
                'angle_class': corner_info['CornerAngle']
            },
            'into_turn': into_aggregated,
            'out_of_turn': out_aggregated
        }
    
    return results

def aggregate_section_data(section_data: pd.DataFrame) -> dict:
    """
    Aggregate telemetry data for a corner section.
    
    This function applies similar aggregations as the original aggregation_function
    but adapted for corner section analysis rather than lap-based analysis.
    
    Parameters:
    -----------
    section_data : pd.DataFrame
        Telemetry data for a specific corner section
        
    Returns:
    --------
    dict
        Dictionary of aggregated metrics
    """
    
    if section_data.empty:
        return None
    
    # Calculate gear changes across the entire section
    section_gear_changes = calculate_lap_gear_changes(section_data['nGear'])
    
    aggregated = {
        'AvgSpeed': section_data['Speed'].mean(),
        'TopSpeed': section_data['Speed'].max(),
        'MinSpeed': section_data['Speed'].min(),
        'EntrySpeed': section_data['Speed'].iloc[0] if len(section_data) > 0 else np.nan,
        'ExitSpeed': section_data['Speed'].iloc[-1] if len(section_data) > 0 else np.nan,
        'AvgGear': section_data['nGear'].mean(),
        'GearChanges': section_gear_changes,
        'AvgThrottleIntensity': section_data['Throttle'].mean(),
        'MaxThrottle': section_data['Throttle'].max(),
        'MinThrottle': section_data['Throttle'].min(),
        'MaxDeceleration': section_data['Acceleration(m/s^2)'].min(),
        'MaxAcceleration': section_data['Acceleration(m/s^2)'].max(),
        'AvgAcceleration': section_data['Acceleration(m/s^2)'].mean(),
        'DataPoints': len(section_data),
        'DistanceCovered': section_data['Distance'].max() - section_data['Distance'].min(),
        'TimeDuration': (section_data['Time'].max() - section_data['Time'].min()).total_seconds()
    }
    
    return aggregated

# %%
def compare_corner_sections(corner_results: dict, corner_name: str = None) -> pd.DataFrame:
    """
    Create a comparison DataFrame for into_turn vs out_of_turn metrics.
    
    Parameters:
    -----------
    corner_results : dict
        Results from extract_corner_telemetry_sections()
    corner_name : str, optional
        Specific corner to analyze. If None, analyzes all corners.
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with metrics for each corner section
    """
    
    comparison_data = []
    
    corners_to_analyze = [corner_name] if corner_name else corner_results.keys()
    
    for corner in corners_to_analyze:
        if corner not in corner_results:
            continue
            
        corner_data = corner_results[corner]
        corner_info = corner_data['corner_info']
        
        # Add into_turn metrics
        if corner_data['into_turn']:
            into_row = corner_data['into_turn'].copy()
            into_row['Corner'] = corner
            into_row['Section'] = 'into_turn'
            into_row['SpeedClass'] = corner_info['speed_class']
            into_row['AngleClass'] = corner_info['angle_class']
            comparison_data.append(into_row)
        
        # Add out_of_turn metrics
        if corner_data['out_of_turn']:
            out_row = corner_data['out_of_turn'].copy()
            out_row['Corner'] = corner
            out_row['Section'] = 'out_of_turn'
            out_row['SpeedClass'] = corner_info['speed_class']
            out_row['AngleClass'] = corner_info['angle_class']
            comparison_data.append(out_row)
    
    return pd.DataFrame(comparison_data)

# %%
# Example usage - commented out for module import
# # Extract telemetry for default corners (fastest, slowest, median)
# corner_analysis = extract_corner_telemetry_sections(china25, 'VER')
# 
# # Extract with custom distances (50m before, 150m after)
# corner_analysis = extract_corner_telemetry_sections(
#     china25, 'VER', 
#     distance_before=100, 
#     distance_after=100
# )
# 
# # Create comparison DataFrame
# comparison_df = compare_corner_sections(corner_analysis)
# print(comparison_df[['Corner', 'SpeedClass', 'Section', 'AvgSpeed', 'MaxAcceleration', 'EntrySpeed', 'ExitSpeed']])

# %%
# comparison_df

# %% [markdown]
# ## Comparing drivers aggregated metrics 
# 

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
import numpy as np
from typing import List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

def extract_all_drivers_corner_data(
    session: fastf1.core.Session,
    drivers: Optional[List[str]] = None,
    distance_before: float = 100.0,
    distance_after: float = 100.0,
    corner_selection_method: str = 'default'
) -> Dict:
    """
    Extract corner telemetry data for all specified drivers in a session.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        FastF1 session object with loaded telemetry data
    drivers : List[str], optional
        List of driver identifiers. If None, uses all drivers in session
    distance_before : float, default=100.0
        Distance in meters before corner apex
    distance_after : float, default=100.0  
        Distance in meters after corner apex
    corner_selection_method : str, default='default'
        Corner selection method ('default', 'all')
        
    Returns:
    --------
    Dict
        Dictionary with driver data and combined comparison DataFrame
    """
    
    if drivers is None:
        drivers = session.laps['Driver'].unique().tolist()
    
    all_driver_data = {}
    all_comparison_data = []
    
    print(f"Extracting corner data for {len(drivers)} drivers...")
    
    for i, driver in enumerate(drivers):
        try:
            print(f"Processing driver {driver} ({i+1}/{len(drivers)})...")
            
            # Extract corner data for this driver
            driver_corners = extract_corner_telemetry_sections(
                session, driver, distance_before, distance_after, 
                corner_selection_method=corner_selection_method
            )
            
            # Create comparison DataFrame for this driver
            driver_comparison = compare_corner_sections(driver_corners)
            if not driver_comparison.empty:
                driver_comparison['Driver'] = driver
                all_comparison_data.append(driver_comparison)
                
            all_driver_data[driver] = driver_corners
            
        except Exception as e:
            print(f"Warning: Could not process driver {driver}: {e}")
            continue
    
    # Combine all driver comparison data
    if all_comparison_data:
        combined_comparison = pd.concat(all_comparison_data, ignore_index=True)
    else:
        combined_comparison = pd.DataFrame()
    
    return {
        'driver_data': all_driver_data,
        'comparison_df': combined_comparison,
        'drivers': drivers
    }

# %%
def plot_driver_comparison_dashboard(
    multi_driver_data: Dict,
    figsize: tuple = (32, 24),
    drivers_subset: Optional[List[str]] = None
):
    """
    Create a comprehensive dashboard comparing driver corner approaches.
    
    Parameters:
    -----------
    multi_driver_data : Dict
        Data from extract_all_drivers_corner_data()
    figsize : tuple, default=(20, 15)
        Figure size for the dashboard
    drivers_subset : List[str], optional
        Subset of drivers to include in comparison
    """
    
    comparison_df = multi_driver_data['comparison_df']
    
    if comparison_df.empty:
        print("No data available for visualization")
        return
    
    # Filter drivers if subset specified
    if drivers_subset:
        comparison_df = comparison_df[comparison_df['Driver'].isin(drivers_subset)]
    
    # Set up the dashboard
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)
    
    # 1. Speed Comparison (Entry vs Exit)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_speed_comparison(comparison_df, ax=ax1)
    
    # 2. Acceleration Patterns
    ax2 = fig.add_subplot(gs[0, 1])
    plot_acceleration_patterns(comparison_df, ax=ax2)
    
    # 3. Throttle Strategies
    ax3 = fig.add_subplot(gs[0, 2])
    plot_throttle_strategies(comparison_df, ax=ax3)
    
    # 4. Performance Radar Chart
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    plot_corner_performance_radar(comparison_df, ax=ax4)
    
    # 5. Corner Type Performance Heatmap
    ax5 = fig.add_subplot(gs[1, 1:])
    plot_performance_heatmap(comparison_df, ax=ax5)
    
    # 6. Speed Profile by Corner Class
    ax6 = fig.add_subplot(gs[2, :])
    plot_speed_by_corner_class(comparison_df, ax=ax6)
    
    plt.suptitle('Driver Corner Approach Comparison Dashboard', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig 

# %%
def plot_driver_comparison_dashboard(
    multi_driver_data: Dict,
    figsize: tuple = (32, 32),
    drivers_subset: Optional[List[str]] = None
):
    """
    Create a comprehensive dashboard comparing driver corner approaches with separate plots by corner type.
    
    Parameters:
    -----------
    multi_driver_data : Dict
        Data from extract_all_drivers_corner_data()
    figsize : tuple, default=(32, 32)
        Figure size for the dashboard
    drivers_subset : List[str], optional
        Subset of drivers to include in comparison
    """
    
    comparison_df = multi_driver_data['comparison_df']
    
    if comparison_df.empty:
        print("No data available for visualization")
        return
    
    # Filter drivers if subset specified
    if drivers_subset:
        comparison_df = comparison_df[comparison_df['Driver'].isin(drivers_subset)]
    
    # Set up the dashboard with updated layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1])
    
    # Row 1: Speed Comparison, Acceleration Patterns, Throttle Strategies
    ax1 = fig.add_subplot(gs[0, 0])
    plot_speed_comparison(comparison_df, ax=ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_acceleration_patterns(comparison_df, ax=ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_throttle_strategies(comparison_df, ax=ax3)
    
    # Row 2: Performance Heatmaps by Corner Type (3 separate heatmaps)
    heatmap_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    plot_performance_heatmaps_by_corner_type(comparison_df, axes=heatmap_axes)
    
    # Row 3: Radar Charts by Corner Type (3 separate radar charts)
    radar_axes = [fig.add_subplot(gs[2, i], projection='polar') for i in range(3)]
    plot_corner_performance_radars_by_type(comparison_df, axes=radar_axes)
    
    # Row 4: Speed Distribution by Corner Class (3 separate plots)
    speed_dist_axes = [fig.add_subplot(gs[3, i]) for i in range(3)]
    plot_speed_by_corner_class_separate(comparison_df, axes=speed_dist_axes)
    
    # Row 5: Additional analysis space (you can add more plots here if needed)
    # ax_extra = fig.add_subplot(gs[4, :])
    # ax_extra.text(0.5, 0.5, 'Additional Analysis Space\n(Add more visualizations here as needed)', 
    #               ha='center', va='center', transform=ax_extra.transAxes, fontsize=16)
    # ax_extra.set_xlim(0, 1)
    # ax_extra.set_ylim(0, 1)
    # ax_extra.axis('off')
    
    plt.suptitle('Driver Corner Approach Comparison Dashboard - Separated by Corner Type', 
                 fontsize=24, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig
   
def plot_speed_comparison(comparison_df: pd.DataFrame, ax=None):
    """Plot entry vs exit speed comparison across drivers."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    drivers = comparison_df['Driver'].unique()
    sections = ['into_turn', 'out_of_turn']
    
    entry_speeds = []
    exit_speeds = []
    driver_labels = []
    
    for driver in drivers:
        driver_data = comparison_df[comparison_df['Driver'] == driver]
        
        # Get average entry and exit speeds
        into_data = driver_data[driver_data['Section'] == 'into_turn']
        out_data = driver_data[driver_data['Section'] == 'out_of_turn']
        
        if not into_data.empty and not out_data.empty:
            entry_speeds.append(into_data['EntrySpeed'].mean())
            exit_speeds.append(out_data['ExitSpeed'].mean())
            driver_labels.append(driver)
    
    x = np.arange(len(driver_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, entry_speeds, width, label='Average Entry Speed', alpha=0.8)
    bars2 = ax.bar(x + width/2, exit_speeds, width, label='Average Exit Speed', alpha=0.8)
    
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Speed (km/h)', fontsize=12)
    ax.set_title('Entry vs Exit Speed Comparison', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(driver_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

def plot_acceleration_patterns(comparison_df: pd.DataFrame, ax=None):
    """Plot acceleration patterns (max acceleration and deceleration) across drivers."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    drivers = comparison_df['Driver'].unique()
    
    max_accel_into = []
    max_accel_out = []
    max_decel_into = []
    driver_labels = []
    
    for driver in drivers:
        driver_data = comparison_df[comparison_df['Driver'] == driver]
        
        into_data = driver_data[driver_data['Section'] == 'into_turn']
        out_data = driver_data[driver_data['Section'] == 'out_of_turn']
        
        if not into_data.empty and not out_data.empty:
            max_accel_into.append(into_data['MaxAcceleration'].mean())
            max_accel_out.append(out_data['MaxAcceleration'].mean())
            max_decel_into.append(abs(into_data['MaxDeceleration'].mean()))
            driver_labels.append(driver)
    
    x = np.arange(len(driver_labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, max_decel_into, width, label='Max Deceleration (Into Turn)', 
                   color='red', alpha=0.7)
    bars2 = ax.bar(x, max_accel_into, width, label='Max Acceleration (Into Turn)', 
                   color='orange', alpha=0.7)
    bars3 = ax.bar(x + width, max_accel_out, width, label='Max Acceleration (Out of Turn)', 
                   color='green', alpha=0.7)
    
    ax.set_xlabel('Driver', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title('Acceleration Patterns by Driver', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(driver_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_throttle_strategies(comparison_df: pd.DataFrame, ax=None):
    """Plot throttle application strategies across drivers."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot of throttle usage
    into_data = comparison_df[comparison_df['Section'] == 'into_turn']
    out_data = comparison_df[comparison_df['Section'] == 'out_of_turn']
    
    drivers = comparison_df['Driver'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(drivers)))
    
    for i, driver in enumerate(drivers):
        driver_into = into_data[into_data['Driver'] == driver]
        driver_out = out_data[out_data['Driver'] == driver]
        
        if not driver_into.empty and not driver_out.empty:
            avg_throttle_into = driver_into['AvgThrottleIntensity'].mean()
            avg_throttle_out = driver_out['AvgThrottleIntensity'].mean()
            
            ax.scatter(avg_throttle_into, avg_throttle_out, 
                      color=colors[i], label=driver, s=100, alpha=0.7)
    
    # Add diagonal line for reference
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Throttle')
    
    ax.set_xlabel('Average Throttle Into Turn (%)', fontsize=12)
    ax.set_ylabel('Average Throttle Out of Turn (%)', fontsize=12)
    ax.set_title('Throttle Application Strategies', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_corner_performance_radar(comparison_df: pd.DataFrame, ax=None):
    """Create radar chart comparing driver performance across multiple metrics."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Select key metrics for radar chart
    metrics = ['AvgSpeed', 'MaxAcceleration', 'AvgThrottleIntensity', 'MinSpeed']
    metric_labels = ['Avg Speed', 'Max Acceleration', 'Avg Throttle', 'Min Speed']
    
    drivers = comparison_df['Driver'].unique()[:5]  # Limit to 5 drivers for clarity
    colors = plt.cm.Set1(np.linspace(0, 1, len(drivers)))
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, driver in enumerate(drivers):
        driver_data = comparison_df[comparison_df['Driver'] == driver]
        
        if driver_data.empty:
            continue
        
        values = []
        for metric in metrics:
            # Normalize values (using percentile rank)
            all_values = comparison_df[metric].dropna()
            if not all_values.empty:
                driver_value = driver_data[metric].mean()
                percentile = (all_values <= driver_value).sum() / len(all_values) * 100
                values.append(percentile)
            else:
                values.append(0)
        
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=driver, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentile Rank', labelpad=20, fontsize=12)
    ax.set_title('Driver Performance Radar Chart', pad=20, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

def plot_corner_performance_radars_by_type(comparison_df: pd.DataFrame, axes=None):
    """Create 3 separate radar charts, one for each corner type."""
    
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    # Select key metrics for radar chart
    metrics = ['AvgSpeed', 'MaxAcceleration', 'AvgThrottleIntensity', 'EntrySpeed', 'ExitSpeed']
    metric_labels = ['Avg Speed', 'Max Acceleration', 'Avg Throttle', 'Entry Speed', 'Exit Speed']
    
    corner_types = ['Fast', 'Medium', 'Slow']
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for i, corner_type in enumerate(corner_types):
        # Filter data for this corner type
        corner_data = comparison_df[comparison_df['SpeedClass'] == corner_type]
        
        if corner_data.empty:
            axes[i].text(0.5, 0.5, f'No data for {corner_type} corners', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{corner_type} Corners', pad=20, fontsize=14)
            continue
        
        drivers = corner_data['Driver'].unique()[:5]  # Limit to 5 drivers for clarity
        colors_radar = plt.cm.Set1(np.linspace(0, 1, len(drivers)))
        
        for j, driver in enumerate(drivers):
            driver_data = corner_data[corner_data['Driver'] == driver]
            
            if driver_data.empty:
                continue
            
            values = []
            for metric in metrics:
                # Normalize values (using percentile rank within this corner type)
                all_values = corner_data[metric].dropna()
                if not all_values.empty and len(all_values) > 1:
                    driver_value = driver_data[metric].mean()
                    percentile = (all_values <= driver_value).sum() / len(all_values) * 100
                    values.append(percentile)
                else:
                    values.append(50)  # neutral value if no data
            
            values += values[:1]  # Complete the circle
            
            axes[i].plot(angles, values, 'o-', linewidth=2, label=driver, color=colors_radar[j])
            axes[i].fill(angles, values, alpha=0.25, color=colors_radar[j])
        
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels(metric_labels, fontsize=10)
        axes[i].set_ylim(0, 100)
        axes[i].set_title(f'{corner_type} Corners - Performance Radar', pad=20, fontsize=14)
        axes[i].tick_params(axis='both', which='major', labelsize=8)
        axes[i].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=8)
        axes[i].grid(True)
    
    return axes

# This is the old heatmap
def plot_performance_heatmap(comparison_df: pd.DataFrame, ax=None):
    """Create heatmap showing driver performance across different corner types."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create pivot table for heatmap
    heatmap_data = comparison_df.groupby(['Driver', 'SpeedClass', 'Section'])['AvgSpeed'].mean().unstack(fill_value=0)
    
    if heatmap_data.empty:
        ax.text(0.5, 0.5, 'No data available for heatmap', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create the heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Average Speed (km/h)'})
    ax.set_title('Average Speed by Driver, Corner Type, and Section', fontsize=14)
    ax.set_xlabel('Section', fontsize=12)
    ax.set_ylabel('Driver - Corner Type', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
# New heatmap for corner type
def plot_performance_heatmaps_by_corner_type(comparison_df: pd.DataFrame, axes=None):
    """Create 3 separate heatmaps showing driver performance for each corner type."""
    
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    corner_types = ['Fast', 'Medium', 'Slow']
    colors = ['Greens', 'Oranges', 'Reds']
    
    for i, (corner_type, cmap) in enumerate(zip(corner_types, colors)):
        # Filter data for this corner type
        corner_data = comparison_df[comparison_df['SpeedClass'] == corner_type]
        
        if corner_data.empty:
            axes[i].text(0.5, 0.5, f'No data for {corner_type} corners', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{corner_type} Corners', fontsize=14)
            continue
        
        # Create pivot table for heatmap
        heatmap_data = corner_data.groupby(['Driver', 'Section'])['AvgSpeed'].mean().unstack(fill_value=0)
        
        if heatmap_data.empty:
            axes[i].text(0.5, 0.5, f'No data for {corner_type} corners', 
                        ha='center', va='center', transform=axes[i].transAxes)
        else:
            # Create the heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap=cmap, ax=axes[i], 
                       cbar_kws={'label': 'Avg Speed (km/h)'})
        
        axes[i].set_title(f'{corner_type} Corners - Average Speed', fontsize=14)
        axes[i].set_xlabel('Section', fontsize=12)
        axes[i].set_ylabel('Driver' if i == 0 else '', fontsize=12)
        axes[i].tick_params(axis='both', which='major', labelsize=10)
    
    return axes

def plot_speed_by_corner_class(comparison_df: pd.DataFrame, ax=None):
    """Plot speed distribution by corner class and section."""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 6))
    
    # Create box plots
    sns.boxplot(data=comparison_df, x='SpeedClass', y='AvgSpeed', hue='Section', ax=ax)
    ax.set_title('Speed Distribution by Corner Class and Section', fontsize=14)
    ax.set_xlabel('Corner Speed Class', fontsize=12)
    ax.set_ylabel('Average Speed (km/h)', fontsize=12)
    ax.legend(title='Section', fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)

def plot_speed_by_corner_class_separate(comparison_df: pd.DataFrame, axes=None):
    """Plot speed distribution with 3 separate plots, one for each corner class."""
    
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    corner_types = ['Fast', 'Medium', 'Slow']
    colors = ['green', 'orange', 'red']
    
    for i, (corner_type, color) in enumerate(zip(corner_types, colors)):
        # Filter data for this corner type
        corner_data = comparison_df[comparison_df['SpeedClass'] == corner_type]
        
        if corner_data.empty:
            axes[i].text(0.5, 0.5, f'No data for {corner_type} corners', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{corner_type} Corners', fontsize=14)
            continue
        
        # Create box plots for this corner type (removed alpha parameter)
        box_plot = sns.boxplot(data=corner_data, x='Section', y='AvgSpeed', ax=axes[i], color=color)
        
        # Set transparency manually on the patches
        for patch in axes[i].artists:
            patch.set_alpha(0.7)
        
        axes[i].set_title(f'{corner_type} Corners - Speed Distribution', fontsize=14)
        axes[i].set_xlabel('Section', fontsize=12)
        axes[i].set_ylabel('Average Speed (km/h)' if i == 0 else '', fontsize=12)
        axes[i].tick_params(axis='both', which='major', labelsize=10)
        axes[i].grid(True, alpha=0.3)
    
    return axes

def create_driver_corner_summary(multi_driver_data: Dict) -> pd.DataFrame:
    """
    Create a summary table of key driver corner metrics.
    
    Parameters:
    -----------
    multi_driver_data : Dict
        Data from extract_all_drivers_corner_data()
        
    Returns:
    --------
    pd.DataFrame
        Summary table with key metrics per driver
    """
    
    comparison_df = multi_driver_data['comparison_df']
    
    if comparison_df.empty:
        return pd.DataFrame()
    
    summary_data = []
    
    for driver in comparison_df['Driver'].unique():
        driver_data = comparison_df[comparison_df['Driver'] == driver]
        
        into_data = driver_data[driver_data['Section'] == 'into_turn']
        out_data = driver_data[driver_data['Section'] == 'out_of_turn']
        
        summary = {
            'Driver': driver,
            'Avg_Entry_Speed': into_data['EntrySpeed'].mean() if not into_data.empty else np.nan,
            'Avg_Exit_Speed': out_data['ExitSpeed'].mean() if not out_data.empty else np.nan,
            'Max_Accel_Into': into_data['MaxAcceleration'].mean() if not into_data.empty else np.nan,
            'Max_Accel_Out': out_data['MaxAcceleration'].mean() if not out_data.empty else np.nan,
            'Max_Decel_Into': into_data['MaxDeceleration'].mean() if not into_data.empty else np.nan,
            'Avg_Throttle_Into': into_data['AvgThrottleIntensity'].mean() if not into_data.empty else np.nan,
            'Avg_Throttle_Out': out_data['AvgThrottleIntensity'].mean() if not out_data.empty else np.nan,
            'Speed_Consistency': driver_data['AvgSpeed'].std() if not driver_data.empty else np.nan
        }
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data)

# %%
def analyze_session_corner_performance(
    session: fastf1.core.Session,
    drivers: Optional[List[str]] = None,
    save_plots: bool = False,
    show_plots: bool = False,
    plot_dir: str = "./plots"
):
    """
    Complete analysis workflow for session corner performance.
    
    Parameters:
    -----------
    session : fastf1.core.Session
        FastF1 session object
    drivers : List[str], optional
        Drivers to analyze
    save_plots : bool, default=False
        Whether to save plots to files
    plot_dir : str, default="./plots"
        Directory to save plots
    """
    
    print("Starting corner performance analysis...")
    
    # Extract data for all drivers
    multi_driver_data = extract_all_drivers_corner_data(session, drivers)
    session_name = session.event.EventName
    season = session.event.EventDate.year
    
    if multi_driver_data['comparison_df'].empty:
        print("No data available for analysis")
        return
    
    # Create comprehensive dashboard
    fig = plot_driver_comparison_dashboard(multi_driver_data)
    
    if save_plots:
        plt.savefig(f"{plot_dir}/corner_comparison_dashboard_{session_name.replace(" ", "")}_{season}.png", dpi=400, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    
    # Create and display summary table
    summary_df = create_driver_corner_summary(multi_driver_data)
    print("\nDriver Corner Performance Summary:")
    print(summary_df.round(2))
    
    return multi_driver_data, summary_df


# %%
def extract_corner_telemetry_season(season: int, include_testing: bool = False,
                                    drivers: Optional[List[str]] = None, save_plots: bool = True, show_plots: bool = False) -> pd.DataFrame:
    """ 
    Extract corner telemetry data for all specified drivers in a season.
    
    Parameters:
    -----------
    season : int
        Season to extract data from
    include_testing : bool, default=False
        Whether to include testing sessions
    drivers : List[str], optional
        List of driver identifiers. If None, uses all drivers in session
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with metrics for each corner section
    """
    fastf1.logger.LoggingManager().set_level(logging.CRITICAL)
    season_calendar = fastf1.events.get_event_schedule(season)
    if not include_testing:
        season_calendar = season_calendar.copy()
        season_calendar = season_calendar[season_calendar['EventFormat'] != 'testing']
    
    for event in season_calendar.itertuples():
        print(f"Processing {event.EventName} {season}")
        race_session = fastf1.get_session(season, event.EventName, 'R')
        race_session.load()
        drivers_in_session = race_session.results.Abbreviation.to_list()
        if drivers is None:
            drivers = drivers_in_session
        
        multi_driver_data, summary = analyze_session_corner_performance(race_session, drivers, save_plots=save_plots, show_plots=show_plots)
        with open(f'./data_outputs/MultDriverData_{event.EventName.replace(" ", "")}_{season}.pkl', 'wb') as f:
            pickle.dump(multi_driver_data, f)

        summary.to_csv(f'./data_outputs/summary_{event.EventName.replace(" ", "")}_{season}.csv')
        
    fastf1.logger.LoggingManager().set_level(logging.INFO)

# %%
# Commented out for module import - uncomment to run analysis
# top10_2022 = ['VER', 'LEC', 'PER', 'RUS', 'SAI', 'HAM', 'NOR', 'OCO', 'ALO', 'BOT']
# extract_corner_telemetry_season(2022, include_testing=False, drivers= top10_2022)

# %%
# top10_2023 = ['VER', 'PER', 'HAM', 'ALO', 'LEC', 'NOR', 'SAI', 'RUS', 'PIA', 'STR']
# extract_corner_telemetry_season(2023, include_testing=False, drivers= top10_2023)

# %%
# Load the saved multi_driver_data dictionary
# with open('multi_driver_data.pkl', 'rb') as f:
#     loaded_data = pickle.load(f)
# 
# # Display the comparison DataFrame
# loaded_data['comparison_df']



