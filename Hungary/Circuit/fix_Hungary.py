import fastf1
import pandas as pd
import numpy as np
import os

# Enable caching (set cache path according to your directory structure)
# If you haven't explicitly set a cache directory, fastf1 will default to a temporary folder,
# or you can specify your 'fastf1_cache' folder
cache_dir = 'Hungary/fastf1_cache' 
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    
fastf1.Cache.enable_cache(cache_dir) 

def calculate_hungary_features():
    print("Loading Hungary 2024 Qualifying data...")
    # 1. Load data
    session = fastf1.get_session(2024, 'Hungary', 'Q')
    session.load()
    
    # 2. Get the fastest lap of the session
    lap = session.laps.pick_fastest()
    tel = lap.get_telemetry()
    
    # 3. Get sector times (this is official data, absolutely accurate)
    s1_time = lap['Sector1Time'].total_seconds()
    s2_time = lap['Sector2Time'].total_seconds()
    s3_time = lap['Sector3Time'].total_seconds()
    total_time = lap['LapTime'].total_seconds()
    
    print(f"Fastest lap time: {total_time}s (S1: {s1_time}, S2: {s2_time}, S3: {s3_time})")

    # 4. Use [time] to infer [distance] for sector segmentation (this is the core of the fix!)
    # Logic: Find the point in telemetry where the timestamp is closest to the end time of S1;
    # its Distance value is the length of S1
    
    # Distance at the end of S1
    s1_end_row = tel.iloc[(tel['Time'].dt.total_seconds() - s1_time).abs().argsort()[:1]]
    s1_dist = s1_end_row['Distance'].values[0]
    
    # Distance at the end of S2 (Note: Telemetry time is cumulative, so it's S1 + S2)
    s2_end_time = s1_time + s2_time
    s2_end_row = tel.iloc[(tel['Time'].dt.total_seconds() - s2_end_time).abs().argsort()[:1]]
    s2_dist = s2_end_row['Distance'].values[0]
    
    # Distance at the end of S3 (i.e., total length)
    total_dist = tel['Distance'].max()
    
    # Calculate sector lengths (km)
    len_s1 = s1_dist / 1000
    len_s2 = (s2_dist - s1_dist) / 1000
    len_s3 = (total_dist - s2_dist) / 1000
    
    # 5. Calculate straight ratio
    # Define straight: Throttle > 95% and Speed > 150 (simple and effective)
    # Calculation logic: Count of points meeting conditions / Total points in sector (approximate)
    # or Distance of points meeting conditions / Total sector distance (more accurate)
    
    tel['is_straight'] = (tel['Throttle'] > 95) & (tel['Speed'] > 150)
    # Simple definition of slow corner: Speed < 120
    tel['is_slow_corner'] = (tel['Speed'] < 120)

    # Slice telemetry data by sectors
    tel_s1 = tel[tel['Distance'] <= s1_dist]
    tel_s2 = tel[(tel['Distance'] > s1_dist) & (tel['Distance'] <= s2_dist)]
    tel_s3 = tel[tel['Distance'] > s2_dist]
    
    # Helper function: Calculate ratio of specified feature in a telemetry segment (based on count)
    def calc_ratio(segment_tel, col_name):
        if len(segment_tel) == 0: return 0
        # Calculate distance covered by rows meeting condition (using difference)
        # Simple estimation: Count of rows meeting condition / Total rows (sufficiently accurate for 5-minute report)
        return round((segment_tel[col_name].sum() / len(segment_tel)) * 100, 2)

    str_ratio_s1 = calc_ratio(tel_s1, 'is_straight')
    str_ratio_s2 = calc_ratio(tel_s2, 'is_straight')
    str_ratio_s3 = calc_ratio(tel_s3, 'is_straight')
    
    slow_ratio_s1 = calc_ratio(tel_s1, 'is_slow_corner')
    slow_ratio_s2 = calc_ratio(tel_s2, 'is_slow_corner')
    slow_ratio_s3 = calc_ratio(tel_s3, 'is_slow_corner')

    # 6. Construct DataFrame
    data = {
        'circuit_name': ['Hungary', 'Hungary', 'Hungary'],
        'sector_number': [1, 2, 3],
        'sector_length_km': [round(len_s1, 3), round(len_s2, 3), round(len_s3, 3)],
        'sector_straight_ratio': [str_ratio_s1, str_ratio_s2, str_ratio_s3],
        'sector_slow_corner_ratio': [slow_ratio_s1, slow_ratio_s2, slow_ratio_s3],
        'sector_time_seconds': [s1_time, s2_time, s3_time],
        'sector_time_ratio': [
            round((s1_time/total_time)*100, 2),
            round((s2_time/total_time)*100, 2),
            round((s3_time/total_time)*100, 2)
        ]
    }
    
    df_new = pd.DataFrame(data)
    
    # 7. Save and overwrite
    output_path = 'Hungary/Circuit/Hungary_sector_features.csv'
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_new.to_csv(output_path, index=False)
    
    print("-" * 30)
    print("Fix successful! Data preview:")
    print(df_new)
    print(f"File saved to: {output_path}")
    print("-" * 30)

if __name__ == "__main__":
    calculate_hungary_features()