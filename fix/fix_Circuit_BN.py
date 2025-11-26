import fastf1
import pandas as pd
import numpy as np
import os

# =================Configuration Area=================
# Hungary is already fixed; here we mainly fix Belgium and Netherlands
tracks_to_fix = [
    {'country': 'Belgium', 'year': 2024, 'event': 'Belgium'},       # Spa-Francorchamps
    {'country': 'Netherlands', 'year': 2024, 'event': 'Netherlands'} # Zandvoort
]

# Enable caching
cache_dir = 'fastf1_cache' 
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir) 
# ==========================================

def calculate_track_features(track_info):
    country = track_info['country']
    year = track_info['year']
    event = track_info['event']
    
    print(f"\n[Processing] Fixing {country} ({year})...")
    
    try:
        # 1. Load data (Qualifying session Q)
        session = fastf1.get_session(year, event, 'Q')
        session.load()
        
        # 2. Get the fastest lap
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
        
        # 3. Get official sector times (these are anchor points)
        s1_time = lap['Sector1Time'].total_seconds()
        s2_time = lap['Sector2Time'].total_seconds()
        s3_time = lap['Sector3Time'].total_seconds()
        total_time = lap['LapTime'].total_seconds()
        
        # 4. Core logic: Infer distance from time (Time -> Distance Mapping)
        # Find the telemetry row corresponding to the end time of Sector 1
        s1_end_row = tel.iloc[(tel['Time'].dt.total_seconds() - s1_time).abs().argsort()[:1]]
        s1_dist = s1_end_row['Distance'].values[0]
        
        # Find the telemetry row corresponding to the end time of Sector 2 (time is cumulative)
        s2_end_time = s1_time + s2_time
        s2_end_row = tel.iloc[(tel['Time'].dt.total_seconds() - s2_end_time).abs().argsort()[:1]]
        s2_dist = s2_end_row['Distance'].values[0]
        
        # End of Sector 3 is the finish line
        total_dist = tel['Distance'].max()
        
        # 5. Calculate physical lengths (km)
        len_s1 = s1_dist / 1000
        len_s2 = (s2_dist - s1_dist) / 1000
        len_s3 = (total_dist - s2_dist) / 1000
        
        # 6. Calculate track features (straight/slow corner ratios)
        # Definition: Straight = Throttle > 95% AND Speed > 150; Slow corner = Speed < 100
        tel['is_straight'] = (tel['Throttle'] > 95) & (tel['Speed'] > 150)
        tel['is_slow_corner'] = (tel['Speed'] < 100)

        # Segment telemetry data by sectors
        tel_s1 = tel[tel['Distance'] <= s1_dist]
        tel_s2 = tel[(tel['Distance'] > s1_dist) & (tel['Distance'] <= s2_dist)]
        tel_s3 = tel[tel['Distance'] > s2_dist]
        
        # Auxiliary calculation function
        def calc_ratio(segment_tel, col_name):
            if len(segment_tel) == 0: return 0
            return round((segment_tel[col_name].sum() / len(segment_tel)) * 100, 2)

        # 7. Assemble data (completely unified format)
        data = {
            'circuit_name': [country, country, country],
            'sector_number': [1, 2, 3],
            'sector_length_km': [round(len_s1, 3), round(len_s2, 3), round(len_s3, 3)],
            'sector_straight_ratio': [
                calc_ratio(tel_s1, 'is_straight'),
                calc_ratio(tel_s2, 'is_straight'),
                calc_ratio(tel_s3, 'is_straight')
            ],
            'sector_slow_corner_ratio': [
                calc_ratio(tel_s1, 'is_slow_corner'),
                calc_ratio(tel_s2, 'is_slow_corner'),
                calc_ratio(tel_s3, 'is_slow_corner')
            ],
            'sector_time_seconds': [s1_time, s2_time, s3_time],
            'sector_time_ratio': [
                round((s1_time/total_time)*100, 2),
                round((s2_time/total_time)*100, 2),
                round((s3_time/total_time)*100, 2)
            ]
        }
        
        df_new = pd.DataFrame(data)
        
        # 8. Save and overwrite
        # Path format: Country/Circuit/Country_sector_features.csv
        output_path = f"{country}/Circuit/{country.capitalize()}_sector_features.csv"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df_new.to_csv(output_path, index=False)
        print(f"[Success] {country} data has been fixed and overwritten! Path: {output_path}")
        print(df_new)

    except Exception as e:
        print(f"[Error] Failed to fix {country}: {e}")

if __name__ == "__main__":
    print("=== Starting Universal Fix (For Perfectionists) ===")
    for track in tracks_to_fix:
        calculate_track_features(track)
    print("\n=== All tasks completed! ===")