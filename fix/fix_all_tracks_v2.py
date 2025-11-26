import fastf1
import pandas as pd
import numpy as np
import os

# Configuration
# Only fix Netherlands and Belgium
tracks_to_fix = [
    {'country': 'Belgium', 'year': 2024, 'event': 'Belgium'},
    {'country': 'Netherlands', 'year': 2024, 'event': 'Netherlands'} 
]

# Enable caching
cache_dir = 'fastf1_cache' 
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def calculate_track_features(track_info):
    country = track_info['country']
    year = track_info['year']
    event = track_info['event']
    
    print(f"\n[Processing] Recalculating {country} (relaxing slow corner criteria)...")
    
    try:
        # Load qualifying session and fastest lap telemetry
        session = fastf1.get_session(year, event, 'Q')
        session.load()
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
        
        # Get official sector times (seconds)
        s1_time = lap['Sector1Time'].total_seconds()
        s2_time = lap['Sector2Time'].total_seconds()
        total_time = lap['LapTime'].total_seconds()
        
        # Map sector end times to distances using telemetry
        s1_end_row = tel.iloc[(tel['Time'].dt.total_seconds() - s1_time).abs().argsort()[:1]]
        s1_dist = s1_end_row['Distance'].values[0]
        
        s2_end_time = s1_time + s2_time
        s2_end_row = tel.iloc[(tel['Time'].dt.total_seconds() - s2_end_time).abs().argsort()[:1]]
        s2_dist = s2_end_row['Distance'].values[0]
        
        total_dist = tel['Distance'].max()
        
        # Calculate sector lengths (km)
        len_s1 = s1_dist / 1000
        len_s2 = (s2_dist - s1_dist) / 1000
        len_s3 = (total_dist - s2_dist) / 1000
        
        # [Key Modification] Relax slow corner definition
        # Previous <100km/h was too strict for high-speed tracks like Netherlands
        # Changed to <130km/h to ensure valid data
        tel['is_slow_corner'] = (tel['Speed'] < 130) 
        
        # Keep straight definition unchanged (throttle >95% and speed >150km/h)
        tel['is_straight'] = (tel['Throttle'] > 95) & (tel['Speed'] > 150)

        # Segment telemetry by sectors
        tel_s1 = tel[tel['Distance'] <= s1_dist]
        tel_s2 = tel[(tel['Distance'] > s1_dist) & (tel['Distance'] <= s2_dist)]
        tel_s3 = tel[tel['Distance'] > s2_dist]
        
        # Calculate ratio of target feature in segment
        def calc_ratio(segment_tel, col_name):
            if len(segment_tel) == 0: return 0.0
            return round((segment_tel[col_name].sum() / len(segment_tel)) * 100, 2)

        # Assemble final dataset
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
            'sector_time_seconds': [s1_time, s2_time, round(total_time - s1_time - s2_time, 3)],
            'sector_time_ratio': [
                round((s1_time/total_time)*100, 2),
                round((s2_time/total_time)*100, 2),
                round((1 - (s1_time+s2_time)/total_time)*100, 2)
            ]
        }
        
        df_new = pd.DataFrame(data)
        
        # Save to file
        output_path = f"{country}/Circuit/{country.capitalize()}_sector_features.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_new.to_csv(output_path, index=False)
        
        print(f"[Success] {country} slow corner data fixed (Threshold < 130km/h)")
        print(df_new[['sector_slow_corner_ratio']])

    except Exception as e:
        print(f"[Error] {country}: {e}")

if __name__ == "__main__":
    for track in tracks_to_fix:
        calculate_track_features(track)