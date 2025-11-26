import fastf1
import pandas as pd
import os
import numpy as np

# 1. Set up caching
cache_dir = 'fastf1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# 2. Define countries to process
tasks = [
    {'country': 'Belgium', 'event': 'Belgium'},
    {'country': 'Hungary', 'event': 'Hungary'},
    {'country': 'Netherlands', 'event': 'Netherlands'}
]

def generate_overall_features(task):
    country = task['country']
    event = task['event']
    print(f"\n[Processing] Generating Overall Features for {country}...")

    try:
        session = fastf1.get_session(2024, event, 'Q')
        session.load()
        
        # --- A. Get track length ---
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
        track_length_km = tel['Distance'].max() / 1000
        
        # --- B. Get number of corners ---
        # FastF1's circuit_info contains corner information
        circuit_info = session.get_circuit_info()
        corners_count = len(circuit_info.corners) if circuit_info else 0
        
        # --- C. Calculate global straight ratio (simple estimation) ---
        tel['is_straight'] = (tel['Throttle'] > 95) & (tel['Speed'] > 150)
        straight_ratio = (tel['is_straight'].sum() / len(tel)) * 100
        
        # --- D. Assemble data ---
        # Column names should match your previous files
        data = {
            'circuit_name': [country],
            'Track Length (km)': [round(track_length_km, 3)],
            'Corners': [corners_count],
            'Global Straight Ratio': [round(straight_ratio, 2)],
            'Event Year': [2024]
        }
        
        df = pd.DataFrame(data)
        
        # --- E. Save file ---
        output_path = f"{country}/Circuit/{country.capitalize()}_overall_features.csv"
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"[Success] {country} Overall features saved to: {output_path}")
        print(df)

    except Exception as e:
        print(f"[Error] Failed to generate for {country}: {e}")

# Execute
if __name__ == "__main__":
    for task in tasks:
        generate_overall_features(task)
    print("\nAll Overall files generated successfully! Please re-run Merge_Clean.py")