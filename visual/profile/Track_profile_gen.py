import fastf1
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib import gridspec

# Cache setup
if not os.path.exists('fastf1_cache'):
    os.makedirs('fastf1_cache')
fastf1.Cache.enable_cache('fastf1_cache')

# Output directory: visual/profile
output_dir = 'visual/profile'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.style.use('dark_background')

# Load dataset
df = pd.read_csv('f1_grand_dataset_full.csv')

# Track configuration
tracks = [
    {'name': 'Belgium', 'event': 'Belgium'},
    {'name': 'Hungary', 'event': 'Hungary'},
    {'name': 'Netherlands', 'event': 'Netherlands'}
]

# Color definitions
C_S1 = '#FF3B3B'
C_S2 = '#4E8BF5'
C_S3 = '#00D2BE'
C_SLOW = '#FFFF00'

print("=== Starting Professional Multi-Panel Track Maps Generation ===")

for track in tracks:
    country = track['name']
    print(f"\nProcessing: {country}...")
    
    try:
        # Get session data
        session = fastf1.get_session(2024, track['event'], 'Q')
        session.load(telemetry=True, weather=False, messages=False)
        lap = session.laps.pick_fastest()
        tel = lap.get_telemetry()
        
        # Get statistical data from CSV (average values)
        track_data = df[df['circuit_name'] == country].mean(numeric_only=True)
        
        # Create figure: 3:1 width ratio (track map : data panel)
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
        
        ax_map = plt.subplot(gs[0])
        ax_text = plt.subplot(gs[1])
        
        # Calculate sector split points
        s1_time = lap['Sector1Time'].total_seconds()
        s2_time = lap['Sector2Time'].total_seconds()
        
        idx_s1_end = tel.index[(tel['Time'].dt.total_seconds() - s1_time).abs().argsort()[:1]][0]
        idx_s2_end = tel.index[(tel['Time'].dt.total_seconds() - (s1_time+s2_time)).abs().argsort()[:1]][0]
        
        # Plot track sectors
        ax_map.plot(tel['X'].iloc[:idx_s1_end], tel['Y'].iloc[:idx_s1_end], 
                    color=C_S1, linewidth=6, alpha=0.7, label='Sector 1 (Fast)')
        ax_map.plot(tel['X'].iloc[idx_s1_end:idx_s2_end], tel['Y'].iloc[idx_s1_end:idx_s2_end], 
                    color=C_S2, linewidth=6, alpha=0.7, label='Sector 2 (Technical)')
        ax_map.plot(tel['X'].iloc[idx_s2_end:], tel['Y'].iloc[idx_s2_end:], 
                    color=C_S3, linewidth=6, alpha=0.7, label='Sector 3 (Fast)')

        # Highlight slow corners (<120km/h)
        slow_points = tel[tel['Speed'] < 120]
        ax_map.scatter(slow_points['X'], slow_points['Y'], 
                       color=C_SLOW, s=30, zorder=5, label='Slow Corner (<120km/h)')

        ax_map.axis('off')
        ax_map.set_aspect('equal')
        ax_map.legend(loc='lower right', fontsize=12)
        ax_map.set_title(f"{country.upper()} CIRCUIT LAYOUT", fontsize=24, weight='bold', color='white')

        # Right panel: Data dashboard
        ax_text.axis('off')
        
        dashboard_title = f"TRACK ANALYTICS\n{'-'*20}"
        
        metrics = [
            (" Avg Speed", f"{track_data.get('AvgSpeed', 0):.1f} km/h"),
            ("", ""),
            (" S1 Straight %", f"{track_data.get('sector_straight_ratio_S1', 0):.1f} %"),
            (" S2 Straight %", f"{track_data.get('sector_straight_ratio_S2', 0):.1f} %"),
            (" S3 Straight %", f"{track_data.get('sector_straight_ratio_S3', 0):.1f} %"),
            ("", ""),
            (" S1 Slow Cnr %", f"{track_data.get('sector_slow_corner_ratio_S1', 0):.1f} %"),
            (" S2 Slow Cnr %", f"{track_data.get('sector_slow_corner_ratio_S2', 0):.1f} %"),
            (" S3 Slow Cnr %", f"{track_data.get('sector_slow_corner_ratio_S3', 0):.1f} %"),
            ("", ""),
            (" Track Length", f"{track_data.get('Track Length (km)', 0):.3f} km"),
            (" Corners", f"{int(track_data.get('Corners', 0))}")
        ]
        
        ax_text.text(0.05, 0.95, dashboard_title, fontsize=22, weight='bold', color='white', transform=ax_text.transAxes)
        
        start_y = 0.85
        gap = 0.06
        
        for i, (label, value) in enumerate(metrics):
            y_pos = start_y - (i * gap)
            ax_text.text(0.05, y_pos, label, fontsize=16, color='#AAAAAA', transform=ax_text.transAxes)
            ax_text.text(0.95, y_pos, value, fontsize=16, color='white', weight='bold', ha='right', transform=ax_text.transAxes)

        # Save figure
        save_path = f"{output_dir}/Pro_Map_{country}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  -> [OK] Professional map generated: {save_path}")
        plt.close()

    except Exception as e:
        print(f"  -> [Error] {country}: {e}")

print(f"\n🎉 All completed! Check the '{output_dir}' folder.")