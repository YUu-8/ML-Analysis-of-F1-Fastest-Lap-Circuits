import pandas as pd
import os

# ==========================
# 1. Configure Paths
# ==========================
countries = ['Belgium', 'Hungary', 'Netherlands']

# ==========================
# 2. Data Processing Functions
# ==========================
def process_sector_features(country):
    """Read and pivot sector features (convert S1/S2/S3 to wide format)."""
    # Auto-construct path: Country/Circuit/Country_sector_features.csv
    path = f"{country}/Circuit/{country.capitalize()}_sector_features.csv"
    
    if not os.path.exists(path):
        print(f"[Skip] Missing sector file: {path}")
        return None
    
    df = pd.read_csv(path)
    
    # Unify column names
    df = df.rename(columns={'sector': 'sector_number', 'Circuit': 'circuit_name', 'Track': 'circuit_name'})
    
    # Ensure circuit_name exists
    if 'circuit_name' not in df.columns:
        df['circuit_name'] = country.capitalize()

    # Select columns to pivot (only existing ones)
    target_cols = [c for c in ['sector_length_km', 'sector_straight_ratio', 'sector_slow_corner_ratio', 
                               'sector_time_seconds', 'sector_time_ratio'] if c in df.columns]
    
    # Pivot to wide format
    try:
        df_pivot = df.pivot(index='circuit_name', columns='sector_number', values=target_cols)
        df_pivot.columns = [f"{col[0]}_S{col[1]}" for col in df_pivot.columns]  # Flatten column names
        return df_pivot.reset_index()
    except Exception as e:
        print(f"[Error] Sector pivot failed for {country}: {e}")
        return None

def process_overall_features(country):
    """Read overall track features."""
    path = f"{country}/Circuit/{country.capitalize()}_overall_features.csv"
    
    if not os.path.exists(path):
        print(f"[Skip] Missing overall file: {path}")
        return None
    
    df = pd.read_csv(path)
    # Unify column names
    df = df.rename(columns={'Track': 'circuit_name', 'Circuit': 'circuit_name'})
    if 'circuit_name' not in df.columns:
        df['circuit_name'] = country.capitalize()
    return df

def process_laps(country):
    """Read and clean lap time data."""
    path = f"{country}/Fastest_laps/lap_times_clean.csv"
    
    if not os.path.exists(path):
        print(f"[Skip] Missing lap times file: {path}")
        return None
    
    df = pd.read_csv(path)

    # Rename "Lap Time" to "LapTime"
    df = df.rename(columns={'Lap Time': 'LapTime'})
    
    # Ensure circuit_name exists
    if 'circuit_name' not in df.columns:
        df['circuit_name'] = country.capitalize()
        
    # Time parsing function
    def parse_time_str(t_str):
        """Convert time strings to seconds."""
        try:
            s = str(t_str).strip()
            if s.replace('.', '', 1).isdigit():
                return float(s)
            
            if ':' in s and 'days' not in s:
                parts = s.split(':')
                if len(parts) == 2:  # MM:SS
                    return float(parts[0]) * 60 + float(parts[1])
                elif len(parts) == 3:  # HH:MM:SS
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            
            if 'days' in s:
                return pd.to_timedelta(s).total_seconds()
                
            return 0
        except:
            return 0

    # Parse time columns
    time_cols = ['LapTime', 'Sector 1', 'Sector 2', 'Sector 3']
    for col in time_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_time_str)

    # Auto-complete LapTime using sector sums
    s1 = df['Sector 1'].fillna(0) if 'Sector 1' in df.columns else 0
    s2 = df['Sector 2'].fillna(0) if 'Sector 2' in df.columns else 0
    s3 = df['Sector 3'].fillna(0) if 'Sector 3' in df.columns else 0
    sectors_sum = s1 + s2 + s3
    
    if 'LapTime' not in df.columns:
        df['LapTime'] = 0

    df['LapTime'] = df['LapTime'].fillna(0)
    mask_zero = (df['LapTime'] == 0)
    if mask_zero.any():
        df.loc[mask_zero, 'LapTime'] = sectors_sum[mask_zero] if isinstance(sectors_sum, pd.Series) else sectors_sum

    # Round to 3 decimal places
    df['LapTime'] = df['LapTime'].round(3)

    return df

# ==========================
# 3. Main Logic
# ==========================
all_track_features = []
all_laps = []

print("=== Starting Standard Full Merge ===")

for country in countries:
    print(f"\nProcessing country: {country}...")
    
    # Get sector features (wide format)
    df_sec = process_sector_features(country)
    
    # Get overall features
    df_ovr = process_overall_features(country)
    
    # Merge sector + overall features for the track
    if df_sec is not None and df_ovr is not None:
        df_track = pd.merge(df_sec, df_ovr, on='circuit_name', how='outer')
        all_track_features.append(df_track)
        print(f"  -> [OK] Track features merged")
    elif df_sec is not None:
        all_track_features.append(df_sec)
    elif df_ovr is not None:
        all_track_features.append(df_ovr)
        
    # Get lap times
    df_lap = process_laps(country)
    if df_lap is not None:
        all_laps.append(df_lap)
        print(f"  -> [OK] Lap times loaded ({len(df_lap)} rows)")

# ==========================
# 4. Final Output
# ==========================
if all_track_features and all_laps:
    # Combine all track features
    master_features = pd.concat(all_track_features, ignore_index=True)
    
    # Combine all lap times
    master_laps = pd.concat(all_laps, ignore_index=True)
    
    # Final merge: lap times + track features
    final_df = pd.merge(master_laps, master_features, on='circuit_name', how='left')
    
    # Save result
    output_filename = 'f1_grand_dataset_full.csv'
    final_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*40)
    print("🎉 Merge Successful!")
    print(f"Final Dataset: {output_filename}")
    print(f"Shape: {final_df.shape}")
    print(f"Columns: {list(final_df.columns)}")
    print("="*40)
    
else:
    print("\n[Error] Insufficient data to merge. Check if files exist.")