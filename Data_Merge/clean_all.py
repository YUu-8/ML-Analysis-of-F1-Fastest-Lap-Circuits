import pandas as pd
import os
import numpy as np

# ==========================================
# Configuration
# ==========================================
KEEP_COLS = ['Driver', 'LapTime', 'Sector 1', 'Sector 2', 'Sector 3', 'circuit_name']
TIME_COLS = ['LapTime', 'Sector 1', 'Sector 2', 'Sector 3']

files_to_clean = [
    {'country': 'Belgium', 'path': 'Belgium/Fastest_laps/lap_times_clean.csv'},
    {'country': 'Hungary', 'path': 'Hungary/Fastest_laps/lap_times_clean.csv'},
    {'country': 'Netherlands', 'path': 'Netherlands/Fastest_laps/lap_times_clean.csv'} 
]

print("=== STARTING SAFE DATA CLEANING ===")

for item in files_to_clean:
    path = item['path']
    country = item['country']
    
    if os.path.exists(path):
        print(f"\nProcessing {country} ...")
        try:
            df = pd.read_csv(path)
            
            # 1. Fix Circuit Name
            if 'circuit_name' not in df.columns:
                df['circuit_name'] = country.capitalize()
            
            # 2. Fix Column Names (Remove " Time" suffix)
            df.columns = df.columns.str.replace(' Time', '', regex=False)
            
            # 3. SMART Time Cleaning (The Fix)
            for col in TIME_COLS:
                if col in df.columns:
                    # Step A: Try converting to numeric directly (Preserves Belgium/Hungary)
                    numeric_vals = pd.to_numeric(df[col], errors='coerce')
                    
                    # Step B: Try converting from Timedelta string (Fixes Netherlands)
                    # Only apply to rows that failed Step A (where numeric_vals is NaN)
                    timedelta_vals = pd.to_timedelta(df[col].astype(str), errors='coerce').dt.total_seconds()
                    
                    # Step C: Combine (Priority: Numeric > Timedelta)
                    # This fills the NaNs in numeric_vals with the values calculated from timedelta
                    df[col] = numeric_vals.fillna(timedelta_vals)
                    
                    # Round for cleanliness
                    df[col] = df[col].round(3)

            # 4. Save (Keep only valid columns)
            existing_cols = [c for c in KEEP_COLS if c in df.columns]
            df_clean = df[existing_cols]
            
            # Safety Check: Don't save if we accidentally wiped everything
            if df_clean[TIME_COLS[0]].isnull().all():
                 print(f"  -> [WARNING] skipped saving {country} because data would be empty. Check source file.")
            else:
                df_clean.to_csv(path, index=False)
                print(f"  -> [OK] Cleaned & Saved: {path}")
                print(f"  -> Preview Row 1: {df_clean[TIME_COLS].head(1).values.tolist()}")

        except Exception as e:
            print(f"  -> [Error] processing {country}: {e}")
            
    else:
        print(f"[Error] File not found: {path}")

print("\n" + "="*40)
print("Cleaning Complete. Run 'Merge_Clean.py' now.")
print("="*40)