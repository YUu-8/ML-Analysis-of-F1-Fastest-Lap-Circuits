import pandas as pd
import os

# Path to Netherlands lap times file
path = 'Netherlands/Fastest_laps/lap_times_clean.csv'

print(f"Fixing Netherlands lap times (by summing Sectors): {path} ...")

if os.path.exists(path):
    df = pd.read_csv(path)
    
    # Ensure sector columns are numeric
    cols = ['Sector 1', 'Sector 2', 'Sector 3']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    # [Core Fix] Calculate LapTime as sum of sectors
    df['LapTime'] = df['Sector 1'] + df['Sector 2'] + df['Sector 3']
    
    # Round to 3 decimal places
    df['LapTime'] = df['LapTime'].round(3)
    
    # Save changes
    df.to_csv(path, index=False)
    
    print("✅ Fix completed!")
    print(f"Preview first 3 rows:\n{df[['LapTime', 'Sector 1', 'Sector 2', 'Sector 3']].head(3)}")
    print("-" * 30)
    print("👉 Please run 'Merge_Clean.py' one final time to complete your dataset!")

else:
    print("[Error] Netherlands file not found. Check the path.")