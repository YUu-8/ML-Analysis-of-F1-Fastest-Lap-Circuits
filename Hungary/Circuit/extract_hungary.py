import fastf1
import pandas as pd
import os

# Create cache directory
cache_dir = "E:/ML project/Hungary/fastf1_cache"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def extract_overall_features(year, circuit_name):
    """Extract overall track characteristics"""
    session = fastf1.get_session(year, circuit_name, "Q")
    session.load()
    
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_car_data().add_distance()
    total_distance = telemetry["Distance"].max()
    
    # Calculate various ratios
    slow_corner_dist = telemetry[telemetry["Speed"] < 160]["Distance"].diff().sum()
    fast_corner_dist = telemetry[(telemetry["Speed"] >= 240) & (telemetry["Speed"] < 280)]["Distance"].diff().sum()
    straight_dist = telemetry[telemetry["Speed"] >= 280]["Distance"].diff().sum()
    
    # Get circuit information - using correct API
    circuit_info = session.get_circuit_info()
    
    # Calculate DRS zones
    try:
        drs_zones = session.event.get('DRSZones', [])
        drs_zone_count = len(drs_zones)
        drs_total_length = sum([zone.get('End', 0) - zone.get('Start', 0) for zone in drs_zones])
    except:
        drs_zone_count = 0
        drs_total_length = 0
    
    return pd.DataFrame([{
        "circuit_name": circuit_name,
        "length_km": round(total_distance / 1000, 3),  # Get directly from telemetry data
        "orientation": 1 if circuit_info.rotation == "anticlockwise" else 0,
        "slow_corner_ratio": round(slow_corner_dist / total_distance * 100, 2) if total_distance > 0 else 0,
        "fast_corner_ratio": round(fast_corner_dist / total_distance * 100, 2) if total_distance > 0 else 0,
        "straight_ratio": round(straight_dist / total_distance * 100, 2) if total_distance > 0 else 0,
        "total_corners": circuit_info.corners["Number"].nunique() if hasattr(circuit_info, 'corners') else 0,
        "drs_zone_count": drs_zone_count,
        "drs_total_length_m": round(drs_total_length, 2),
        "fastest_lap_seconds": round(fastest_lap["LapTime"].total_seconds(), 3)
    }])

def extract_sector_features(year, circuit_name):
    """Extract sector-wise track characteristics"""
    session = fastf1.get_session(year, circuit_name, "Q")
    session.load()
    
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_car_data().add_distance()
    circuit_info = session.get_circuit_info()
    
    # Get sector boundaries
    sector_data = []
    
    # Method 1: Get sector information from circuit_info
    try:
        sectors = circuit_info.sectors
        for _, sector in sectors.iterrows():
            sec_num = sector["Number"]
            sec_start = sector["Distance"]
            
            # Get the start point of the next sector as the end point of current sector
            next_sectors = sectors[sectors["Number"] > sec_num]
            if len(next_sectors) > 0:
                sec_end = next_sectors.iloc[0]["Distance"]
            else:
                sec_end = telemetry["Distance"].max()
            
            # Filter telemetry data for this sector
            sec_telemetry = telemetry[(telemetry["Distance"] >= sec_start) & (telemetry["Distance"] < sec_end)]
            sec_total_dist = sec_end - sec_start
            
            # Calculate straight and slow corner ratios for this sector
            sec_straight_dist = sec_telemetry[sec_telemetry["Speed"] >= 280]["Distance"].diff().sum()
            sec_slow_corner_dist = sec_telemetry[sec_telemetry["Speed"] < 160]["Distance"].diff().sum()
            
            # Get sector time
            sec_time_col = f"Sector{sec_num}Time"
            if sec_time_col in fastest_lap.index and pd.notna(fastest_lap[sec_time_col]):
                sec_time = fastest_lap[sec_time_col].total_seconds()
            else:
                sec_time = 0
            
            sector_data.append({
                "circuit_name": circuit_name,
                "sector_number": int(sec_num),
                "sector_length_km": round(sec_total_dist / 1000, 3),
                "sector_straight_ratio": round(sec_straight_dist / sec_total_dist * 100, 2) if sec_total_dist > 0 else 0,
                "sector_slow_corner_ratio": round(sec_slow_corner_dist / sec_total_dist * 100, 2) if sec_total_dist > 0 else 0,
                "sector_time_seconds": round(sec_time, 3)
            })
    except Exception as e:
        print(f"⚠️  Unable to extract detailed sector information: {e}")
        # Fallback: Extract sector times only
        for i in range(1, 4):
            sec_time_col = f"Sector{i}Time"
            if sec_time_col in fastest_lap.index and pd.notna(fastest_lap[sec_time_col]):
                sec_time = fastest_lap[sec_time_col].total_seconds()
                sector_data.append({
                    "circuit_name": circuit_name,
                    "sector_number": i,
                    "sector_length_km": 0,
                    "sector_straight_ratio": 0,
                    "sector_slow_corner_ratio": 0,
                    "sector_time_seconds": round(sec_time, 3)
                })
    
    return pd.DataFrame(sector_data)

def extract_hungary_features(year=2024):
    """Extract Hungary circuit features"""
    circuit_name = "Hungary"
    
    print(f"🏁 Extracting {year} {circuit_name} circuit data...")
    
    try:
        # Extract overall features
        print("📊 Extracting overall features...")
        overall_df = extract_overall_features(year, circuit_name)
        overall_df.to_csv(f"{circuit_name}_overall_features.csv", index=False, encoding="UTF-8")
        print(f"✅ Overall features saved to {circuit_name}_overall_features.csv")
        
        # Extract sector features
        print("📊 Extracting sector features...")
        sector_df = extract_sector_features(year, circuit_name)
        
        if len(sector_df) > 0:
            total_lap_time = overall_df["fastest_lap_seconds"].iloc[0]
            sector_df["sector_time_ratio"] = round(sector_df["sector_time_seconds"] / total_lap_time * 100, 2)
            sector_df.to_csv(f"{circuit_name}_sector_features.csv", index=False, encoding="UTF-8")
            print(f"✅ Sector features saved to {circuit_name}_sector_features.csv")
        else:
            print("⚠️  Unable to extract sector features")
        
        return overall_df, sector_df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    overall_df, sector_df = extract_hungary_features(year=2024)  # Try 2024 first
    
    if overall_df is not None:
        print("\n" + "="*50)
        print("Overall Features Preview:")
        print("="*50)
        print(overall_df)
        
        if sector_df is not None and len(sector_df) > 0:
            print("\n" + "="*50)
            print("Sector Features Preview:")
            print("="*50)
            print(sector_df)
        
        print("\n✅ Data extraction completed!")