import fastf1
import pandas as pd
import os

# Enable FastF1 cache
cache_dir = r"C:\Users\lisee\OneDrive\Documents\4ème année\Machine learning\Projet\Belgium"
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def extract_overall_features(year, circuit_name):
    """Extract global track features"""
    session = fastf1.get_session(year, circuit_name, "Q")
    session.load()

    # Get fastest lap and telemetry
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_car_data().add_distance()
    total_distance = telemetry["Distance"].max()

    # Speed-based distance indicators
    slow_corner_dist = telemetry[telemetry["Speed"] < 160]["Distance"].diff().sum()
    fast_corner_dist = telemetry[(telemetry["Speed"] >= 240) & (telemetry["Speed"] < 280)]["Distance"].diff().sum()
    straight_dist = telemetry[telemetry["Speed"] >= 280]["Distance"].diff().sum()

    # Circuit geometry and metadata
    circuit_info = session.get_circuit_info()

    # Removed DRS and orientation fields from output
    return pd.DataFrame([{
        "year": year,
        "circuit_name": circuit_name,
        "length_km": round(total_distance / 1000, 3),
        "slow_corner_ratio": round(slow_corner_dist / total_distance * 100, 2),
        "fast_corner_ratio": round(fast_corner_dist / total_distance * 100, 2),
        "straight_ratio": round(straight_dist / total_distance * 100, 2),
        "total_corners": circuit_info.corners["Number"].nunique() if hasattr(circuit_info, "corners") else 0,
        "fastest_lap_seconds": round(fastest_lap["LapTime"].total_seconds(), 3)
    }])


def extract_sector_features(year, circuit_name):
    """Extract sector times only"""
    session = fastf1.get_session(year, circuit_name, "Q")
    session.load()

    fastest_lap = session.laps.pick_fastest()
    circuit_info = session.get_circuit_info()
    sector_data = []

    # Try to read sector definitions from FIA data
    try:
        sectors = circuit_info.sectors

        for _, sector in sectors.iterrows():
            sec_num = sector["Number"]

            # Read the official sector time
            sec_time_col = f"Sector{sec_num}Time"
            sec_time = fastest_lap[sec_time_col].total_seconds() if sec_time_col in fastest_lap.index else 0

            sector_data.append({
                "year": year,
                "circuit_name": circuit_name,
                "sector_number": int(sec_num),
                "sector_time_seconds": round(sec_time, 3)
            })

    # Fallback if detailed sector geometry is unavailable
    except:
        for i in range(1, 4):
            sec_time_col = f"Sector{i}Time"
            if sec_time_col in fastest_lap.index:
                sec_time = fastest_lap[sec_time_col].total_seconds()
                sector_data.append({
                    "year": year,
                    "circuit_name": circuit_name,
                    "sector_number": i,
                    "sector_time_seconds": round(sec_time, 3)
                })

    return pd.DataFrame(sector_data)


def extract_belgium_features(year=2024):
    """Full extraction workflow for Belgium"""
    circuit_name = "Belgium"

    print(f"Extracting Belgium {year} data...")

    # Extract overall circuit features
    overall_df = extract_overall_features(year, circuit_name)
    overall_df.to_csv("Belgium_overall_features.csv", index=False, encoding="UTF-8")
    print("Saved: Belgium_overall_features.csv")

    # Extract sector-based features
    sector_df = extract_sector_features(year, circuit_name)

    # Compute each sector share of total lap time
    total_lap_time = overall_df["fastest_lap_seconds"].iloc[0]
    sector_df["sector_time_ratio"] = round(sector_df["sector_time_seconds"] / total_lap_time * 100, 2)

    sector_df.to_csv("Belgium_sector_features.csv", index=False, encoding="UTF-8")
    print("Saved: Belgium_sector_features.csv")

    return overall_df, sector_df


# Script entry point
if __name__ == "__main__":
    overall_df, sector_df = extract_belgium_features(year=2024)

    print("\End!")
