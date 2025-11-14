# F1 Track Characteristics and Fastest Lap Time Analysis

## 📋 Project Overview

This project quantitatively analyzes the impact of F1 track characteristics on fastest lap times using machine learning techniques. Our goal is to establish clear correlations between track features (corner distribution, straight ratio, DRS configuration, etc.) and lap performance through data-driven analysis.

### 🎯 Key Milestones

| Date | Deliverable |
|------|-------------|
| **November 26** | Project Introduction Slides |
| **December 10** | Complete Analysis Report |
| **December 23** | Code Package + Demo Video |

---
## 👥 Team & Division of Labor

### Data & Feature Engineering Phase

| Team Member | Responsibility |
|-------------|----------------|
| **Yuchun Wang** | Hungary circuit - data extraction & feature processing |
| **Elise Fouillet** | Belgium circuit - data extraction & feature processing |
| **Jinxin Zhou** | Netherlands circuit - data extraction & feature processing |

### Collaborative Phase

All team members will jointly complete:
- Data merging & validation
- Model training & optimization
- Results analysis & visualization
- Report & presentation preparation
```

## 🔧 1. Core Description: Data Source & Tool Dependencies

### 1.1 Data Source

All track characteristics and fastest lap time data are extracted via the **`fastF1`** Python library, including:
- Track geometric parameters
- Telemetry information
- Sector times for specified years

> **Important**: Use the same year for both track characteristics and lap time data (e.g., 2025 data for 2025 analysis). Minor year differences (adjacent years) generally don't affect overall trend analysis.

### 1.2 fastF1 Library Usage

#### Installation

```bash
# Mandatory libraries (version locked for compatibility)
pip install --upgrade fastf1>=2.3.0 pandas>=2.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0
```

#### Core Usage Rules

- **Data Loading**: Use `fastf1.get_session(year, circuit_name, "Q")` where `"Q"` = Qualifying (cleaner data, no traffic interference)
- **Cache Configuration**: Enable caching to speed up subsequent runs (first run downloads ~50-100MB per track):
  ```python
  fastf1.Cache.enable_cache("./fastf1_cache")
  ```

#### Key APIs

| Function | Purpose |
|----------|---------|
| `session.load()` | Load complete session data (mandatory) |
| `session.circuit_length` | Track length in meters |
| `session.circuit_orientation` | Track direction (clockwise/anticlockwise) |
| `session.get_circuit_info().sectors` | Sector boundary information |
| `lap.get_car_data().add_distance()` | Per-meter telemetry data |

---

## 📊 2. Data Layer: Unified Track Characteristic Extraction

### 2.1 Overall Track Characteristics (10 Fields)

| Field Name | Definition & Calculation | Unit | Source |
|------------|-------------------------|------|--------|
| `circuit_name` | English name (e.g., Hungary, Belgium) | - | Session metadata |
| `length_km` | Lap length (`circuit_length / 1000`) | km | `session.circuit_length` |
| `orientation` | Track direction (1=anticlockwise, 0=clockwise) | Boolean | `session.circuit_orientation` |
| `slow_corner_ratio` | % of sections with speed < 160 km/h | % | Telemetry calculation |
| `fast_corner_ratio` | % of sections with speed ≥ 240 km/h | % | Telemetry calculation |
| `straight_ratio` | % of sections with speed ≥ 280 km/h | % | Telemetry calculation |
| `total_corners` | Total number of corners | Count | `corners["Number"].nunique()` |
| `drs_zone_count` | Number of DRS zones | Count | `len(session.drs_zones)` |
| `drs_total_length_m` | Total DRS zone length | m | Sum of zone lengths |
| `fastest_lap_seconds` | Fastest qualifying lap time | s | Qualifying data (3 decimals) |

### 2.2 Sector-Wise Track Characteristics (6 Fields)

Extracted for Sector 1, 2, and 3 individually:

| Field Name | Definition & Calculation | Unit | Source |
|------------|-------------------------|------|--------|
| `circuit_name` | Circuit identifier | - | Same as overall |
| `sector_number` | Sector ID (1/2/3) | Integer | `sectors["Number"]` |
| `sector_length_km` | Sector length (end - start) | km | Sector boundaries |
| `sector_straight_ratio` | % of sector with speed ≥ 280 km/h | % | Filtered telemetry |
| `sector_slow_corner_ratio` | % of sector with speed < 160 km/h | % | Filtered telemetry |
| `sector_time_seconds` | Fastest sector time | s | Sector timing data (3 decimals) |

### 2.3 Standardized Extraction Code

```python
import fastf1
import pandas as pd

# Enable caching (optional, speeds up repeated runs)
fastf1.Cache.enable_cache("./fastf1_cache")

def extract_overall_features(year, circuit_name):
    """Extract overall track characteristics for the specified year"""
    session = fastf1.get_session(year, circuit_name, "Q")
    session.load()
    
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_car_data().add_distance()
    total_distance = telemetry["Distance"].max()
    
    # Calculate corner/straight ratios
    slow_corner_dist = telemetry[telemetry["Speed"] < 160]["Distance"].diff().sum()
    fast_corner_dist = telemetry[telemetry["Speed"] >= 240]["Distance"].diff().sum()
    straight_dist = telemetry[telemetry["Speed"] >= 280]["Distance"].diff().sum()
    
    return pd.DataFrame([{
        "circuit_name": circuit_name,
        "length_km": round(session.circuit_length / 1000, 3),
        "orientation": 1 if session.circuit_orientation == "anticlockwise" else 0,
        "slow_corner_ratio": round(slow_corner_dist / total_distance * 100, 2),
        "fast_corner_ratio": round(fast_corner_dist / total_distance * 100, 2),
        "straight_ratio": round(straight_dist / total_distance * 100, 2),
        "total_corners": session.get_circuit_info().corners["Number"].nunique(),
        "drs_zone_count": len(session.drs_zones),
        "drs_total_length_m": sum(zone["end"] - zone["start"] for zone in session.drs_zones),
        "fastest_lap_seconds": round(fastest_lap["LapTime"].total_seconds(), 3)
    }])

def extract_sector_features(year, circuit_name):
    """Extract sector-wise track characteristics for the specified year"""
    session = fastf1.get_session(year, circuit_name, "Q")
    session.load()
    fastest_lap = session.laps.pick_fastest()
    telemetry = fastest_lap.get_car_data().add_distance()
    circuit_info = session.get_circuit_info()
    
    sector_bounds = {
        sec["Number"]: (sec["Start"], sec["End"]) 
        for sec in circuit_info.sectors
    }
    
    sector_features = []
    for sec_num, (start, end) in sector_bounds.items():
        sec_telemetry = telemetry[(telemetry["Distance"] >= start) & (telemetry["Distance"] <= end)]
        sec_total_dist = end - start
        
        sec_straight_dist = sec_telemetry[sec_telemetry["Speed"] >= 280]["Distance"].diff().sum()
        sec_slow_corner_dist = sec_telemetry[sec_telemetry["Speed"] < 160]["Distance"].diff().sum()
        
        sec_time_col = f"Sector{sec_num}Time"
        sec_time = fastest_lap[sec_time_col].total_seconds() if not pd.isna(fastest_lap[sec_time_col]) else 0
        
        sector_features.append({
            "circuit_name": circuit_name,
            "sector_number": sec_num,
            "sector_length_km": round(sec_total_dist / 1000, 3),
            "sector_straight_ratio": round(sec_straight_dist / sec_total_dist * 100, 2),
            "sector_slow_corner_ratio": round(sec_slow_corner_dist / sec_total_dist * 100, 2),
            "sector_time_seconds": round(sec_time, 3)
        })
    return pd.DataFrame(sector_features)

# Example: Extract data
if __name__ == "__main__":
    year = 2025  # Match the year of lap time data
    circuit = "Hungary"  # Replace with target circuit
    
    # Extract overall characteristics
    overall_df = extract_overall_features(year, circuit)
    overall_df.to_csv(f"{circuit}_overall_features.csv", index=False, encoding="UTF-8")
    
    # Extract sector characteristics and calculate derived features
    sector_df = extract_sector_features(year, circuit)
    total_lap_time = overall_df["fastest_lap_seconds"].iloc[0]
    sector_df["sector_time_ratio"] = round(sector_df["sector_time_seconds"] / total_lap_time * 100, 2)
    
    sector_df.to_csv(f"{circuit}_sector_features.csv", index=False, encoding="UTF-8")
    
    print("✅ Data extraction completed!")
    print(f"📄 {circuit}_overall_features.csv")
    print(f"📄 {circuit}_sector_features.csv")
```

---

## 🔬 3. Feature Engineering Specifications

### 3.1 Derived Feature Construction (Mandatory)

**Sector Time Ratio**: Reflects each sector's weight in total lap time

```python
sector_time_ratio = (sector_time_seconds / fastest_lap_seconds) × 100
```

*Precision: 2 decimal places*

### 3.2 Data Output Requirements

#### File Naming Convention
- **Overall**: `{circuit_name}_overall_features.csv` (e.g., `Hungary_overall_features.csv`)
- **Sector**: `{circuit_name}_sector_features.csv` (e.g., `Hungary_sector_features.csv`)

#### Format Requirements
- ✅ UTF-8 encoding
- ✅ Strict field order (as per tables above)
- ✅ Unified precision:
  - Ratios: 2 decimal places
  - Length/Time: 3 decimal places
- ✅ No missing values (document exceptions if unavoidable)
- ✅ Validated `sector_time_ratio` calculations

---

## 📅 4. Project Timeline & Phases

| Phase | Timeline | Core Tasks | Deliverables |
|-------|----------|------------|--------------|
| **Data Extraction** | Now - Nov 25 | Extract all circuit features, calculate derived features, merge & validate data | Individual + merged CSV files |
| **Slides Preparation** | Nov 20 - 25 | Create project introduction covering objectives, data specs, initial progress | **Nov 26: Project Slides** |
| **Modeling & Analysis** | Nov 26 - Dec 3 | Data standardization, train models (Linear Regression + Random Forest), feature importance analysis, visualizations | Modeling code, metrics, charts |
| **Report Writing** | Dec 4 - 9 | Write complete analysis report with methodology, results, discussion, conclusions | **Dec 10: Analysis Report** |
| **Code & Video** | Dec 10 - 22 | Organize commented code with dependencies; create demo video | **Dec 23: Code + Video** |

---

## 📦 5. Deliverables Requirements

### 5.1 November 26: Project Introduction Slides

**Core Content:**
1. Project Background
2. Research Objectives
3. Data Specifications (field definitions + extraction logic)
4. Initial Progress (data extraction examples)
5. Next Steps

**Focus**: Demonstrate project feasibility and data uniformity (modeling results not required yet)

### 5.2 December 10: Analysis Report

**Structure:**
1. Introduction
2. Research Methodology
3. Data Overview
4. Experimental Results
5. Analysis & Discussion
6. Conclusion & Outlook

**Requirements**: 
- Clear logical flow
- Data-supported conclusions
- Key findings (e.g., "magnitude of specific characteristic's impact on lap time")

### 5.3 December 23: Code Package + Demo Video

**Code Requirements:**
- Complete pipeline: data extraction → feature engineering → modeling
- `requirements.txt` with library versions
- Comprehensive code comments
- Organized file structure

**Video Requirements:**
- Duration: 3-5 minutes
- Content: Operation demonstration + key conclusions
- Format: MP4

---

## 🤝 6. Collaboration Guidelines

### Repository Management
- **Shared Repository**: [Insert project repository URL]
- **File Naming**: Unified conventions to prevent overwrites
- **Status Tracking**: Mark completion status after submissions

### Communication
- Regular progress updates
- Immediate issue reporting (data extraction/modeling challenges)
- Ensure milestone alignment

### Version Control
- Clear commit messages
- Branch strategy for feature development
- Code review before merging

---

## 📌 Quick Reference

### Speed Classification Thresholds
- **Slow Corners**: < 160 km/h
- **Fast Corners**: ≥ 240 km/h
- **Straights**: ≥ 280 km/h

### Data Precision Standards
| Metric Type | Precision |
|-------------|-----------|
| Ratios (%) | 2 decimals |
| Length (km/m) | 3 decimals |
| Time (s) | 3 decimals |

### Required Libraries
```
fastf1 >= 2.3.0
pandas >= 2.0.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
```

---

**Last Updated**: 14/11/2025
**Project Type**: F1 Data Analysis & Machine Learning  

