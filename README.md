# F1 Track Characteristics & Fastest Lap Analysis 🏎️💨

> **Unlocking Speed: A Data-Driven Approach to Circuit Geometry**

This project quantitatively analyzes the impact of F1 track characteristics (geometry, sector layout) on average speed using machine learning techniques. We aim to establish correlations between features like **Straight Ratio**, **Corner Density** and **Performance**.

---
### 🎯 Key Milestones

| Date | Deliverable |
|------|-------------|
| **November 26** | Project Introduction Slides |
| **December 10** | Complete Analysis Report |
| **December 23** | Code Package + Demo Video |

---

## 📅 Project Timeline & Updates

* **[2025-12-03] 🚀 Current Status:**
    * **Final Report Phase Initiated:** We have started drafting the comprehensive Final Analysis Report.
    * **Format:** The report is being authored in **LaTeX** (via Overleaf) to ensure academic formatting standards (+Bonus points target).
    * **Data Readiness:** The data pipeline is fully operational with the `f1_grand_dataset_full.csv` ready for advanced modeling.

* **[2025-11-26]** Project Introduction Slides submitted.

---

## 🛠️ Project Architecture

### 1. Data Pipeline
We utilize the **FastF1 API** to extract telemetry and timing data.
* **Preprocessing:** Cleaning heterogeneous time formats and imputing missing values.
* **Feature Engineering:**
    * `Straight Ratio`: Percentage of distance spent at full throttle/straight line.
    * `Slow Corner Ratio`: Density of technical corners (<130 km/h).
    * `AvgSpeed`: Target variable derived from Track Length / Lap Time.

### 2. Repository Structure
```bash
├── Belgium/                # Source data for Spa-Francorchamps
├── Hungary/                # Source data for Hungaroring
├── Netherlands/            # Source data for Zandvoort
├── visual/                 # Generated HD plots (Heatmaps, Distributions)
├── fastf1_cache/           # API cache
├── Merge_Clean.py          # [CORE] Data Fusion & Cleaning Script
├── Viz_Final.py            # [CORE] Visualization Generator
├── Baseline.py             # [CORE] Linear Regression Model
├── f1_grand_dataset_full.csv # The Final Processed Dataset
└── README.md
