# 🏎️ ML Analysis of F1 Fastest Lap Circuits

> **"Geometry is Destiny"**: Using Machine Learning to decode how F1 track characteristics dictate racing performance.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20%26%20FastF1-green)]()

## 📖 Overview

This project applies Machine Learning to analyze Formula 1 telemetry and track data. Our goal was to determine how specific circuit characteristics (geometry, sector types, straight-line ratios) influence the **Fastest Lap Speed**.

Unlike traditional analyses that focus on driver skill, this project isolates **Track Geometry** as the primary variable, effectively creating a **"Track Characteristic Simulator"** that serves as a performance baseline for race engineers.

---

## 📂 Repository Structure & Guide

This repository contains the full data pipeline. Here is how the code is organized:

### 1. Main Execution
* `Optimization_and_Comparison.py`: **(Entry Point)** The primary script. It loads the final processed data, runs the GridSearch for model optimization (Linear, RF, XGBoost), and generates the final metrics.

### 2. Engineering & Fixes (Critical Modules)
* **`fix/`**: **(Key Technical Module)**
    * This directory contains the specific scripts used to resolve **OS Compatibility issues** (Windows vs. Linux pathing) and initial data alignment errors.
    * It serves as the "sandbox" where we engineered the solutions for cross-platform stability before integrating them into the main pipeline.
* `Data_Merge/`:
    * Contains the logic for the **Hybrid Data Strategy**. It handles the synchronization of Kaggle CSV data with FastF1 API telemetry.

### 3. Data & Artifacts
* `Belgium/`, `Hungary/`, `Netherlands/`: Specific data containers for each Grand Prix.
* `model/`: Stores the trained `.pkl` model files.
* `visual/`: Output directory for generated plots and heatmaps.

---

## 🛠️ Key Technical Implementations

We faced significant engineering challenges in integrating different data standards. Here is how the code addresses them:

### 1. The "Fix" for OS Compatibility
Early iterations of the project failed on Windows environments due to hardcoded path separators (`\` vs `/`).
* **The Solution (Found in `fix/`):** We developed a path-handling routine using `os.path.join()`. This ensures the project is fully **OS-Agnostic** and runs robustly on Windows, macOS, and Linux without manual adjustment.

### 2. The Data Fusion Pipeline
We utilized a **Hybrid Data Strategy** combining:
* **Kaggle Dataset:** Structured sector data.
* **FastF1 API:** Live telemetry streams.

**The Solution:**
* Implemented **Fuzzy Matching algorithms** to align lap data between the static CSVs and the live API stream.
* Used **Linear Interpolation** to fill gaps in telemetry data, creating a unified "Master Dataset".

---

## 🚧 The "Pivot": Obstacles & Evolution

### ❌ Initial Approach: Predicting "Lap Time"
Initially, we attempted to predict the raw **Lap Time**. However, Exploratory Data Analysis (EDA) revealed a logical trap:
* **The Issue:** Our heatmap showed a **0.99 correlation** between *Track Length* and *Lap Time*.
* **The Result:** The model simply learned that "longer tracks take more time." This was statistically accurate but **engineeringly useless**.

### ✅ The Solution: Predicting "Average Speed"
We reformulated our target variable to **Average Speed**.
* This forced the model to **ignore distance** and focus on **physics**.
* It allows us to understand how "Straight Ratio," "Slow Corner Density," and "Sector Complexity" impact the ultimate pace of an F1 car.

---

## 🚀 Utility: The "Performance Calculator"

This project provides real-world value to F1 engineering teams:

1.  **Physics Baseline:** It calculates a theoretical speed limit based on geometry. If a driver is slower than this baseline, they are underperforming.
2.  **Aero Strategy:** Helps engineers decide between **High Downforce** vs. **Low Drag** setups by predicting the track's theoretical average speed profile.

---

## 👥 Contributors

* **Yuchun Wang** - *Project Lead / Data Engineering / Conclusion*
* [Teammate Name] - *Model Development*
* [Teammate Name] - *Visualization*

---
*Created for the Data Science & Machine Learning Project.*
