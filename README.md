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

Below is a guide to navigating the codebase. While the repository contains various experimental folders, here are the core modules:

### 1. The Core Script
* `Optimization_and_Comparison.py`: **(Start Here)** This is the main execution script. It loads the processed data, runs the GridSearch for model optimization (Linear, RF, XGBoost), and generates the performance comparison metrics.

### 2. Data Engineering & Processing
* `Data_Merge/`: **(Key Engineering Module)** Contains the logic for the **Hybrid Data Strategy**.
    * It handles the fusion of Kaggle CSV data (sector times) with FastF1 API data (telemetry).
    * Includes the synchronization logic to align mismatched timestamps.
* `Belgium/`, `Hungary/`, `Netherlands/`: These directories contain the raw and intermediate data specific to each Grand Prix analyzed in this project.

### 3. Artifacts & Outputs
* `model/`: Stores the trained `.pkl` model files (Linear Regression, XGBoost, etc.).
* `visual/`: Contains generated output plots, including the Correlation Heatmap and Model Comparison charts.
* `fastf1_cache/`: Caching directory for the FastF1 API to speed up data loading.

---

## 🛠️ Key Technical Implementations

We faced significant engineering challenges in integrating different data standards. Here is how the code addresses them:

### 🔄 The Data Fusion Pipeline (Located in `Data_Merge/`)
We utilized a **Hybrid Data Strategy** combining:
1.  **Kaggle Dataset:** Structured sector data (Belgium, Hungary, Dutch GPs).
2.  **FastF1 API:** Live telemetry streams (Speed, Throttle, Gear).

**The Challenge:** The data sources had no common key and mismatched timestamps.
**The Solution:**
* Implemented **Fuzzy Matching algorithms** to align lap data between the static CSVs and the live API stream.
* Used **Linear Interpolation** to fill gaps in telemetry data, creating a unified "Master Dataset" for training.

### 💻 OS-Agnostic Path Handling
Early iterations failed on Windows due to path separator issues (`\` vs `/`).
**The Fix:**
* The codebase (`Optimization_and_Comparison.py` and data loaders) was refactored to use `os.path.join()`.
* This ensures the project is fully compatible with **Windows, macOS, and Linux** environments without manual path adjustment.

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

## 📊 Models & Results

We benchmarked three models. The results validated **Occam's Razor**:

| Model | R² Score | RMSE | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | **0.972** | **Low** | **Best Performance** |
| Random Forest | 0.968 | Medium | Good, but overfitting risk |
| XGBoost (Optimized) | 0.971 | Low | High accuracy, complex |

**Conclusion: Linearity Dominates.**
The simple Linear Regression performed on par with complex ensemble models. This proves that with high-quality feature engineering (e.g., precise Straight Ratios), the relationship between Track Geometry and Speed is fundamentally linear.

---

## 👥 Contributors

* **Yuchun Wang** - *Project Lead / Data Engineering / Conclusion*
* [Teammate Name] - *Model Development*
* [Teammate Name] - *Visualization*

---
*Created for the Data Science & Machine Learning Project.*
