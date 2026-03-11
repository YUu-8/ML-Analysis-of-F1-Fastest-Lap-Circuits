# 🏎️ ML Analysis of F1 Fastest Lap Circuits

> **"Geometry is Destiny"**: Using Machine Learning to decode how F1 track characteristics dictate racing performance.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20%26%20FastF1-green)]()
## 🚀 Latest Updates by Yuu

**March 2026** — Added MLOps layer on top of the existing ML pipeline:
- Deployed model as a REST API using **FastAPI** (`app.py`)
- Containerized with **Docker** for reproducible deployment
- Automated testing and container build via **GitHub Actions CI/CD** (`.github/workflows/ci.yml`)

## 📖 Overview

This project applies Machine Learning to analyze Formula 1 telemetry and track data. Our goal was to determine how specific circuit characteristics (geometry, sector types, straight-line ratios) influence the **Fastest Lap Speed**.

Unlike traditional analyses that focus on driver skill, this project isolates **Track Geometry** as the primary variable, effectively creating a **"Track Characteristic Simulator"** that serves as a performance baseline for race engineers.

---

## 📂 Repository Structure & Guide

This repository contains the full data pipeline. Here is the technical breakdown of the modules:

### 1. Main Execution
* `Optimization_and_Comparison.py`: **(Entry Point)** The primary script. It loads the final processed data, runs the GridSearch for model optimization (Linear, RF, XGBoost), and generates the final metrics.

### 2. Core Engineering Modules
* **`fix/`**: **(Data Unification Engine)**
    * **Crucial Module:** This directory contains the heavy-lifting code for **unifying our hybrid data sources**.
    * It implements the logic to merge the structured Kaggle dataset with high-frequency FastF1 telemetry.
    * It handles the complex tasks of timestamp synchronization, data cleaning, and resolving OS-path compatibility issues.
* `Data_Merge/`:
    * Helper scripts for organizing file locations and managing the directory structure for the merged datasets.

### 3. Data & Artifacts
* `Belgium/`, `Hungary/`, `Netherlands/`: Specific data containers for each Grand Prix.
* `model/`: Stores the trained `.pkl` model files.
* `visual/`: Output directory for generated plots and heatmaps.

---

## 🛠️ Key Technical Implementations

We faced significant engineering challenges in integrating different data standards. Here is how the code addresses them:

### 1. The Data Fusion Pipeline (Located in `fix/`)
We utilized a **Hybrid Data Strategy**, combining static CSVs (Kaggle) with live API streams (FastF1).
* **The Challenge:** The data sources had no common key and mismatched timestamps.
* **The Solution:** The scripts inside `fix/` implement **Fuzzy Matching algorithms** to align lap data and use **Linear Interpolation** to fill gaps in telemetry. This creates the unified "Master Dataset" used for training.

### 2. Cross-Platform Compatibility
Early iterations failed on Windows due to path separator issues.
* **The Solution:** The code in `fix/` standardizes path handling using `os.path.join()`, making the entire pipeline **OS-Agnostic** (robust on Windows, macOS, and Linux).

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
*Created for the Data Science & Machine Learning Project.*
