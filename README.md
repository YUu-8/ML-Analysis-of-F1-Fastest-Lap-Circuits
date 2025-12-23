# 🏎️ ML Analysis of F1 Fastest Lap Circuits

> **"Geometry is Destiny"**: Using Machine Learning to decode how F1 track characteristics dictate racing performance.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20%26%20FastF1-green)]()

## 📖 Overview

This project applies Machine Learning to analyze Formula 1 telemetry and track data. Our goal was to determine how specific circuit characteristics (geometry, sector types, straight-line ratios) influence the **Fastest Lap Speed**.

Unlike traditional analyses that focus on driver skill, this project isolates **Track Geometry** as the primary variable, effectively creating a **"Track Characteristic Simulator"** that serves as a performance baseline for race engineers.

---

## 📂 Project Structure & Code Modules

The codebase is organized to handle the full data lifecycle, from ingestion of hybrid sources to model deployment.

```text
├── data/                      # Raw and Processed Data
│   ├── raw/                   # Original Kaggle CSVs & FastF1 cache
│   └── processed/             # Merged master datasets
│
├── notebooks/                 # Jupyter Notebooks for EDA
│   └── 01_EDA_Heatmap.ipynb   # Analysis revealing the "Lap Time" correlation trap
│
├── src/                       # Source Code (The Core Logic)
│   ├── __init__.py
│   ├── data_loader.py         # Handles OS-agnostic pathing & loading
│   ├── data_fusion.py         # Merges Kaggle datasets with FastF1 API data
│   ├── feature_eng.py         # Calculus for Straight Ratios & Corner Geometry
│   └── visualization.py       # Generates Track Maps & Correlation Matrices
│
├── models/                    # Saved ML Models (.pkl)
│   ├── linear_model.pkl
│   └── xgboost_model.pkl
│
├── main.py                    # Entry point for the pipeline
├── requirements.txt           # Dependencies
└── README.md
