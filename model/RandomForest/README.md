
# F1 Fastest Lap Prediction - Random Forest

This folder contains the final version of the Random Forest regression model for predicting F1 fastest lap times based on circuit characteristics.

## Main Files

- **Code_RandomForest.py**  
  Main script for data loading, preprocessing, model training, and visualization. Run directly for full analysis and output.

- **diagnostic_panel.png**  
  Diagnostic panel: feature importance, error distribution, prediction accuracy, residual analysis.

- **trend_panel.png**  
  Key feature trends: shows relationships between main circuit features and lap times.

- **partial_dependence_panel.png**  
  Partial dependence plots: reveals how top features affect lap times.

## Data & Features

- Data source: `../../Data_Merge/f1_grand_dataset_full.csv`
- Samples: 60 (F1 circuit data)
- Train/Test split: 80/20
- Features: sector straight ratios, slow corner ratios, global straight ratio, corners (8 total)

## Model & Performance

- Algorithm: RandomForestRegressor (scikit-learn)
- Key parameters: n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42
- Metrics:
  - Train/Test R²: 0.9909 / 0.9723
  - MAE: 1.3~2.2 seconds
  - RMSE: 1.7~2.7 seconds
- Conclusion: Random Forest and Linear Regression perform similarly, indicating dominant linear relationships in the data.

## Visualization Explanation

1. **diagnostic_panel.png**
   - Feature importance bar chart
   - Error distribution and residual analysis
   - Prediction accuracy scatter plot
   - Quick overview of model performance

2. **trend_panel.png**
   - Shows key circuit features (corners, straight ratios, etc.) and their relationship to lap times

3. **partial_dependence_panel.png**
   - Partial dependence plots for top features, illustrating their impact on lap times and model decisions

## How to Run

```bash
python Code_RandomForest.py
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install: `pip install pandas numpy scikit-learn matplotlib seaborn`

## Additional Notes

- All code comments are in English
- Paths are auto-configured for cross-platform use
- Structure is clear and ready for extension or production deployment


## Feature Importance Ranking

1. **Corners** (弯角数量) - Most critical feature
2. Global Straight Ratio
3. sector_straight_ratio_S3
4. sector_straight_ratio_S1
5. sector_slow_corner_ratio_S3
6. sector_slow_corner_ratio_S2
7. sector_straight_ratio_S2
8. sector_slow_corner_ratio_S1

**Key Insight**: Circuit geometry (corners and straight line ratios) is the most important factor for predicting fastest lap times.

## How to Run

### Using Python Script
```bash
python Code_RandomForest.py
```

### Using Jupyter Notebook
```bash
jupyter notebook RandomForest.ipynb
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install with: `pip install pandas numpy scikit-learn matplotlib seaborn`

## Output

The script generates:
1. **randomforest_f1.png** - Feature importance visualization
2. **RandomForest_vs_Baseline.png** - Model comparison visualization
3. Console output with detailed performance metrics and summary report

## Integration

This implementation follows the project structure with:
- Data source from `Data_Merge` folder
- Parallel structure with `XGBOOST` folder for organization
- All paths automatically configured for cross-platform compatibility
- Proper separation of concerns following ML project best practices

## Author Notes

- All code comments are in English for clarity
- Model uses random_state=42 for reproducible results
- Suitable for production use and further optimization
- Can be extended with additional features or algorithms
