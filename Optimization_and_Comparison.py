import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import time

if not os.path.exists('visual'): os.makedirs('visual')
if not os.path.exists('model'): os.makedirs('model')

sns.set(style="whitegrid", context="talk")

file_name = 'E:\\Machine learning\\ML-Analysis-of-F1-Fatest-Lap-Circuits\\Data_Merge\\f1_grand_dataset_full.csv '
file_path = os.path.join(os.getcwd(), file_name)
df = pd.read_csv(file_path)

if 'sector_length_km_S1' in df.columns and 'LapTime' in df.columns:
    total_len = df['sector_length_km_S1'] + df['sector_length_km_S2'] + df['sector_length_km_S3']
    df['AvgSpeed'] = np.where(df['LapTime'] > 0, total_len / (df['LapTime'] / 3600), 0)
else:
    print("erroe: missing columns")
    exit()

features = [
    'sector_straight_ratio_S1', 'sector_straight_ratio_S2', 'sector_straight_ratio_S3',
    'sector_slow_corner_ratio_S1', 'sector_slow_corner_ratio_S2',
    'sector_length_km_S1', 'sector_length_km_S2', 'sector_length_km_S3'
]
valid_cols = [c for c in features if c in df.columns]

data = df[valid_cols + ['AvgSpeed']].dropna()
data = data[data['AvgSpeed'] > 0]

X = data[valid_cols]
y = data['AvgSpeed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"train: {len(X_train)} | test: {len(X_test)}")
print("="*40)

results = []

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
results.append({
    'Model': 'Linear Regression',
    'R2': r2_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
})


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results.append({
    'Model': 'Random Forest',
    'R2': r2_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
})

xgb_def = xgb.XGBRegressor(random_state=42)
xgb_def.fit(X_train, y_train)
y_pred = xgb_def.predict(X_test)
results.append({
    'Model': 'XGBoost (Default)',
    'R2': r2_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
})


start_time = time.time()

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

grid_search = GridSearchCV(estimator=xgb_model, 
                           param_grid=param_grid, 
                           cv=3, 
                           scoring='r2', 
                           n_jobs=-1, 
                           verbose=1)

grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)

end_time = time.time()
print(f" : {end_time - start_time:.1f}s")
print(f"  : {grid_search.best_params_}")

results.append({
    'Model': 'XGBoost (Optimized)',
    'R2': r2_score(y_test, y_pred_best),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_best))
})


model_path = os.path.join('model', 'best_xgboost_model.pkl')
joblib.dump(best_xgb, model_path)
print(f" {model_path}")

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print(results_df)

plt.figure(figsize=(10, 6))

colors = ['#cccccc' if 'Default' in x or 'Linear' in x else '#4c72b0' for x in results_df['Model']]
if 'XGBoost (Optimized)' in results_df['Model'].values:
    colors = ['#e84d60' if x == 'XGBoost (Optimized)' else c for x, c in zip(results_df['Model'], colors)]

ax = sns.barplot(data=results_df, x='Model', y='R2', palette=colors)
plt.title('Final Model Comparison (R² Score)', fontsize=16, weight='bold')
plt.ylim(0.9, 1.0) 
plt.ylabel('R² Score')
plt.xlabel('')
plt.xticks(rotation=15)

for i, v in enumerate(results_df['R2']):
    ax.text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=12, weight='bold')

save_path = os.path.join('visual', 'Viz_6_Model_Comparison.png')
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f" {save_path}")