import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

if not os.path.exists('visual'): os.makedirs('visual')

sns.set(style="whitegrid", context="talk")
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# Data Preparation
# ==========================================
# 注意：必须使用和训练时【完全一样】的数据处理逻辑
file_path = 'E:\\Machine learning\\ML-Analysis-of-F1-Fatest-Lap-Circuits\\Data_Merge\\f1_grand_dataset_full.csv'
if not os.path.exists(file_path):
    print("error")
    exit()

df = pd.read_csv(file_path)

# AvgSpeed
if 'sector_length_km_S1' in df.columns and 'LapTime' in df.columns:
    total_len = df['sector_length_km_S1'] + df['sector_length_km_S2'] + df['sector_length_km_S3']
    df['AvgSpeed'] = np.where(df['LapTime'] > 0, total_len / (df['LapTime'] / 3600), 0)

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

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"finish preparing ({len(X_test)} samples)")

# ==========================================
# load model
# ==========================================
model_files = {
    'Linear Regression': 'model/baseline.pkl',       
    'Random Forest': 'model/RandomForest_model.pkl',               
    'XGBoost (Optimized)': 'model/best_xgboost_model.pkl' 
}

results = []

print("\n=== began compare models ===")

for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        print(f"Loading {model_name} from {file_path}...")
        try:
    
            model = joblib.load(file_path)
            
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results.append({
                'Model': model_name,
                'R2': r2,
                'RMSE': rmse
            })
            print(f"   -> [OK] R²: {r2:.4f}")
            
        except Exception as e:
            print(f"   -> [Error] 加载失败: {e}")
    else:
        print(f"cant find {file_path},jump")

# ==========================================
# visual
# ==========================================
if results:
    results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
    
    print("\n=== final ===")
    print(results_df)
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#e84d60' if i == 0 else '#4c72b0' for i in range(len(results_df))]
    
    ax = sns.barplot(data=results_df, x='Model', y='R2', palette=colors)
    
    plt.title('Final Model Comparison (R² Score)', fontsize=16, weight='bold')
    plt.ylim(0.9, 1.0) 
    plt.ylabel('R² Score')
    plt.xlabel('')
    

    for i, v in enumerate(results_df['R2']):
        ax.text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=12, weight='bold')
    
    save_path = 'visual/Viz_Final_Comparison_Loaded.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\finish: {save_path}")
    
else:
    print("\n cant do that, no results collected.")