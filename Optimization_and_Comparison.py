import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import os

# ==========================================
# 0. 设置
# ==========================================
OUTPUT_FOLDER = 'visual'
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
sns.set(style="whitegrid", context="talk")

# 读取数据
df = pd.read_csv('E:\Machine learning\ML-Analysis-of-F1-Fatest-Lap-Circuits\Data_Merge\f1_grand_dataset_full.csv')

# 现场计算 AvgSpeed
if 'sector_length_km_S1' in df.columns and 'LapTime' in df.columns:
    total_len = df['sector_length_km_S1'] + df['sector_length_km_S2'] + df['sector_length_km_S3']
    df['AvgSpeed'] = np.where(df['LapTime']>0, total_len / (df['LapTime']/3600), 0)
    
# ==========================================
# 1. 准备数据
# ==========================================
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

# 用于存储结果的列表
results = []

print("=== 🏁 开始模型大比拼 (Model Comparison) ===")

# ==========================================
# 模型 1: Linear Regression (Baseline)
# ==========================================
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append({
    'Model': 'Linear Baseline',
    'R2': r2_score(y_test, y_pred_lr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr))
})
print(f"1. Linear Regression Done. (R2: {results[-1]['R2']:.4f})")

# ==========================================
# 模型 2: Random Forest (Default)
# ==========================================
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append({
    'Model': 'Random Forest',
    'R2': r2_score(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf))
})
print(f"2. Random Forest Done. (R2: {results[-1]['R2']:.4f})")

# ==========================================
# 模型 3: XGBoost (Default)
# ==========================================
xg = xgb.XGBRegressor(random_state=42)
xg.fit(X_train, y_train)
y_pred_xg = xg.predict(X_test)
results.append({
    'Model': 'XGBoost (Default)',
    'R2': r2_score(y_test, y_pred_xg),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xg))
})
print(f"3. XGBoost Default Done. (R2: {results[-1]['R2']:.4f})")

# ==========================================
# 模型 4: XGBoost + Grid Search (你的高光时刻!)
# ==========================================
print("\n🔍 正在进行网格搜索 (Grid Search)... 这可能需要一点时间...")

# 定义要尝试的参数网格
param_grid = {
    'n_estimators': [50, 100, 200],     # 多少棵树
    'learning_rate': [0.01, 0.1, 0.2],  # 学习率
    'max_depth': [3, 5, 7],             # 树的深度
    'subsample': [0.8, 1.0]             # 采样比例
}

# 启动搜索
grid_search = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=3, # 3折交叉验证
                           n_jobs=-1, #以此电脑全力跑
                           verbose=0)

grid_search.fit(X_train, y_train)

# 获取最佳模型
best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)

results.append({
    'Model': 'XGBoost (Tuned)',
    'R2': r2_score(y_test, y_pred_best),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_best))
})

print(f"4. ✅ Grid Search 完成!")
print(f"   最佳参数: {grid_search.best_params_}")
print(f"   最佳 R2: {results[-1]['R2']:.4f}")

# ==========================================
# 5. 可视化对比 (生成最终结论图)
# ==========================================
results_df = pd.DataFrame(results)

# 画对比图
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=results_df, x='Model', y='R2', palette='magma')
plt.title('Final Model Comparison: R² Score', fontsize=16, weight='bold')
plt.ylim(0.8, 1.0) # 设置Y轴范围，让差异更明显
plt.ylabel('R² Score (Higher is Better)')
plt.xlabel('')

# 在柱子上标数值
for i, v in enumerate(results_df['R2']):
    ax.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=12, weight='bold')

save_path = f"{OUTPUT_FOLDER}/Viz_6_Model_Comparison.png"
plt.savefig(save_path, dpi=300)
print(f"\n🏆 对比图已保存: {save_path}")

# 打印最终表格供报告使用
print("\n=== 最终成绩单 (Copy to Report) ===")
print(results_df)