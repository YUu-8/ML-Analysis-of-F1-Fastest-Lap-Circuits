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

# ==========================================
# 0. 设置与目录
# ==========================================
if not os.path.exists('visual'): os.makedirs('visual')
if not os.path.exists('model'): os.makedirs('model')

sns.set(style="whitegrid", context="talk")
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 准备数据
# ==========================================
file_path = 'E:\\Machine learning\\ML-Analysis-of-F1-Fatest-Lap-Circuits\\Data_Merge\\f1_grand_dataset_full.csv'
if not os.path.exists(file_path):
    print("❌ 错误：找不到 CSV 文件")
    exit()

df = pd.read_csv(file_path)

# 现场计算 AvgSpeed
if 'sector_length_km_S1' in df.columns and 'LapTime' in df.columns:
    total_len = df['sector_length_km_S1'] + df['sector_length_km_S2'] + df['sector_length_km_S3']
    df['AvgSpeed'] = np.where(df['LapTime'] > 0, total_len / (df['LapTime'] / 3600), 0)

# 特征选择
features = [
    'sector_straight_ratio_S1', 'sector_straight_ratio_S2', 'sector_straight_ratio_S3',
    'sector_slow_corner_ratio_S1', 'sector_slow_corner_ratio_S2',
    'sector_length_km_S1', 'sector_length_km_S2', 'sector_length_km_S3'
]
valid_cols = [c for c in features if c in df.columns]

# 数据清洗
data = df[valid_cols + ['AvgSpeed']].dropna()
data = data[data['AvgSpeed'] > 0]

X = data[valid_cols]
y = data['AvgSpeed']

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ 数据准备就绪。训练集: {len(X_train)} | 测试集: {len(X_test)}")
print("="*40)

results = []

# ==========================================
# 模型 1: Linear Regression (基准)
# ==========================================
print("1️⃣  训练 Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
results.append({
    'Model': 'Linear Regression',
    'R2': r2_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
})
# 保存
joblib.dump(lr, 'model/linear_model.pkl')

# ==========================================
# 模型 2: Random Forest (非线性基准)
# ==========================================
print("2️⃣  训练 Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results.append({
    'Model': 'Random Forest',
    'R2': r2_score(y_test, y_pred),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
})
# 保存
joblib.dump(rf, 'model/rf_model.pkl')

# ==========================================
# 模型 3: XGBoost (Grid Search 优化)
# ==========================================
print("\n3️⃣  🚀 启动 XGBoost 网格搜索 (Grid Search)...")
print("    (正在尝试不同参数组合，请稍候...)")
start_time = time.time()

# 定义要搜索的参数网格
param_grid = {
    'n_estimators': [100, 200, 300],     # 树的数量
    'learning_rate': [0.01, 0.05, 0.1],  # 学习率
    'max_depth': [3, 5, 7],              # 树的深度
    'subsample': [0.8, 1.0]              # 样本采样
}

xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# 设置 Grid Search
grid_search = GridSearchCV(estimator=xgb_model, 
                           param_grid=param_grid, 
                           cv=3,                 # 3折交叉验证
                           scoring='r2',         # 以 R2 为优化目标
                           n_jobs=-1, 
                           verbose=1)

# 开始搜索
grid_search.fit(X_train, y_train)

end_time = time.time()
print(f"   ✅ 搜索完成！耗时: {end_time - start_time:.1f}s")

# 获取并保存最佳模型
best_xgb = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"   🏆 最佳参数组合: {best_params}")

# 预测
y_pred_best = best_xgb.predict(X_test)
results.append({
    'Model': 'XGBoost (Optimized)',
    'R2': r2_score(y_test, y_pred_best),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_best))
})

# 保存最佳模型
joblib.dump(best_xgb, 'model/best_xgboost_model.pkl')
print("   💾 最优模型已覆盖保存至 model/best_xgboost_model.pkl")

# ==========================================
# 4. 结果对比与可视化
# ==========================================
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)

print("\n=== 最终对比成绩单 ===")
print(results_df)

# 画图
plt.figure(figsize=(10, 6))
colors = ['#e84d60' if 'XGBoost' in x else '#4c72b0' for x in results_df['Model']]

ax = sns.barplot(data=results_df, x='Model', y='R2', palette=colors)
plt.title('Final Model Comparison (After Optimization)', fontsize=16, weight='bold')
plt.ylim(0.9, 1.0)
plt.ylabel('R² Score')
plt.xlabel('')

for i, v in enumerate(results_df['R2']):
    ax.text(i, v + 0.002, f"{v:.4f}", ha='center', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('visual/Viz_6_Model_Comparison_Optimized.png', dpi=300)
print("\n📊 对比图已生成: visual/Viz_6_Model_Comparison_Optimized.png")