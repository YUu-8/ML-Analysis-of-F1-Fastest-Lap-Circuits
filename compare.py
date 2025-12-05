import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

# ==========================================
# 0. 设置
# ==========================================
if not os.path.exists('visual'): os.makedirs('visual')
sns.set(style="whitegrid", context="talk")
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 准备测试数据 (Data Preparation)
# ==========================================
file_path = 'E:\\Machine learning\\ML-Analysis-of-F1-Fatest-Lap-Circuits\\Data_Merge\\f1_grand_dataset_full.csv'
if not os.path.exists(file_path):
    print(f"❌ 错误：找不到文件 {file_path}")
    exit()

df = pd.read_csv(file_path)

# --- 现场计算 AvgSpeed ---
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

# 切分测试集 (random_state=42 保证一致性)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ 测试数据准备完毕: {len(X_test)} 个样本")

# ==========================================
# 2. 加载模型并计算所有指标
# ==========================================
# 请根据实际文件名修改路径
model_files = {
    'Linear Regression': 'model/baseline.pkl',
    'Random Forest': 'model/rf_model.pkl', 
    'XGBoost (Optimized)': 'model/best_xgboost_model.pkl'
}

results = []

print("\n=== 🏁 开始评估模型指标 ===")

for model_name, path in model_files.items():
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            
            # 计算三大指标
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                'Model': model_name,
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae
            })
            print(f"   -> {model_name}: R2={r2:.4f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
            
        except Exception as e:
            print(f"   -> [Error] {model_name}: {e}")
    else:
        print(f"⚠️ 跳过: 找不到 {path}")

# ==========================================
# 3. 生成三合一对比图
# ==========================================
if results:
    results_df = pd.DataFrame(results)
    
    # 创建画布：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 通用颜色逻辑：XGBoost 红色，其他蓝色
    colors = ['#e84d60' if 'XGBoost' in x else '#4c72b0' for x in results_df['Model']]

    # --- 子图 1: R2 Score (越高越好) ---
    sns.barplot(data=results_df, x='Model', y='R2', palette=colors, ax=axes[0])
    axes[0].set_title('R² Score (Higher is Better)', fontweight='bold')
    axes[0].set_ylim(min(results_df['R2'])-0.02, 1.0) # 动态缩放
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(results_df['R2']):
        axes[0].text(i, v, f"{v:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- 子图 2: RMSE (越低越好) ---
    sns.barplot(data=results_df, x='Model', y='RMSE', palette=colors, ax=axes[1])
    axes[1].set_title('RMSE (Lower is Better)', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(results_df['RMSE']):
        axes[1].text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- 子图 3: MAE (越低越好) ---
    sns.barplot(data=results_df, x='Model', y='MAE', palette=colors, ax=axes[2])
    axes[2].set_title('MAE (Lower is Better)', fontweight='bold')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=15)
    for i, v in enumerate(results_df['MAE']):
        axes[2].text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    
    save_path = 'visual/Viz_All_Metrics_Comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 三合一对比图已生成: {save_path}")
    
else:
    print("\n❌ 没有结果，无法绘图。")