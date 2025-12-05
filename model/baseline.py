import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import joblib

# Settings
OUTPUT_FOLDER = 'visual'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
sns.set(style="whitegrid", context="talk")
plt.rcParams['axes.unicode_minus'] = False 

# Load data
df = pd.read_csv('E:\\Machine learning\\ML-Analysis-of-F1-Fatest-Lap-Circuits\\Data_Merge\\f1_grand_dataset_full.csv')

# Print column names
print("=== Data Columns List ===")
print(df.columns.tolist())
print("====================")

# Feature engineering: Calculate average speed
if 'sector_length_km_S1' in df.columns:
    total_length = df['sector_length_km_S1'] + df['sector_length_km_S2'] + df['sector_length_km_S3']
    df['AvgSpeed'] = np.where(df['LapTime'] > 0, total_length / (df['LapTime'] / 3600), 0)
    print("Success: Calculated average speed (AvgSpeed) using sector lengths")
    
elif 'Track Length (km)' in df.columns:
    df['AvgSpeed'] = df['Track Length (km)'] / (df['LapTime'] / 3600)
    print("Success: Calculated average speed (AvgSpeed) using total track length")
else:
    print("Warning: No length data found, cannot calculate speed")
    df['AvgSpeed'] = 0

# Select features and target
features = [
    'sector_straight_ratio_S1', 
    'sector_straight_ratio_S2', 
    'sector_straight_ratio_S3',
    'sector_slow_corner_ratio_S1', 
    'sector_slow_corner_ratio_S2'
]

valid_features = [c for c in features if c in df.columns]

# Clean data
data = df[valid_features + ['AvgSpeed']].dropna()
data = data[data['AvgSpeed'] > 0]

X = data[valid_features]
y = data['AvgSpeed']

if len(X) == 0:
    print("Error: No valid data for training")
    exit()

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Visualization: Feature importance
coef_df = pd.DataFrame({
    'Feature': valid_features,
    'Impact (Coefficient)': model.coef_
})
coef_df = coef_df.sort_values(by='Impact (Coefficient)', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=coef_df, x='Impact (Coefficient)', y='Feature', palette='coolwarm')

plt.title(f'What drives Speed? (Feature Importance)\nBaseline R² = {r2:.3f}', fontsize=16, weight='bold')
plt.xlabel('Impact on Average Speed (Positive=Faster, Negative=Slower)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()

save_path = f"{OUTPUT_FOLDER}/Viz_5_Feature_Impact.png"
plt.savefig(save_path, dpi=300)
print(f"Success: Analysis complete! Feature importance plot saved: {save_path}")
print(f"New R² (for speed): {r2:.3f}")
model_save_path = 'E:\\Machine learning\\ML-Analysis-of-F1-Fatest-Lap-Circuits\\model\\baseline.pkl'
joblib.dump(model, model_save_path)