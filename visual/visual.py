import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Set output folder
OUTPUT_FOLDER = 'visual'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['axes.unicode_minus'] = False 

# Load data
file_path = 'f1_grand_dataset_full.csv'
if not os.path.exists(file_path):
    print("Error: File not found. Please run the Merge script first.")
    exit()

df = pd.read_csv(file_path)

# Calculate average speed
if 'sector_length_km_S1' in df.columns:
    total_length = df['sector_length_km_S1'] + df['sector_length_km_S2'] + df['sector_length_km_S3']
    df['AvgSpeed'] = np.where(df['LapTime'] > 0, total_length / (df['LapTime'] / 3600), 0)
else:
    df['AvgSpeed'] = 0 

print("Data loaded. Generating plots...")

# Plot 1: Sector Time Distribution
print("Plotting: [1] Sector Distribution...")
plt.figure(figsize=(10, 6))

cols = ['sector_time_ratio_S1', 'sector_time_ratio_S2', 'sector_time_ratio_S3']
if all(c in df.columns for c in cols):
    df_grouped = df.groupby('circuit_name')[cols].mean()
    df_grouped.plot(kind='bar', stacked=True, color=['#FF9999', '#66B2FF', '#99FF99'], figsize=(10, 6))
    
    plt.title('Lap Time Distribution by Sector', fontsize=15)
    plt.ylabel('Percentage of Lap Time (%)', fontsize=12)
    plt.xlabel('Circuit', fontsize=12)
    plt.legend(['Sector 1', 'Sector 2', 'Sector 3'], loc='upper right')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_FOLDER}/Viz_1_Sector_Distribution.png", dpi=300)
    print("  -> Saved to visual folder")

# Plot 2: Correlation Heatmap
print("Plotting: [2] Correlation Heatmap...")
plt.figure(figsize=(12, 10))

cols_of_interest = [
    'LapTime', 'AvgSpeed',
    'sector_straight_ratio_S1', 'sector_straight_ratio_S2', 'sector_straight_ratio_S3',
    'sector_length_km_S1', 'sector_length_km_S2', 'sector_length_km_S3',
    'sector_slow_corner_ratio_S1', 'sector_slow_corner_ratio_S2'
]
existing_cols = [c for c in cols_of_interest if c in df.columns]

if len(existing_cols) > 2:
    corr_matrix = df[existing_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    
    plt.title('Correlation Matrix: Geometry vs Performance', fontsize=15)
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_FOLDER}/Viz_2_Correlation_Heatmap.png", dpi=300)
    print("  -> Saved to visual folder")

# Plot 3: Regression Scatter Plot
print("Plotting: [3] Regression Scatter...")

if 'sector_straight_ratio_S1' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sector_straight_ratio_S1', y='AvgSpeed', hue='circuit_name', s=100, palette='deep')
    
    plt.title('Impact of S1 Straight Ratio on Average Speed', fontsize=15)
    plt.xlabel('Sector 1 Straight Line Ratio (%)', fontsize=12)
    plt.ylabel('Average Lap Speed (km/h)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_FOLDER}/Viz_3_Regression_Scatter.png", dpi=300)
    print("  -> Saved to visual folder")

print("\nAll plots generated and saved to visual folder.")