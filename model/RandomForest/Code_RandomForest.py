import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# Configuration
# ============================================================================
# Get the absolute path to the project root
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL_DIR))

DATA_PATH = os.path.join(PROJECT_ROOT, "Data_Merge", "f1_grand_dataset_full.csv")
RF_VIZ_PATH = os.path.join(MODEL_DIR, "randomforest_f1.png")
COMPARISON_VIZ_PATH = os.path.join(MODEL_DIR, "RandomForest_vs_Baseline.png")

# Settings
sns.set(style="whitegrid", context="talk")
plt.rcParams['axes.unicode_minus'] = False
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def load_and_preprocess_data():
    """
    Load F1 circuit data from merged CSV and preprocess for modeling.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, full_data)
    """
    print("=" * 70)
    print("Loading F1 Circuit Data...")
    print("=" * 70)
    
    # Load data from CSV file
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Print available column names
    print("\nAvailable columns:")
    print(df.columns.tolist())
    
    # Feature engineering: Calculate fastest lap time in seconds
    df['fastest_lap_seconds'] = df['LapTime']
    
    # Calculate average speed from sector information
    if 'sector_length_km_S1' in df.columns:
        total_length = (df['sector_length_km_S1'] + 
                       df['sector_length_km_S2'] + 
                       df['sector_length_km_S3'])
        df['AvgSpeed'] = np.where(
            df['LapTime'] > 0, 
            total_length / (df['LapTime'] / 3600), 
            0
        )
        print("✓ Calculated average speed (AvgSpeed) using sector lengths")
    elif 'Track Length (km)' in df.columns:
        df['AvgSpeed'] = df['Track Length (km)'] / (df['LapTime'] / 3600)
        print("✓ Calculated average speed (AvgSpeed) using total track length")
    else:
        print("⚠ Warning: No length data found, cannot calculate speed")
        df['AvgSpeed'] = 0
    
    # Select features for Random Forest model
    features = [
        'sector_straight_ratio_S1', 
        'sector_straight_ratio_S2', 
        'sector_straight_ratio_S3',
        'sector_slow_corner_ratio_S1', 
        'sector_slow_corner_ratio_S2',
        'sector_slow_corner_ratio_S3',
        'Global Straight Ratio',
        'Corners'
    ]
    
    # Keep only valid features that exist in the data
    valid_features = [col for col in features if col in df.columns]
    print(f"\n✓ Selected {len(valid_features)} valid features for modeling:")
    for i, feat in enumerate(valid_features, 1):
        print(f"  {i}. {feat}")
    
    # Clean data: remove rows with missing values or invalid target
    data = df[valid_features + ['AvgSpeed', 'circuit_name', 'Event Year']].copy()
    data = data.dropna()
    data = data[data['AvgSpeed'] > 0]
    
    print(f"\n✓ Data cleaned: {data.shape[0]} valid samples remaining")
    print(f"  - Removed {df.shape[0] - data.shape[0]} rows with missing/invalid values")
    
    # Separate features (X) and target variable (y)
    X = data[valid_features]
    y = data['AvgSpeed']
    
    if len(X) == 0:
        raise ValueError("No valid data available for training!")
    
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"\n✓ Data split completed:")
    print(f"  - Training set: {len(X_train)} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"  - Test set: {len(X_test)} samples ({TEST_SIZE*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test, valid_features, data


# ============================================================================
# Model Training and Evaluation
# ============================================================================
def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """
    Train Random Forest regressor and evaluate performance.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        feature_names: List of feature names
        
    Returns:
        dict: Model performance metrics
    """
    print("\n" + "=" * 70)
    print("Training Random Forest Regressor...")
    print("=" * 70)
    
    # Initialize and train Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    rf_model.fit(X_train, y_train)
    print("✓ Random Forest model training completed")
    
    # Make predictions on training and test sets
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    # Calculate evaluation metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n✓ Model Evaluation Metrics:")
    print(f"\n  Training Performance:")
    print(f"    - R² Score: {train_r2:.4f}")
    print(f"    - MAE: {train_mae:.4f}")
    print(f"    - RMSE: {train_rmse:.4f}")
    
    print(f"\n  Test Performance:")
    print(f"    - R² Score: {test_r2:.4f}")
    print(f"    - MAE: {test_mae:.4f}")
    print(f"    - RMSE: {test_rmse:.4f}")
    
    metrics = {
        'model': rf_model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_test_pred': y_test_pred,
        'feature_names': feature_names,
        'feature_importance': rf_model.feature_importances_
    }
    
    return metrics


def train_baseline_linear_model(X_train, X_test, y_train, y_test):
    """
    Train Linear Regression baseline model for comparison.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        
    Returns:
        dict: Baseline model metrics
    """
    print("\n" + "=" * 70)
    print("Training Baseline Linear Regression Model...")
    print("=" * 70)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    print("✓ Linear Regression model training completed")
    
    # Make predictions
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print("\n✓ Baseline Model Evaluation Metrics:")
    print(f"\n  Training Performance:")
    print(f"    - R² Score: {train_r2:.4f}")
    print(f"    - MAE: {train_mae:.4f}")
    print(f"    - RMSE: {train_rmse:.4f}")
    
    print(f"\n  Test Performance:")
    print(f"    - R² Score: {test_r2:.4f}")
    print(f"    - MAE: {test_mae:.4f}")
    print(f"    - RMSE: {test_rmse:.4f}")
    
    baseline_metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'y_test_pred': y_test_pred
    }
    
    return baseline_metrics


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_feature_importance(metrics):
    """
    Visualize Random Forest feature importance.
    
    Args:
        metrics: Dictionary containing model metrics and feature importance
    """
    print("\n" + "=" * 70)
    print("Generating Feature Importance Visualization...")
    print("=" * 70)
    
    feature_names = metrics['feature_names']
    feature_importance = metrics['feature_importance']
    test_r2 = metrics['test_r2']
    
    # Create feature importance dataframe and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot with color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    
    # Format plot labels and title
    ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Random Forest: Feature Importance for F1 Fastest Lap Prediction\nTest R² = {test_r2:.4f}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add value labels on bars
    for i, v in enumerate(importance_df['Importance']):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)
    
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save figure to file
    plt.savefig(RF_VIZ_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved: {RF_VIZ_PATH}")
    

def plot_diagnostic_panel(rf_metrics, baseline_metrics, y_test, feature_names, X_test):
    """
    综合诊断面板：特征重要性、误差分布、预测对比、残差分析
    """
    print("\nGenerating diagnostic_panel.png ...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Random Forest Diagnostic Panel', fontsize=18, fontweight='bold', y=0.98)

    # 1. 特征重要性
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_metrics['feature_importance']
    }).sort_values(by='Importance', ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(importance_df)))
    ax = axes[0, 0]
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_title('Feature Importance', fontsize=14)
    ax.set_xlabel('Importance')
    for i, v in enumerate(importance_df['Importance']):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 2. 误差分布
    errors = rf_metrics['y_test_pred'] - y_test
    ax = axes[0, 1]
    sns.histplot(errors, bins=10, kde=True, color='teal', edgecolor='black', ax=ax)
    ax.set_title('Prediction Error Distribution (Test Set)', fontsize=14)
    ax.set_xlabel('Error (km/h)')
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.3)

    # 3. 预测对比
    ax = axes[1, 0]
    ax.scatter(y_test, rf_metrics['y_test_pred'], alpha=0.7, label='Random Forest', s=60)
    ax.scatter(y_test, baseline_metrics['y_test_pred'], alpha=0.7, label='Linear Regression', s=60)
    min_val = min(y_test.min(), rf_metrics['y_test_pred'].min(), baseline_metrics['y_test_pred'].min())
    max_val = max(y_test.max(), rf_metrics['y_test_pred'].max(), baseline_metrics['y_test_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    ax.set_xlabel('Actual AvgSpeed')
    ax.set_ylabel('Predicted AvgSpeed')
    ax.set_title('Prediction Accuracy (Test Set)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. 残差与最重要特征
    importances = rf_metrics['feature_importance']
    top_idx = np.argsort(importances)[-1]
    top_feat = feature_names[top_idx]
    residuals = rf_metrics['y_test_pred'] - y_test
    ax = axes[1, 1]
    ax.scatter(X_test[top_feat], residuals, alpha=0.8, color='purple')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel(top_feat)
    ax.set_ylabel('Residual (km/h)')
    ax.set_title(f'Residuals vs {top_feat}', fontsize=14)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = os.path.join(MODEL_DIR, 'diagnostic_panel.png')
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✓ Saved: {fname}")

def plot_trend_panel(X_train, y_train, X_test, y_test, feature_names):
    """
    选取最重要的2个特征，展示趋势分析
    """
    print("\nGenerating trend_panel.png ...")
    importances = [np.std(X_train[feat]) for feat in feature_names]
    # 这里用标准差选2个变化最大的特征（也可用重要性）
    top2 = np.argsort(importances)[-2:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, idx in enumerate(top2):
        feat = feature_names[idx]
        ax = axes[i]
        ax.scatter(X_train[feat], y_train, alpha=0.5, label='Train', color='dodgerblue')
        ax.scatter(X_test[feat], y_test, alpha=0.8, label='Test', color='orange')
        z = np.polyfit(X_train[feat], y_train, 1)
        p = np.poly1d(z)
        x_line = np.linspace(X_train[feat].min(), X_train[feat].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2, label='Trend')
        ax.set_xlabel(feat)
        ax.set_ylabel('AvgSpeed')
        ax.set_title(f'Trend: {feat} vs AvgSpeed')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(MODEL_DIR, 'trend_panel.png')
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✓ Saved: {fname}")

def plot_partial_dependence_panel(rf_model, X_train, feature_names):
    """
    只对最重要的2个特征做部分依赖图
    """
    print("\nGenerating partial_dependence_panel.png ...")
    from sklearn.inspection import PartialDependenceDisplay
    importances = rf_model.feature_importances_
    top2 = np.argsort(importances)[-2:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, idx in enumerate(top2):
        PartialDependenceDisplay.from_estimator(rf_model, X_train, [idx], ax=axes[i])
        axes[i].set_title(f'Partial Dependence: {feature_names[idx]}')
    plt.tight_layout()
    fname = os.path.join(MODEL_DIR, 'partial_dependence_panel.png')
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"✓ Saved: {fname}")

def plot_feature_trends(X_train, y_train, X_test, y_test, feature_names, output_dir):
    """
    Plot each feature vs target variable (trend analysis).
    """
    print("\nGenerating feature vs target trend plots...")
    for feat in feature_names:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[feat], y_train, alpha=0.5, label='Train', color='dodgerblue')
        plt.scatter(X_test[feat], y_test, alpha=0.8, label='Test', color='orange')
        # Fit linear trend
        z = np.polyfit(X_train[feat], y_train, 1)
        p = np.poly1d(z)
        x_line = np.linspace(X_train[feat].min(), X_train[feat].max(), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2, label='Trend')
        plt.xlabel(feat, fontsize=12)
        plt.ylabel('AvgSpeed', fontsize=12)
        plt.title(f'Trend: {feat} vs AvgSpeed', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        fname = os.path.join(output_dir, f'trend_{feat}.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"✓ Saved: {fname}")

def plot_error_distribution(y_test, y_pred, output_dir):
    """
    Plot error distribution for test set predictions.
    """
    errors = y_pred - y_test
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=10, kde=True, color='teal', edgecolor='black')
    plt.title('Prediction Error Distribution (Test Set)', fontsize=14)
    plt.xlabel('Prediction Error (km/h)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(alpha=0.3)
    fname = os.path.join(output_dir, 'error_distribution.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"✓ Saved: {fname}")

def plot_train_vs_test_scatter(y_train, y_train_pred, y_test, y_test_pred, output_dir):
    """
    Scatter plot: train vs test predictions.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train', color='dodgerblue')
    plt.scatter(y_test, y_test_pred, alpha=0.8, label='Test', color='orange')
    min_val = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    plt.xlabel('Actual AvgSpeed', fontsize=12)
    plt.ylabel('Predicted AvgSpeed', fontsize=12)
    plt.title('Train vs Test Prediction Accuracy', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    fname = os.path.join(output_dir, 'train_vs_test_scatter.png')
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"✓ Saved: {fname}")

def plot_partial_dependence(rf_model, X, features, feature_names, output_dir):
    """
    Partial dependence plots for key features.
    """
    try:
        from sklearn.inspection import PartialDependenceDisplay
        for idx in features:
            fname = os.path.join(output_dir, f'partial_dependence_{feature_names[idx]}.png')
            fig, ax = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(rf_model, X, [idx], ax=ax)
            plt.title(f'Partial Dependence: {feature_names[idx]}', fontsize=13)
            plt.tight_layout()
            plt.savefig(fname, dpi=200)
            plt.close()
            print(f"✓ Saved: {fname}")
    except Exception as e:
        print(f"Partial dependence plot error: {e}")

def plot_residual_vs_features(X_test, y_test, y_pred, feature_names, output_dir):
    """
    Residual vs feature plots for test set.
    """
    residuals = y_pred - y_test
    for feat in feature_names:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[feat], residuals, alpha=0.8, color='purple')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel(feat, fontsize=12)
        plt.ylabel('Residual (km/h)', fontsize=12)
        plt.title(f'Residuals vs {feat} (Test Set)', fontsize=14)
        plt.grid(alpha=0.3)
        fname = os.path.join(output_dir, f'residual_vs_{feat}.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"✓ Saved: {fname}")


def plot_model_comparison(rf_metrics, baseline_metrics, y_test):
    """
    Compare Random Forest and Linear Regression models.
    
    Args:
        rf_metrics: Random Forest model metrics
        baseline_metrics: Linear Regression baseline metrics
        y_test: Actual test values
    """
    print("\nGenerating Model Comparison Visualization...")
    
    # Prepare comparison data for plotting
    models = ['Random Forest', 'Linear Regression']
    train_r2 = [rf_metrics['train_r2'], baseline_metrics['train_r2']]
    test_r2 = [rf_metrics['test_r2'], baseline_metrics['test_r2']]
    train_mae = [rf_metrics['train_mae'], baseline_metrics['train_mae']]
    test_mae = [rf_metrics['test_mae'], baseline_metrics['test_mae']]
    train_rmse = [rf_metrics['train_rmse'], baseline_metrics['train_rmse']]
    test_rmse = [rf_metrics['test_rmse'], baseline_metrics['test_rmse']]
    
    # Create subplots (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Random Forest vs Linear Regression: Model Comparison', 
                 fontsize=16, fontweight='bold', y=1.00)
    
    # Plot 1: R² Score Comparison
    x = np.arange(len(models))
    width = 0.35
    ax = axes[0, 0]
    ax.bar(x - width/2, train_r2, width, label='Train', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, test_r2, width, label='Test', alpha=0.8, color='coral')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('R² Score Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(max(train_r2), max(test_r2)) * 1.1])
    
    # Plot 2: MAE Comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, train_mae, width, label='Train', alpha=0.8, color='lightgreen')
    ax.bar(x + width/2, test_mae, width, label='Test', alpha=0.8, color='salmon')
    ax.set_ylabel('MAE', fontweight='bold')
    ax.set_title('Mean Absolute Error Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: RMSE Comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, train_rmse, width, label='Train', alpha=0.8, color='plum')
    ax.bar(x + width/2, test_rmse, width, label='Test', alpha=0.8, color='gold')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('Root Mean Squared Error Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Prediction Accuracy Scatter Plot
    ax = axes[1, 1]
    ax.scatter(y_test, rf_metrics['y_test_pred'], alpha=0.6, label='Random Forest', s=50)
    ax.scatter(y_test, baseline_metrics['y_test_pred'], alpha=0.6, label='Linear Regression', s=50)
    
    # Add perfect prediction line (diagonal)
    min_val = min(y_test.min(), y_test.max())
    max_val = max(y_test.min(), y_test.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Values', fontweight='bold')
    ax.set_ylabel('Predicted Values', fontweight='bold')
    ax.set_title('Prediction Accuracy (Test Set)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(COMPARISON_VIZ_PATH, dpi=300, bbox_inches='tight')
    print(f"✓ Model comparison plot saved: {COMPARISON_VIZ_PATH}")
    
    plt.close()


def print_summary_report(rf_metrics, baseline_metrics):
    """
    Print a comprehensive summary report comparing both models.
    
    Args:
        rf_metrics: Random Forest model metrics
        baseline_metrics: Linear Regression baseline metrics
    """
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY REPORT")
    print("=" * 70)
    
    print("\n╔════════════════════════════════════════════════════════════════════╗")
    print("║             RANDOM FOREST vs LINEAR REGRESSION                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    # Display R² comparison
    print("\n┌─ R² Score (Higher is Better) ───────────────────────────────────┐")
    print(f"│ Random Forest    - Train: {rf_metrics['train_r2']:.4f}  |  Test: {rf_metrics['test_r2']:.4f}")
    print(f"│ Linear Regression - Train: {baseline_metrics['train_r2']:.4f}  |  Test: {baseline_metrics['test_r2']:.4f}")
    rf_r2_improvement = ((rf_metrics['test_r2'] - baseline_metrics['test_r2']) / 
                         abs(baseline_metrics['test_r2'])) * 100 if baseline_metrics['test_r2'] != 0 else 0
    print(f"│ Improvement (RF): {rf_r2_improvement:+.2f}%")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Display MAE comparison
    print("\n┌─ MAE - Mean Absolute Error (Lower is Better) ──────────────────┐")
    print(f"│ Random Forest    - Train: {rf_metrics['train_mae']:.4f}  |  Test: {rf_metrics['test_mae']:.4f}")
    print(f"│ Linear Regression - Train: {baseline_metrics['train_mae']:.4f}  |  Test: {baseline_metrics['test_mae']:.4f}")
    mae_improvement = ((baseline_metrics['test_mae'] - rf_metrics['test_mae']) / 
                       baseline_metrics['test_mae']) * 100 if baseline_metrics['test_mae'] != 0 else 0
    print(f"│ Improvement (RF): {mae_improvement:+.2f}%")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Display RMSE comparison
    print("\n┌─ RMSE - Root Mean Squared Error (Lower is Better) ──────────────┐")
    print(f"│ Random Forest    - Train: {rf_metrics['train_rmse']:.4f}  |  Test: {rf_metrics['test_rmse']:.4f}")
    print(f"│ Linear Regression - Train: {baseline_metrics['train_rmse']:.4f}  |  Test: {baseline_metrics['test_rmse']:.4f}")
    rmse_improvement = ((baseline_metrics['test_rmse'] - rf_metrics['test_rmse']) / 
                        baseline_metrics['test_rmse']) * 100 if baseline_metrics['test_rmse'] != 0 else 0
    print(f"│ Improvement (RF): {rmse_improvement:+.2f}%")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n✓ Key Insights:")
    print(f"  • Random Forest captures {rf_metrics['test_r2']*100:.1f}% of the variance in fastest lap times")
    print(f"  • Average prediction error (MAE): {rf_metrics['test_mae']:.4f} seconds")
    print(f"  • Model standard deviation (RMSE): {rf_metrics['test_rmse']:.4f} seconds")
    
    if rf_r2_improvement > 0:
        print(f"  • RF outperforms Linear Regression by {rf_r2_improvement:.1f}% in R² score")
    else:
        print(f"  • Linear Regression performs {-rf_r2_improvement:.1f}% better in R² score")
    
    print("\n" + "=" * 70)


# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Main execution function."""
    try:
        print("\n")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║     F1 Fastest Lap Prediction: Random Forest Model Analysis       ║")
        print("╚════════════════════════════════════════════════════════════════════╝")

        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names, full_data = load_and_preprocess_data()

        # Train Random Forest model
        rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, feature_names)

        # Train Baseline Linear Regression for comparison
        baseline_metrics = train_baseline_linear_model(X_train, X_test, y_train, y_test)

        # 只生成3张高信息密度PNG
        plot_diagnostic_panel(rf_metrics, baseline_metrics, y_test, feature_names, X_test)
        plot_trend_panel(X_train, y_train, X_test, y_test, feature_names)
        plot_partial_dependence_panel(rf_metrics['model'], X_train, feature_names)

        # Print summary report
        print_summary_report(rf_metrics, baseline_metrics)

        print("\n✓ Analysis Complete!")
        print(f"✓ Visualizations saved to: {MODEL_DIR}/")
        print("\n")
        return 0
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
