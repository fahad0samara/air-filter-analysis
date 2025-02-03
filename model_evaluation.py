import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def evaluate_predictions(y_true, y_pred):
    """Calculate multiple evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape,
        'Explained Variance': explained_variance
    }

def get_learning_curve(estimator, X, y, cv=5):
    """Generate learning curve data."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=train_sizes,
        scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    return {
        'train_sizes': train_sizes,
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std
    }

print("Loading and preparing data...")
# Read the cleaned data
df = pd.read_csv('cleaned_air_filter_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature engineering
def create_features(df):
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Performance metrics
    df['pm_ratio'] = np.where(df['inlet_pm10'] > 0, 
                             df['inlet_pm25'] / df['inlet_pm10'], 
                             0)
    df['pressure_efficiency_ratio'] = np.where(df['pressure_drop_pa'] > 0,
                                             df['efficiency'] / df['pressure_drop_pa'],
                                             0)
    df['load_age_ratio'] = np.where(df['filter_age_days'] > 0,
                                   df['load_factor'] / df['filter_age_days'],
                                   df['load_factor'])
    
    # Clip extreme values
    for col in ['pm_ratio', 'pressure_efficiency_ratio', 'load_age_ratio']:
        q1 = df[col].quantile(0.01)
        q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(q1, q3)
    
    return df

print("Creating enhanced features...")
df_enhanced = create_features(df)

# Define features
features = [
    'filter_age_days', 'load_factor', 'pressure_drop_pa',
    'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week', 'month',
    'pm_ratio', 'pressure_efficiency_ratio', 'load_age_ratio'
]

# Initialize model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Results storage
evaluation_results = []
cv_results = []
learning_curve_results = {}

print("\nPerforming detailed model evaluation...")
for filter_type in df['filter_class'].unique():
    print(f"\nEvaluating {filter_type}...")
    
    # Filter data for current type
    filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type].copy()
    
    # Prepare features and target
    X = filter_data[features]
    y = filter_data['efficiency']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # 1. Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_predictions = []
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = evaluate_predictions(y_val, y_pred)
        metrics['Fold'] = fold + 1
        fold_metrics.append(metrics)
        
        # Store predictions
        fold_predictions.append(pd.DataFrame({
            'Actual': y_val,
            'Predicted': y_pred,
            'Fold': fold + 1
        }))
    
    # Combine fold results
    fold_metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.concat(fold_predictions, ignore_index=True)
    
    # 2. Learning curves
    learning_curve_data = get_learning_curve(model, X_scaled, y)
    learning_curve_results[filter_type] = learning_curve_data
    
    # 3. Feature importance analysis
    model.fit(X_scaled, y)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Save results
    print(f"\nResults for {filter_type}:")
    print("\nCross-validation metrics (mean ± std):")
    for metric in ['R²', 'RMSE', 'MAE', 'MAPE', 'Explained Variance']:
        mean_val = fold_metrics_df[metric].mean()
        std_val = fold_metrics_df[metric].std()
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nTop 5 important features:")
    print(feature_importance.head())
    
    # Save detailed results to files
    fold_metrics_df.to_csv(f'{filter_type}_cross_validation_metrics.csv', index=False)
    predictions_df.to_csv(f'{filter_type}_predictions.csv', index=False)
    feature_importance.to_csv(f'{filter_type}_feature_importance_detailed.csv', index=False)
    
    # Save learning curve data
    pd.DataFrame({
        'Training Size': learning_curve_data['train_sizes'],
        'Training Score': learning_curve_data['train_mean'],
        'Validation Score': learning_curve_data['val_mean'],
        'Training Std': learning_curve_data['train_std'],
        'Validation Std': learning_curve_data['val_std']
    }).to_csv(f'{filter_type}_learning_curve.csv', index=False)

# Save overall summary
print("\nGenerating final summary report...")
with open('model_evaluation_summary.txt', 'w') as f:
    f.write("Model Evaluation Summary\n")
    f.write("======================\n\n")
    
    for filter_type in df['filter_class'].unique():
        f.write(f"\n{filter_type}\n")
        f.write("-" * len(filter_type) + "\n")
        
        # Load metrics for this filter type
        metrics_df = pd.read_csv(f'{filter_type}_cross_validation_metrics.csv')
        
        f.write("\nCross-validation metrics (mean ± std):\n")
        for metric in ['R²', 'RMSE', 'MAE', 'MAPE', 'Explained Variance']:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            f.write(f"{metric}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        # Load feature importance
        importance_df = pd.read_csv(f'{filter_type}_feature_importance_detailed.csv')
        
        f.write("\nTop 5 important features:\n")
        for _, row in importance_df.head().iterrows():
            f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
        
        f.write("\n" + "="*50 + "\n")

print("\nEvaluation complete! Generated files:")
print("1. '*_cross_validation_metrics.csv' - Detailed metrics for each fold")
print("2. '*_predictions.csv' - Actual vs predicted values")
print("3. '*_feature_importance_detailed.csv' - Detailed feature importance analysis")
print("4. '*_learning_curve.csv' - Learning curve data")
print("5. 'model_evaluation_summary.txt' - Complete evaluation summary")
