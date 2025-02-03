import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Read the cleaned data
print("Loading and preparing data...")
df = pd.read_csv('cleaned_air_filter_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Feature engineering
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['pm_ratio'] = df['inlet_pm25'] / df['inlet_pm10']

# Prepare features for modeling
features = ['filter_age_days', 'load_factor', 'pressure_drop_pa', 
           'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week', 'month', 'pm_ratio']

# Create separate models for each filter type
filter_types = df['filter_class'].unique()
models = {}
results = {}

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Function to plot actual vs predicted values
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Efficiency')
    plt.ylabel('Predicted Efficiency')
    plt.title(title)
    plt.tight_layout()
    return plt

print("\nTraining models for each filter type...")
for filter_type in filter_types:
    print(f"\nProcessing {filter_type}...")
    
    # Filter data for current filter type
    filter_data = df[df['filter_class'] == filter_type].copy()
    
    # Prepare features and target
    X = filter_data[features]
    y = filter_data['efficiency']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models_to_train = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Linear Regression': LinearRegression()
    }
    
    # Train and evaluate models
    filter_results = {}
    for model_name, model in models_to_train.items():
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate performance
        metrics = evaluate_model(y_test, y_pred, model_name)
        filter_results[model_name] = metrics
        
        # Plot predictions
        plt = plot_predictions(y_test, y_pred, f'{filter_type} - {model_name}')
        plt.savefig(f'{filter_type}_{model_name}_predictions.png')
        plt.close()
        
        # If Random Forest, get feature importance
        if model_name == 'Random Forest':
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=importance)
            plt.title(f'Feature Importance for {filter_type}')
            plt.tight_layout()
            plt.savefig(f'{filter_type}_feature_importance.png')
            plt.close()
    
    results[filter_type] = filter_results

# Print summary results
print("\nModel Performance Summary:")
print("=" * 50)
for filter_type in filter_types:
    print(f"\n{filter_type}:")
    for model_name, metrics in results[filter_type].items():
        print(f"\n{model_name}:")
        print(f"R² Score: {metrics['R2']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")

# Save results to file
with open('ml_results.txt', 'w') as f:
    f.write("Machine Learning Analysis Results\n")
    f.write("==============================\n\n")
    
    for filter_type in filter_types:
        f.write(f"\n{filter_type}:\n")
        f.write("-" * 30 + "\n")
        for model_name, metrics in results[filter_type].items():
            f.write(f"\n{model_name}:\n")
            f.write(f"R² Score: {metrics['R2']:.3f}\n")
            f.write(f"RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"MAE: {metrics['MAE']:.4f}\n")

print("\nAnalysis complete! Generated files:")
print("1. Prediction plots for each filter type and model")
print("2. Feature importance plots for each filter type")
print("3. ml_results.txt with detailed performance metrics")

# Predict optimal replacement times
print("\nCalculating optimal replacement times...")
best_models = {}
for filter_type in filter_types:
    # Get best model based on R² score
    best_model_name = max(results[filter_type].items(), 
                         key=lambda x: x[1]['R2'])[0]
    print(f"\n{filter_type} - Best model: {best_model_name}")
    
    # Filter data
    filter_data = df[df['filter_class'] == filter_type].copy()
    
    # Create prediction data for different ages
    age_range = np.arange(0, 60, 1)  # Predict up to 60 days
    pred_data = pd.DataFrame({
        'filter_age_days': age_range,
        'load_factor': [filter_data['load_factor'].mean()] * len(age_range),
        'pressure_drop_pa': [filter_data['pressure_drop_pa'].mean()] * len(age_range),
        'inlet_pm25': [filter_data['inlet_pm25'].mean()] * len(age_range),
        'inlet_pm10': [filter_data['inlet_pm10'].mean()] * len(age_range),
        'hour': [12] * len(age_range),
        'day_of_week': [3] * len(age_range),
        'month': [6] * len(age_range),
        'pm_ratio': [filter_data['pm_ratio'].mean()] * len(age_range)
    })
    
    # Scale prediction data
    scaler = StandardScaler()
    scaler.fit(filter_data[features])
    pred_data_scaled = scaler.transform(pred_data)
    
    # Get predictions
    model = models_to_train[best_model_name]
    efficiency_predictions = model.predict(pred_data_scaled)
    
    # Plot efficiency prediction curve
    plt.figure(figsize=(10, 6))
    plt.plot(age_range, efficiency_predictions)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Efficiency Threshold')
    plt.xlabel('Filter Age (days)')
    plt.ylabel('Predicted Efficiency')
    plt.title(f'{filter_type} - Predicted Efficiency Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filter_type}_efficiency_prediction.png')
    plt.close()
    
    # Find optimal replacement time (when efficiency drops below 95% of initial)
    initial_efficiency = efficiency_predictions[0]
    threshold = initial_efficiency * 0.95
    replacement_age = age_range[efficiency_predictions < threshold][0] if any(efficiency_predictions < threshold) else 60
    
    print(f"Recommended replacement age: {replacement_age} days")
    print(f"Initial efficiency: {initial_efficiency:.3f}")
    print(f"Efficiency at replacement: {efficiency_predictions[age_range == replacement_age][0]:.3f}")

print("\nML analysis and optimization complete!")
