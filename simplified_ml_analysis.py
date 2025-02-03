import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Loading and preparing data...")
# Read the cleaned data
df = pd.read_csv('cleaned_air_filter_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Enhanced feature engineering
def create_features(df):
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Performance metrics
    df['pm_ratio'] = df['inlet_pm25'] / df['inlet_pm10']
    df['pressure_efficiency_ratio'] = df['efficiency'] / df['pressure_drop_pa']
    df['load_age_ratio'] = df['load_factor'] / df['filter_age_days']
    
    return df

# Create enhanced features
df_enhanced = create_features(df)

# Define features for modeling
features = [
    'filter_age_days', 'load_factor', 'pressure_drop_pa',
    'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week', 'month',
    'pm_ratio', 'pressure_efficiency_ratio', 'load_age_ratio'
]

# Initialize models
models = {
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Results storage
results = []

# Train and evaluate models for each filter type
print("\nTraining models and generating predictions...")
for filter_type in df['filter_class'].unique():
    print(f"\nProcessing {filter_type}...")
    
    # Filter data for current type
    filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type].copy()
    
    # Prepare features and target
    X = filter_data[features]
    y = filter_data['efficiency']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Filter_Type': filter_type,
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
        
        # If using Random Forest or XGBoost, get feature importance
        if model_name in ['Random Forest', 'XGBoost']:
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 5 important features for {filter_type} using {model_name}:")
            print(importance.head())

# Create summary results
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df.groupby(['Filter_Type', 'Model'])[['R2', 'RMSE', 'MAE']].mean())

# Generate maintenance recommendations
print("\nMaintenance Recommendations:")
for filter_type in df['filter_class'].unique():
    filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type]
    
    # Get best model based on R2 score
    best_model_info = results_df[results_df['Filter_Type'] == filter_type].sort_values('R2', ascending=False).iloc[0]
    best_model_name = best_model_info['Model']
    best_model = models[best_model_name]
    
    # Create prediction data for different ages
    max_age = 60
    test_ages = np.arange(0, max_age)
    
    # Create test data with mean values
    test_data = pd.DataFrame({
        'filter_age_days': test_ages,
        'load_factor': [filter_data['load_factor'].mean()] * max_age,
        'pressure_drop_pa': [filter_data['pressure_drop_pa'].mean()] * max_age,
        'inlet_pm25': [filter_data['inlet_pm25'].mean()] * max_age,
        'inlet_pm10': [filter_data['inlet_pm10'].mean()] * max_age,
        'hour': [12] * max_age,
        'day_of_week': [3] * max_age,
        'month': [6] * max_age
    })
    
    # Add engineered features
    test_data['pm_ratio'] = test_data['inlet_pm25'] / test_data['inlet_pm10']
    test_data['pressure_efficiency_ratio'] = 0.9 / test_data['pressure_drop_pa']  # Assuming initial efficiency of 0.9
    test_data['load_age_ratio'] = test_data['load_factor'] / test_data['filter_age_days'].replace(0, 1)
    
    # Scale features
    test_data_scaled = scaler.transform(test_data[features])
    
    # Predict efficiencies
    predicted_efficiencies = best_model.predict(test_data_scaled)
    
    # Find replacement point (when efficiency drops below 95% of initial)
    initial_efficiency = predicted_efficiencies[0]
    threshold = initial_efficiency * 0.95
    replacement_age = test_ages[predicted_efficiencies < threshold][0] if any(predicted_efficiencies < threshold) else max_age
    
    print(f"\n{filter_type}:")
    print(f"Best Model: {best_model_name} (R² = {best_model_info['R2']:.3f})")
    print(f"Initial Efficiency: {initial_efficiency:.3f}")
    print(f"Recommended Replacement Age: {replacement_age} days")
    print(f"Efficiency at Replacement: {predicted_efficiencies[replacement_age]:.3f}")

# Save results to file
with open('ml_analysis_results.txt', 'w') as f:
    f.write("Machine Learning Analysis Results\n")
    f.write("================================\n\n")
    
    f.write("Model Performance Summary:\n")
    f.write(results_df.groupby(['Filter_Type', 'Model'])[['R2', 'RMSE', 'MAE']].mean().to_string())
    
    f.write("\n\nMaintenance Recommendations:\n")
    for filter_type in df['filter_class'].unique():
        best_model_info = results_df[results_df['Filter_Type'] == filter_type].sort_values('R2', ascending=False).iloc[0]
        f.write(f"\n{filter_type}:\n")
        f.write(f"Best Model: {best_model_info['Model']} (R² = {best_model_info['R2']:.3f})\n")

print("\nAnalysis complete! Results saved to 'ml_analysis_results.txt'")
