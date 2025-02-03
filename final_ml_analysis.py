import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Loading and preparing data...")
# Read the cleaned data
df = pd.read_csv('cleaned_air_filter_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Enhanced feature engineering with numerical stability
def create_features(df):
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Performance metrics (with safety checks)
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

# Create enhanced features
print("Creating enhanced features...")
df_enhanced = create_features(df)

# Define features for modeling
features = [
    'filter_age_days', 'load_factor', 'pressure_drop_pa',
    'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week', 'month',
    'pm_ratio', 'pressure_efficiency_ratio', 'load_age_ratio'
]

# Initialize models with conservative parameters
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, 
                                         max_depth=10,
                                         min_samples_split=5,
                                         min_samples_leaf=2,
                                         random_state=42)
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
    
    # Use RobustScaler for better handling of outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
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
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 important features for {filter_type} using {model_name}:")
        print(importance.head())
        
        # Save feature importance
        importance.to_csv(f'{filter_type}_feature_importance.csv', index=False)

# Create summary results
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df.groupby(['Filter_Type', 'Model'])[['R2', 'RMSE', 'MAE']].mean())

# Generate maintenance recommendations
print("\nMaintenance Recommendations:")
for filter_type in df['filter_class'].unique():
    filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type]
    model = models['Random Forest']  # Using Random Forest for all predictions
    
    # Create prediction data for different ages
    max_age = 60
    test_ages = np.arange(0, max_age)
    
    # Create test data with median values (more robust than mean)
    test_data = pd.DataFrame({
        'filter_age_days': test_ages,
        'load_factor': [filter_data['load_factor'].median()] * max_age,
        'pressure_drop_pa': [filter_data['pressure_drop_pa'].median()] * max_age,
        'inlet_pm25': [filter_data['inlet_pm25'].median()] * max_age,
        'inlet_pm10': [filter_data['inlet_pm10'].median()] * max_age,
        'hour': [12] * max_age,
        'day_of_week': [3] * max_age,
        'month': [6] * max_age,
        'pm_ratio': [filter_data['pm_ratio'].median()] * max_age,
        'pressure_efficiency_ratio': [filter_data['pressure_efficiency_ratio'].median()] * max_age,
        'load_age_ratio': [filter_data['load_age_ratio'].median()] * max_age
    })
    
    # Scale features
    test_data_scaled = scaler.transform(test_data[features])
    
    # Predict efficiencies
    predicted_efficiencies = model.predict(test_data_scaled)
    
    # Find replacement point (when efficiency drops below 95% of initial)
    initial_efficiency = predicted_efficiencies[0]
    threshold = initial_efficiency * 0.95
    replacement_age = test_ages[predicted_efficiencies < threshold][0] if any(predicted_efficiencies < threshold) else max_age
    
    print(f"\n{filter_type}:")
    print(f"Initial Efficiency: {initial_efficiency:.3f}")
    print(f"Recommended Replacement Age: {replacement_age} days")
    print(f"Efficiency at Replacement: {predicted_efficiencies[replacement_age]:.3f}")
    
    # Save predictions for this filter type
    pd.DataFrame({
        'Age': test_ages,
        'Predicted_Efficiency': predicted_efficiencies
    }).to_csv(f'{filter_type}_efficiency_predictions.csv', index=False)

# Save results to file
with open('final_ml_analysis_results.txt', 'w') as f:
    f.write("Final Machine Learning Analysis Results\n")
    f.write("====================================\n\n")
    
    f.write("Model Performance Summary:\n")
    f.write(results_df.groupby(['Filter_Type', 'Model'])[['R2', 'RMSE', 'MAE']].mean().to_string())
    
    f.write("\n\nMaintenance Recommendations:\n")
    for filter_type in df['filter_class'].unique():
        best_model_info = results_df[results_df['Filter_Type'] == filter_type].iloc[0]
        f.write(f"\n{filter_type}:\n")
        f.write(f"R² Score: {best_model_info['R2']:.3f}\n")
        f.write(f"RMSE: {best_model_info['RMSE']:.4f}\n")
        f.write(f"MAE: {best_model_info['MAE']:.4f}\n")

print("\nAnalysis complete! Results saved to:")
print("1. 'final_ml_analysis_results.txt' - Complete analysis results")
print("2. '*_feature_importance.csv' - Feature importance for each filter type")
print("3. '*_efficiency_predictions.csv' - Efficiency predictions over time")
