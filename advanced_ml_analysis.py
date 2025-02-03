import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
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
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Performance metrics
    df['pm_ratio'] = df['inlet_pm25'] / df['inlet_pm10']
    df['pressure_efficiency_ratio'] = df['efficiency'] / df['pressure_drop_pa']
    df['load_age_ratio'] = df['load_factor'] / df['filter_age_days']
    
    # Particle removal efficiency
    df['pm25_removal_efficiency'] = (df['inlet_pm25'] - df['outlet_pm25']) / df['inlet_pm25']
    df['pm10_removal_efficiency'] = (df['inlet_pm10'] - df['outlet_pm10']) / df['inlet_pm10']
    
    return df

# Create enhanced features
df_enhanced = create_features(df)

# Define features for modeling
numerical_features = [
    'filter_age_days', 'load_factor', 'pressure_drop_pa',
    'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week', 'month',
    'pm_ratio', 'pressure_efficiency_ratio', 'load_age_ratio',
    'pm25_removal_efficiency', 'pm10_removal_efficiency'
]

categorical_features = ['filter_class', 'location']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models with hyperparameter grids
models = {
    'XGBoost': (xgb.XGBRegressor(random_state=42),
                {'learning_rate': [0.01, 0.1],
                 'max_depth': [3, 5],
                 'n_estimators': [100, 200]}),
    
    'Neural Network': (MLPRegressor(random_state=42),
                      {'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                       'activation': ['relu', 'tanh'],
                       'alpha': [0.0001, 0.001]}),
    
    'Gradient Boosting': (GradientBoostingRegressor(random_state=42),
                         {'learning_rate': [0.01, 0.1],
                          'max_depth': [3, 5],
                          'n_estimators': [100, 200]})
}

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

# Function to create maintenance schedule
def create_maintenance_schedule(model, scaler, filter_data, filter_type):
    # Create prediction data for next 60 days
    future_dates = pd.date_range(start=filter_data['timestamp'].max(),
                               periods=60, freq='D')
    
    pred_data = pd.DataFrame({
        'timestamp': future_dates,
        'filter_age_days': range(len(future_dates)),
        'load_factor': [filter_data['load_factor'].mean()] * len(future_dates),
        'pressure_drop_pa': [filter_data['pressure_drop_pa'].mean()] * len(future_dates),
        'inlet_pm25': [filter_data['inlet_pm25'].mean()] * len(future_dates),
        'inlet_pm10': [filter_data['inlet_pm10'].mean()] * len(future_dates),
        'filter_class': [filter_type] * len(future_dates),
        'location': [filter_data['location'].iloc[0]] * len(future_dates)
    })
    
    # Create features for prediction
    pred_data = create_features(pred_data)
    
    # Transform features
    X_pred = preprocessor.transform(pred_data)
    
    # Make predictions
    efficiency_predictions = model.predict(X_pred)
    
    return pd.DataFrame({
        'Date': future_dates,
        'Predicted_Efficiency': efficiency_predictions
    })

# Results storage
results = []
maintenance_schedules = {}

# Train and evaluate models for each filter type
print("\nTraining models and generating maintenance schedules...")
for filter_type in df['filter_class'].unique():
    print(f"\nProcessing {filter_type}...")
    
    # Filter data for current type
    filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type].copy()
    
    # Prepare features and target
    X = filter_data[numerical_features + categorical_features]
    y = filter_data['efficiency']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model = None
    best_score = -np.inf
    
    # Train and evaluate each model
    for model_name, (model, param_grid) in models.items():
        print(f"Training {model_name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(pipeline, param_grid={'regressor__' + key: value 
                                                       for key, value in param_grid.items()},
                                 cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Make predictions
        y_pred = grid_search.predict(X_test)
        
        # Evaluate performance
        metrics = evaluate_model(y_test, y_pred, model_name)
        metrics['Filter_Type'] = filter_type
        results.append(metrics)
        
        # Update best model if necessary
        if metrics['R2'] > best_score:
            best_score = metrics['R2']
            best_model = grid_search
            
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Efficiency')
        plt.ylabel('Predicted Efficiency')
        plt.title(f'{filter_type} - {model_name} Predictions')
        plt.tight_layout()
        plt.savefig(f'{filter_type}_{model_name}_predictions.png')
        plt.close()
    
    # Generate maintenance schedule using best model
    schedule = create_maintenance_schedule(best_model, preprocessor, filter_data, filter_type)
    maintenance_schedules[filter_type] = schedule
    
    # Plot efficiency prediction
    plt.figure(figsize=(12, 6))
    plt.plot(schedule['Date'], schedule['Predicted_Efficiency'])
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Efficiency Threshold')
    plt.xlabel('Date')
    plt.ylabel('Predicted Efficiency')
    plt.title(f'{filter_type} - Predicted Efficiency Over Time')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filter_type}_efficiency_forecast.png')
    plt.close()

# Create summary results
results_df = pd.DataFrame(results)
print("\nModel Performance Summary:")
print(results_df.groupby(['Filter_Type', 'Model'])['R2', 'RMSE', 'MAE'].mean())

# Generate maintenance recommendations
print("\nMaintenance Recommendations:")
for filter_type, schedule in maintenance_schedules.items():
    threshold = 0.95
    replacement_date = schedule[schedule['Predicted_Efficiency'] < threshold].iloc[0]['Date']
    initial_efficiency = schedule['Predicted_Efficiency'].iloc[0]
    
    print(f"\n{filter_type}:")
    print(f"Initial Efficiency: {initial_efficiency:.3f}")
    print(f"Recommended Replacement Date: {replacement_date.strftime('%Y-%m-%d')}")
    print(f"Days until replacement: {(replacement_date - pd.Timestamp.now()).days}")

# Save detailed results
with open('advanced_ml_results.txt', 'w') as f:
    f.write("Advanced Machine Learning Analysis Results\n")
    f.write("=====================================\n\n")
    
    f.write("Model Performance Summary:\n")
    f.write(results_df.groupby(['Filter_Type', 'Model'])['R2', 'RMSE', 'MAE'].mean().to_string())
    
    f.write("\n\nMaintenance Recommendations:\n")
    for filter_type, schedule in maintenance_schedules.items():
        threshold = 0.95
        replacement_date = schedule[schedule['Predicted_Efficiency'] < threshold].iloc[0]['Date']
        initial_efficiency = schedule['Predicted_Efficiency'].iloc[0]
        
        f.write(f"\n{filter_type}:\n")
        f.write(f"Initial Efficiency: {initial_efficiency:.3f}\n")
        f.write(f"Recommended Replacement Date: {replacement_date.strftime('%Y-%m-%d')}\n")
        f.write(f"Days until replacement: {(replacement_date - pd.Timestamp.now()).days}\n")

print("\nAnalysis complete! Generated files:")
print("1. Model prediction plots for each filter type and model")
print("2. Efficiency forecast plots for each filter type")
print("3. advanced_ml_results.txt with detailed performance metrics and recommendations")
