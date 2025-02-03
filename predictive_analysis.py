import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Read the cleaned data
df = pd.read_csv('cleaned_air_filter_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Predictive Model for Filter Efficiency
def build_efficiency_model(data, filter_type):
    filter_data = data[data['filter_class'] == filter_type].copy()
    
    # Features for prediction
    features = ['filter_age_days', 'load_factor', 'pressure_drop_pa', 
                'inlet_pm25', 'inlet_pm10', 'hour']
    
    X = filter_data[features]
    y = filter_data['efficiency']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.pred(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, r2, rmse, importance

# 2. Calculate Optimal Replacement Points
def calculate_replacement_metrics(data):
    replacement_metrics = {}
    
    for filter_type in data['filter_class'].unique():
        filter_data = data[data['filter_class'] == filter_type]
        
        # Calculate efficiency threshold (5% below max efficiency)
        max_eff = filter_data['efficiency'].max()
        threshold = max_eff * 0.95
        
        # Find average age when efficiency drops below threshold
        below_threshold = filter_data[filter_data['efficiency'] < threshold]
        if not below_threshold.empty:
            optimal_age = below_threshold['filter_age_days'].min()
        else:
            optimal_age = filter_data['filter_age_days'].max()
            
        replacement_metrics[filter_type] = {
            'optimal_replacement_age': optimal_age,
            'max_efficiency': max_eff,
            'efficiency_threshold': threshold
        }
    
    return replacement_metrics

# 3. Cost-Effectiveness Analysis
# Assuming relative cost ratios for different filter types
filter_costs = {
    'HEPA': 100,  # Base cost reference
    'Activated Carbon': 80,
    'Electrostatic': 70,
    'Pre-Filter': 30
}

def calculate_cost_effectiveness(data, replacement_metrics, filter_costs):
    cost_metrics = {}
    
    for filter_type in data['filter_class'].unique():
        filter_data = data[data['filter_class'] == filter_type]
        
        # Calculate average daily efficiency
        avg_efficiency = filter_data.groupby('filter_age_days')['efficiency'].mean()
        
        # Calculate efficiency-days (area under efficiency curve)
        efficiency_days = avg_efficiency.sum()
        
        # Calculate cost per efficiency-day
        replacement_age = replacement_metrics[filter_type]['optimal_replacement_age']
        cost_per_day = filter_costs[filter_type] / replacement_age
        cost_per_efficiency = cost_per_day / avg_efficiency.mean()
        
        cost_metrics[filter_type] = {
            'efficiency_days': efficiency_days,
            'cost_per_day': cost_per_day,
            'cost_per_efficiency': cost_per_efficiency
        }
    
    return cost_metrics

# 4. Visualization Functions
def plot_efficiency_prediction(data, filter_type, model):
    plt.figure(figsize=(12, 6))
    
    filter_data = data[data['filter_class'] == filter_type]
    
    # Sort by age for smooth plotting
    age_range = np.linspace(0, filter_data['filter_age_days'].max(), 100)
    
    # Create prediction data
    pred_data = pd.DataFrame({
        'filter_age_days': age_range,
        'load_factor': [filter_data['load_factor'].mean()] * len(age_range),
        'pressure_drop_pa': [filter_data['pressure_drop_pa'].mean()] * len(age_range),
        'inlet_pm25': [filter_data['inlet_pm25'].mean()] * len(age_range),
        'inlet_pm10': [filter_data['inlet_pm10'].mean()] * len(age_range),
        'hour': [12] * len(age_range)  # Midday reference
    })
    
    # Plot actual vs predicted
    plt.scatter(filter_data['filter_age_days'], filter_data['efficiency'], 
                alpha=0.3, label='Actual Data')
    plt.plot(age_range, model.predict(pred_data), 'r-', label='Predicted Trend')
    
    plt.xlabel('Filter Age (days)')
    plt.ylabel('Efficiency')
    plt.title(f'{filter_type} Efficiency Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{filter_type.lower()}_prediction.png')
    plt.close()

def plot_cost_effectiveness(cost_metrics):
    metrics_df = pd.DataFrame(cost_metrics).T
    
    plt.figure(figsize=(12, 6))
    metrics_df[['cost_per_day', 'cost_per_efficiency']].plot(kind='bar')
    plt.title('Cost Metrics by Filter Type')
    plt.xlabel('Filter Type')
    plt.ylabel('Cost (relative units)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cost_effectiveness.png')
    plt.close()

# Main Analysis
print("Starting Predictive Analysis...")

# Build predictive models for each filter type
models = {}
for filter_type in df['filter_class'].unique():
    print(f"\nAnalyzing {filter_type}...")
    model, r2, rmse, importance = build_efficiency_model(df, filter_type)
    models[filter_type] = {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'importance': importance
    }
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print("\nFeature Importance:")
    print(importance)
    
    # Generate prediction plot
    plot_efficiency_prediction(df, filter_type, model)

# Calculate replacement metrics
replacement_metrics = calculate_replacement_metrics(df)
print("\nOptimal Replacement Ages:")
for filter_type, metrics in replacement_metrics.items():
    print(f"\n{filter_type}:")
    print(f"Optimal replacement age: {metrics['optimal_replacement_age']:.1f} days")
    print(f"Maximum efficiency: {metrics['max_efficiency']:.3f}")
    print(f"Efficiency threshold: {metrics['efficiency_threshold']:.3f}")

# Calculate cost effectiveness
cost_metrics = calculate_cost_effectiveness(df, replacement_metrics, filter_costs)
plot_cost_effectiveness(cost_metrics)

print("\nCost-Effectiveness Metrics:")
for filter_type, metrics in cost_metrics.items():
    print(f"\n{filter_type}:")
    print(f"Cost per day: {metrics['cost_per_day']:.2f}")
    print(f"Cost per efficiency unit: {metrics['cost_per_efficiency']:.2f}")

# Save comprehensive results to file
with open('predictive_analysis_results.txt', 'w') as f:
    f.write("Predictive Analysis Results\n")
    f.write("=========================\n\n")
    
    f.write("1. Model Performance\n")
    for filter_type, model_metrics in models.items():
        f.write(f"\n{filter_type}:\n")
        f.write(f"R² Score: {model_metrics['r2']:.3f}\n")
        f.write(f"RMSE: {model_metrics['rmse']:.3f}\n")
        f.write("Feature Importance:\n")
        f.write(model_metrics['importance'].to_string())
        f.write("\n")
    
    f.write("\n2. Replacement Recommendations\n")
    for filter_type, metrics in replacement_metrics.items():
        f.write(f"\n{filter_type}:\n")
        f.write(f"Optimal replacement age: {metrics['optimal_replacement_age']:.1f} days\n")
        f.write(f"Maximum efficiency: {metrics['max_efficiency']:.3f}\n")
        f.write(f"Efficiency threshold: {metrics['efficiency_threshold']:.3f}\n")
    
    f.write("\n3. Cost-Effectiveness Analysis\n")
    for filter_type, metrics in cost_metrics.items():
        f.write(f"\n{filter_type}:\n")
        f.write(f"Cost per day: {metrics['cost_per_day']:.2f}\n")
        f.write(f"Cost per efficiency unit: {metrics['cost_per_efficiency']:.2f}\n")

print("\nAnalysis complete! Generated files:")
print("1. [filter_type]_prediction.png - Efficiency prediction plots for each filter type")
print("2. cost_effectiveness.png - Cost comparison across filter types")
print("3. predictive_analysis_results.txt - Comprehensive analysis results")
