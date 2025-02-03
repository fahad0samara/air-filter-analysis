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

# Calculate efficiency decline rate
def calculate_efficiency_decline(data, filter_type):
    filter_data = data[data['filter_class'] == filter_type].copy()
    filter_data = filter_data.sort_values('filter_age_days')
    
    # Calculate daily efficiency change
    z = np.polyfit(filter_data['filter_age_days'], filter_data['efficiency'], 1)
    slope = z[0]
    
    return slope * 100  # Convert to percentage

# Calculate optimal replacement timing
def calculate_optimal_replacement(data, filter_type, efficiency_threshold=0.95):
    filter_data = data[data['filter_class'] == filter_type].copy()
    max_efficiency = filter_data['efficiency'].max()
    threshold = max_efficiency * efficiency_threshold
    
    # Find the earliest age where efficiency drops below threshold
    below_threshold = filter_data[filter_data['efficiency'] < threshold]
    if not below_threshold.empty:
        optimal_age = below_threshold['filter_age_days'].min()
    else:
        optimal_age = filter_data['filter_age_days'].max()
    
    return optimal_age, max_efficiency, threshold

# Cost analysis (using relative costs)
filter_costs = {
    'HEPA': 100,
    'Activated Carbon': 80,
    'Electrostatic': 70,
    'Pre-Filter': 30
}

# Analyze performance patterns
def analyze_performance_patterns(data, filter_type):
    filter_data = data[data['filter_class'] == filter_type].copy()
    
    # Calculate key metrics
    avg_efficiency = filter_data['efficiency'].mean()
    std_efficiency = filter_data['efficiency'].std()
    avg_pressure_drop = filter_data['pressure_drop_pa'].mean()
    pm_removal = (
        (filter_data['inlet_pm25'] - filter_data['outlet_pm25']).mean() / filter_data['inlet_pm25'].mean() * 100,
        (filter_data['inlet_pm10'] - filter_data['outlet_pm10']).mean() / filter_data['inlet_pm10'].mean() * 100
    )
    
    return {
        'avg_efficiency': avg_efficiency,
        'std_efficiency': std_efficiency,
        'avg_pressure_drop': avg_pressure_drop,
        'pm25_removal': pm_removal[0],
        'pm10_removal': pm_removal[1]
    }

# Visualization of efficiency trends
plt.figure(figsize=(12, 6))
for filter_type in df['filter_class'].unique():
    filter_data = df[df['filter_class'] == filter_type]
    plt.scatter(filter_data['filter_age_days'], filter_data['efficiency'], 
                alpha=0.1, label=f'{filter_type} (data)')
    
    # Add trend line
    z = np.polyfit(filter_data['filter_age_days'], filter_data['efficiency'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(filter_data['filter_age_days'].min(), 
                         filter_data['filter_age_days'].max(), 100)
    plt.plot(x_trend, p(x_trend), '--', label=f'{filter_type} (trend)')

plt.xlabel('Filter Age (days)')
plt.ylabel('Efficiency')
plt.title('Filter Efficiency Trends Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('efficiency_trends.png')
plt.close()

# Calculate and store results
results = {}
for filter_type in df['filter_class'].unique():
    decline_rate = calculate_efficiency_decline(df, filter_type)
    optimal_age, max_eff, threshold = calculate_optimal_replacement(df, filter_type)
    patterns = analyze_performance_patterns(df, filter_type)
    
    cost_per_day = filter_costs[filter_type] / optimal_age
    cost_per_efficiency = cost_per_day / patterns['avg_efficiency']
    
    results[filter_type] = {
        'decline_rate': decline_rate,
        'optimal_replacement_age': optimal_age,
        'max_efficiency': max_eff,
        'cost_per_day': cost_per_day,
        'cost_per_efficiency': cost_per_efficiency,
        **patterns
    }

# Create summary visualizations
# 1. Cost-effectiveness comparison
plt.figure(figsize=(10, 6))
cost_data = pd.DataFrame({
    'Cost per Day': [results[ft]['cost_per_day'] for ft in results],
    'Cost per Efficiency': [results[ft]['cost_per_efficiency'] for ft in results]
}, index=results.keys())
cost_data.plot(kind='bar')
plt.title('Cost-Effectiveness Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('cost_comparison.png')
plt.close()

# 2. Performance metrics comparison
plt.figure(figsize=(12, 6))
performance_data = pd.DataFrame({
    'Average Efficiency': [results[ft]['avg_efficiency'] for ft in results],
    'PM2.5 Removal (%)': [results[ft]['pm25_removal'] for ft in results],
    'PM10 Removal (%)': [results[ft]['pm10_removal'] for ft in results]
}, index=results.keys())
performance_data.plot(kind='bar')
plt.title('Performance Metrics Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.close()

# Print detailed results
print("\nDetailed Analysis Results:")
print("=" * 50)
for filter_type, metrics in results.items():
    print(f"\n{filter_type}:")
    print(f"Efficiency Decline Rate: {metrics['decline_rate']:.3f}% per day")
    print(f"Optimal Replacement Age: {metrics['optimal_replacement_age']:.1f} days")
    print(f"Maximum Efficiency: {metrics['max_efficiency']:.3f}")
    print(f"Average Efficiency: {metrics['avg_efficiency']:.3f}")
    print(f"PM2.5 Removal: {metrics['pm25_removal']:.1f}%")
    print(f"PM10 Removal: {metrics['pm10_removal']:.1f}%")
    print(f"Cost per Day: {metrics['cost_per_day']:.2f}")
    print(f"Cost per Efficiency Unit: {metrics['cost_per_efficiency']:.2f}")

# Save results to file
with open('detailed_analysis_results.txt', 'w') as f:
    f.write("Air Filter Analysis Results\n")
    f.write("=========================\n\n")
    
    for filter_type, metrics in results.items():
        f.write(f"\n{filter_type}:\n")
        f.write(f"Efficiency Decline Rate: {metrics['decline_rate']:.3f}% per day\n")
        f.write(f"Optimal Replacement Age: {metrics['optimal_replacement_age']:.1f} days\n")
        f.write(f"Maximum Efficiency: {metrics['max_efficiency']:.3f}\n")
        f.write(f"Average Efficiency: {metrics['avg_efficiency']:.3f}\n")
        f.write(f"PM2.5 Removal: {metrics['pm25_removal']:.1f}%\n")
        f.write(f"PM10 Removal: {metrics['pm10_removal']:.1f}%\n")
        f.write(f"Cost per Day: {metrics['cost_per_day']:.2f}\n")
        f.write(f"Cost per Efficiency Unit: {metrics['cost_per_efficiency']:.2f}\n")
        f.write("-" * 40 + "\n")

print("\nAnalysis complete! Generated files:")
print("1. efficiency_trends.png - Efficiency trends over time for all filter types")
print("2. cost_comparison.png - Cost-effectiveness comparison")
print("3. performance_comparison.png - Performance metrics comparison")
print("4. detailed_analysis_results.txt - Comprehensive analysis results")
