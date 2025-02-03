import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read the cleaned data
df = pd.read_csv('cleaned_air_filter_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Correlation Analysis
numerical_cols = ['filter_age_days', 'load_factor', 'pressure_drop_pa', 'efficiency',
                 'inlet_pm25', 'outlet_pm25', 'inlet_pm10', 'outlet_pm10']

plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Key Metrics')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 2. Efficiency Degradation Analysis
plt.figure(figsize=(12, 6))
for filter_type in df['filter_class'].unique():
    filter_data = df[df['filter_class'] == filter_type]
    z = np.polyfit(filter_data['filter_age_days'], filter_data['efficiency'], 1)
    p = np.poly1d(z)
    plt.scatter(filter_data['filter_age_days'], filter_data['efficiency'], 
                alpha=0.1, label=f'{filter_type} (data)')
    plt.plot(filter_data['filter_age_days'], p(filter_data['filter_age_days']), 
             label=f'{filter_type} (trend)')

plt.xlabel('Filter Age (days)')
plt.ylabel('Efficiency')
plt.title('Efficiency Degradation Over Time')
plt.legend()
plt.tight_layout()
plt.savefig('efficiency_degradation.png')
plt.close()

# 3. Performance Under Load Analysis
df['performance_ratio'] = df['efficiency'] / df['pressure_drop_pa']

plt.figure(figsize=(12, 6))
sns.boxplot(x='filter_class', y='performance_ratio', data=df)
plt.title('Performance Ratio (Efficiency/Pressure Drop) by Filter Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_ratio.png')
plt.close()

# 4. Time Series Analysis
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Hourly patterns
plt.figure(figsize=(15, 5))
hourly_efficiency = df.groupby(['hour', 'filter_class'])['efficiency'].mean().unstack()
hourly_efficiency.plot(marker='o')
plt.title('Hourly Efficiency Patterns by Filter Type')
plt.xlabel('Hour of Day')
plt.ylabel('Average Efficiency')
plt.legend(title='Filter Type')
plt.tight_layout()
plt.savefig('hourly_patterns.png')
plt.close()

# 5. Principal Component Analysis
scaler = StandardScaler()
pca = PCA()
numerical_data = df[numerical_cols]
scaled_data = scaler.fit_transform(numerical_data)
pca_result = pca.fit_transform(scaled_data)

# Explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Analysis - Explained Variance')
plt.tight_layout()
plt.savefig('pca_analysis.png')
plt.close()

# 6. Statistical Analysis
stats_by_filter = df.groupby('filter_class').agg({
    'efficiency': ['mean', 'std', 'min', 'max'],
    'pressure_drop_pa': ['mean', 'std', 'min', 'max'],
    'performance_ratio': ['mean', 'std'],
    'filter_age_days': ['mean', 'std']
}).round(3)

# Save detailed statistics
stats_by_filter.to_csv('advanced_statistics.csv')

# Calculate and print key findings
print("\nAdvanced Analysis Results:")
print("\n1. Correlation Analysis:")
high_correlations = []
for i in range(len(numerical_cols)):
    for j in range(i+1, len(numerical_cols)):
        corr = correlation_matrix.iloc[i,j]
        if abs(corr) > 0.5:
            high_correlations.append(f"{numerical_cols[i]} vs {numerical_cols[j]}: {corr:.3f}")
print("\nStrong correlations found (|r| > 0.5):")
for corr in high_correlations:
    print(corr)

print("\n2. Efficiency Analysis by Filter Type:")
for filter_type in df['filter_class'].unique():
    filter_data = df[df['filter_class'] == filter_type]
    slope = np.polyfit(filter_data['filter_age_days'], filter_data['efficiency'], 1)[0]
    print(f"\n{filter_type}:")
    print(f"Average efficiency: {filter_data['efficiency'].mean():.3f}")
    print(f"Efficiency degradation rate: {slope:.6f} per day")
    print(f"Average performance ratio: {filter_data['performance_ratio'].mean():.3f}")

# 7. Peak Performance Analysis
peak_performance = df.groupby('filter_class').agg({
    'efficiency': lambda x: x.nlargest(10).mean(),
    'pressure_drop_pa': lambda x: x.nsmallest(10).mean()
}).round(3)

print("\n3. Peak Performance Metrics:")
print(peak_performance)

# 8. Save summary to file
with open('analysis_summary.txt', 'w') as f:
    f.write("Air Filter Performance Analysis Summary\n")
    f.write("=====================================\n\n")
    f.write("1. Correlation Analysis:\n")
    for corr in high_correlations:
        f.write(f"  - {corr}\n")
    
    f.write("\n2. Efficiency Analysis by Filter Type:\n")
    for filter_type in df['filter_class'].unique():
        filter_data = df[df['filter_class'] == filter_type]
        slope = np.polyfit(filter_data['filter_age_days'], filter_data['efficiency'], 1)[0]
        f.write(f"\n{filter_type}:\n")
        f.write(f"  - Average efficiency: {filter_data['efficiency'].mean():.3f}\n")
        f.write(f"  - Efficiency degradation rate: {slope:.6f} per day\n")
        f.write(f"  - Average performance ratio: {filter_data['performance_ratio'].mean():.3f}\n")

print("\nAnalysis complete! Generated files:")
print("1. correlation_matrix.png - Correlation heatmap of key metrics")
print("2. efficiency_degradation.png - Efficiency trends over time")
print("3. performance_ratio.png - Performance ratio by filter type")
print("4. hourly_patterns.png - Hourly efficiency patterns")
print("5. pca_analysis.png - Principal Component Analysis results")
print("6. advanced_statistics.csv - Detailed statistical analysis")
print("7. analysis_summary.txt - Comprehensive analysis summary")
