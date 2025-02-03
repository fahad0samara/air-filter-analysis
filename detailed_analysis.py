import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the cleaned data
df = pd.read_csv('cleaned_air_filter_data.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Filter Performance Analysis
plt.figure(figsize=(15, 8))
sns.boxplot(x='filter_class', y='efficiency', data=df)
plt.title('Efficiency Distribution by Filter Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('filter_efficiency_distribution.png')
plt.close()

# 2. PM Removal Efficiency
df['pm25_removal_efficiency'] = (df['inlet_pm25'] - df['outlet_pm25']) / df['inlet_pm25'] * 100
df['pm10_removal_efficiency'] = (df['inlet_pm10'] - df['outlet_pm10']) / df['inlet_pm10'] * 100

plt.figure(figsize=(12, 6))
removal_data = pd.DataFrame({
    'PM2.5': df.groupby('filter_class')['pm25_removal_efficiency'].mean(),
    'PM10': df.groupby('filter_class')['pm10_removal_efficiency'].mean()
})
removal_data.plot(kind='bar')
plt.title('PM Removal Efficiency by Filter Type')
plt.ylabel('Removal Efficiency (%)')
plt.xticks(rotation=45)
plt.legend(title='Particle Size')
plt.tight_layout()
plt.savefig('pm_removal_efficiency.png')
plt.close()

# 3. Pressure Drop vs Load Factor
plt.figure(figsize=(10, 6))
for filter_type in df['filter_class'].unique():
    filter_data = df[df['filter_class'] == filter_type]
    plt.scatter(filter_data['load_factor'], filter_data['pressure_drop_pa'], 
                alpha=0.5, label=filter_type)
plt.xlabel('Load Factor')
plt.ylabel('Pressure Drop (Pa)')
plt.title('Pressure Drop vs Load Factor by Filter Type')
plt.legend()
plt.tight_layout()
plt.savefig('pressure_vs_load.png')
plt.close()

# 4. Filter Age Impact
plt.figure(figsize=(12, 6))
age_bins = [0, 7, 14, 21, 28, float('inf')]
df['age_category'] = pd.cut(df['filter_age_days'], bins=age_bins, 
                           labels=['0-7 days', '7-14 days', '14-21 days', '21-28 days', '28+ days'])
sns.boxplot(x='age_category', y='efficiency', hue='filter_class', data=df)
plt.title('Filter Efficiency by Age and Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('efficiency_by_age.png')
plt.close()

# 5. Performance Statistics
performance_stats = df.groupby('filter_class').agg({
    'efficiency': ['mean', 'std'],
    'pressure_drop_pa': ['mean', 'std'],
    'pm25_removal_efficiency': 'mean',
    'pm10_removal_efficiency': 'mean',
    'filter_age_days': 'mean'
}).round(2)

print("\nPerformance Statistics by Filter Type:")
print(performance_stats)

# 6. Time-based Analysis
df['month'] = df['timestamp'].dt.month
monthly_performance = df.groupby(['filter_class', 'month'])['efficiency'].mean().unstack()
plt.figure(figsize=(12, 6))
monthly_performance.plot(marker='o')
plt.title('Monthly Filter Efficiency Trends')
plt.xlabel('Filter Type')
plt.ylabel('Average Efficiency')
plt.legend(title='Month')
plt.tight_layout()
plt.savefig('monthly_efficiency.png')
plt.close()

# Save detailed statistics to CSV
performance_stats.to_csv('filter_performance_stats.csv')

print("\nAnalysis complete! Generated visualizations:")
print("1. filter_efficiency_distribution.png - Shows efficiency distribution for each filter type")
print("2. pm_removal_efficiency.png - Compares PM2.5 and PM10 removal efficiency")
print("3. pressure_vs_load.png - Shows relationship between pressure drop and load factor")
print("4. efficiency_by_age.png - Shows how efficiency changes with filter age")
print("5. monthly_efficiency.png - Shows monthly efficiency trends")
print("\nDetailed statistics have been saved to 'filter_performance_stats.csv'")
