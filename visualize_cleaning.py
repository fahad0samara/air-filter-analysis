import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read both original and cleaned data
df_original = pd.read_csv('air_filter_data.csv')
df_cleaned = pd.read_csv('cleaned_air_filter_data.csv')

# Set style for better visualization
plt.style.use('seaborn')
sns.set_palette("husl")

# Create a function to plot distributions
def plot_distributions(original_data, cleaned_data, columns, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        # Plot original data
        sns.kdeplot(data=original_data, x=col, ax=axes[idx], label='Original', alpha=0.6)
        # Plot cleaned data
        sns.kdeplot(data=cleaned_data, x=col, ax=axes[idx], label='Cleaned', alpha=0.6)
        
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('distribution_comparison.png')
    plt.close()

# Create box plots for numerical columns
def create_boxplots(original_data, cleaned_data, columns, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        # Prepare data for box plots
        data_to_plot = [
            original_data[col].values,
            cleaned_data[col].values
        ]
        
        # Create box plot
        axes[idx].boxplot(data_to_plot, labels=['Original', 'Cleaned'])
        axes[idx].set_title(f'Box Plot of {col}')
        axes[idx].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('boxplot_comparison.png')
    plt.close()

# Create efficiency comparison by filter type
def plot_efficiency_by_filter(original_data, cleaned_data):
    plt.figure(figsize=(12, 6))
    
    # Calculate mean efficiency by filter type
    orig_eff = original_data.groupby('filter_class')['efficiency'].mean()
    clean_eff = cleaned_data.groupby('filter_class')['efficiency'].mean()
    
    # Create grouped bar plot
    x = range(len(orig_eff))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], orig_eff.values, width, label='Original', alpha=0.6)
    plt.bar([i + width/2 for i in x], clean_eff.values, width, label='Cleaned', alpha=0.6)
    
    plt.xlabel('Filter Class')
    plt.ylabel('Average Efficiency')
    plt.title('Average Filter Efficiency by Filter Class')
    plt.xticks(x, orig_eff.index, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('efficiency_comparison.png')
    plt.close()

# Select columns to visualize
numerical_cols = [
    'filter_age_days', 'load_factor', 'pressure_drop_pa', 'efficiency',
    'inlet_pm25', 'outlet_pm25', 'inlet_pm10', 'outlet_pm10'
]

# Create visualizations
plot_distributions(df_original, df_cleaned, numerical_cols, rows=2, cols=4)
create_boxplots(df_original, df_cleaned, numerical_cols, rows=2, cols=4)
plot_efficiency_by_filter(df_original, df_cleaned)

# Print summary statistics
print("\nSummary of changes in key metrics:")
for col in numerical_cols:
    print(f"\n{col}:")
    print(f"Original mean: {df_original[col].mean():.2f}")
    print(f"Cleaned mean: {df_cleaned[col].mean():.2f}")
    print(f"Original std: {df_original[col].std():.2f}")
    print(f"Cleaned std: {df_cleaned[col].std():.2f}")
