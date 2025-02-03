import pandas as pd
import numpy as np

# Read the data
df = pd.read_csv('air_filter_data.csv')

# 1. Remove NA values
df_clean = df.dropna()

# 2. Remove outliers using IQR method for numerical columns
numerical_cols = ['filter_age_days', 'load_factor', 'pressure_drop_pa', 'efficiency', 
                 'inlet_pm25', 'outlet_pm25', 'inlet_pm10', 'outlet_pm10']

for col in numerical_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

# Save the cleaned data
df_clean.to_csv('cleaned_air_filter_data.csv', index=False)

# Print summary statistics
print("\nOriginal data shape:", df.shape)
print("Cleaned data shape:", df_clean.shape)
print("\nSummary of removed data:")
print(f"Total rows removed: {df.shape[0] - df_clean.shape[0]}")
print("\nSummary statistics of cleaned data:")
print(df_clean.describe())
