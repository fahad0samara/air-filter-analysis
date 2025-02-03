# Air Filter Analysis Dashboard 🌬️

An interactive Streamlit dashboard for analyzing air filter performance with advanced visualizations and machine learning insights.

## Features

### 1. Overview Dashboard 📊
- Real-time performance metrics
- Interactive gauge charts
- Time series analysis
- 3D performance visualization
- Pressure vs Load analysis
- Radar charts and parallel coordinates

### 2. Statistical Analysis 📈
- Correlation analysis
- Distribution analysis
- ANOVA test results
- Box plots by filter class

### 3. Pattern Analysis 🔄
- Multi-parameter parallel coordinates
- Interactive filtering
- Pattern detection

### 4. Advanced Metrics 🔬
- Efficiency stability
- Pressure efficiency
- Load rate analysis
- Performance over time

### 5. Machine Learning Analysis 🤖
- Random Forest and XGBoost models
- Learning curves
- Feature importance analysis
- Performance metrics (R², RMSE, MAE)

## Installation

1. Clone the repository:

cd air-filter-analysis
```

2. Install requirements:
```bash
pip install -r requirements_streamlit.txt
```

3. Run the app:
```bash
streamlit run enhanced_visualizations.py
```

## Data Format

The dashboard expects a CSV file with the following columns:
- timestamp
- filter_class
- efficiency
- pressure_drop_pa
- load_factor
- filter_age_days
- inlet_pm25
- inlet_pm10

## Usage

1. Select analysis type from the sidebar
2. Choose filter types to analyze
3. Interact with visualizations:
   - Click and drag on plots
   - Hover for detailed information
   - Use sliders and dropdowns for filtering

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- xgboost
- scipy

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

