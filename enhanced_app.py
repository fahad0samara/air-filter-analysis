import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="Advanced Air Filter Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Caching functions
@st.cache_data
def load_data():
    """Load and preprocess the data"""
    df = pd.read_csv('cleaned_air_filter_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_data
def create_features(df):
    """Create enhanced features for modeling"""
    df = df.copy()
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Performance metrics
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

def get_model_predictions(filter_data, features, scaler):
    """Get model predictions and performance metrics"""
    X = filter_data[features]
    y = filter_data['efficiency']
    
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'metrics': {
                'R²': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
        }
    
    return results

def create_download_link(df, filename):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def generate_pdf_report(filter_type, model_results, predictions):
    """Generate a PDF report with analysis results"""
    report = f"""
    Air Filter Analysis Report
    =========================
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Filter Type: {filter_type}
    
    Model Performance
    ----------------
    """
    
    for model_name, results in model_results.items():
        report += f"\n{model_name}:\n"
        for metric, value in results['metrics'].items():
            report += f"{metric}: {value:.4f}\n"
    
    return report

# Load data
try:
    df = load_data()
    df_enhanced = create_features(df)
    
    # Define features
    features = [
        'filter_age_days', 'load_factor', 'pressure_drop_pa',
        'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week', 'month',
        'pm_ratio', 'pressure_efficiency_ratio', 'load_age_ratio',
        'is_weekend'
    ]
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Advanced Analysis", "Comparison Tool", "Report Generator"]
    )
    
    if page == "Dashboard":
        st.title("📊 Air Filter Performance Dashboard")
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Average Efficiency", f"{df['efficiency'].mean():.3f}")
        with col3:
            st.metric("Date Range", f"{(df['timestamp'].max() - df['timestamp'].min()).days} days")
        
        # Interactive filter selection
        filter_type = st.selectbox("Select Filter Type", df['filter_class'].unique())
        filter_data = df[df['filter_class'] == filter_type]
        
        # Performance over time
        st.subheader("Performance Trends")
        fig = px.line(filter_data, 
                     x='timestamp', 
                     y='efficiency',
                     title=f'{filter_type} Efficiency Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(filter_data, 
                             x='efficiency',
                             title='Efficiency Distribution',
                             marginal='box')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(filter_data,
                           x='filter_age_days',
                           y='efficiency',
                           color='load_factor',
                           title='Efficiency vs Age',
                           trendline="lowess")
            st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Advanced Analysis":
        st.title("🔬 Advanced Analysis")
        
        filter_type = st.selectbox("Select Filter Type", df['filter_class'].unique())
        filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type]
        
        # Model training and evaluation
        scaler = RobustScaler()
        model_results = get_model_predictions(filter_data, features, scaler)
        
        # Model comparison
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame([
            {
                'Model': model_name,
                **results['metrics']
            }
            for model_name, results in model_results.items()
        ])
        st.write(metrics_df)
        
        # Feature importance
        st.subheader("Feature Importance Analysis")
        best_model = model_results['Random Forest']['model']
        importance = pd.DataFrame({
            'feature': features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(importance, 
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-validation analysis
        st.subheader("Cross-validation Analysis")
        cv_scores = cross_val_score(best_model, 
                                  scaler.transform(filter_data[features]),
                                  filter_data['efficiency'],
                                  cv=5)
        st.write(f"Cross-validation scores: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
        
    elif page == "Comparison Tool":
        st.title("🔄 Filter Comparison Tool")
        
        col1, col2 = st.columns(2)
        with col1:
            filter1 = st.selectbox("Select First Filter", df['filter_class'].unique(), key='filter1')
        with col2:
            filter2 = st.selectbox("Select Second Filter", df['filter_class'].unique(), key='filter2')
        
        # Compare efficiencies
        fig = go.Figure()
        for filter_type in [filter1, filter2]:
            filter_data = df[df['filter_class'] == filter_type]
            fig.add_trace(go.Box(y=filter_data['efficiency'],
                                name=filter_type))
        fig.update_layout(title='Efficiency Comparison',
                         yaxis_title='Efficiency')
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical comparison
        st.subheader("Statistical Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['Mean Efficiency', 'Median Efficiency', 'Std Dev', 'Min', 'Max'],
            filter1: [
                df[df['filter_class'] == filter1]['efficiency'].mean(),
                df[df['filter_class'] == filter1]['efficiency'].median(),
                df[df['filter_class'] == filter1]['efficiency'].std(),
                df[df['filter_class'] == filter1]['efficiency'].min(),
                df[df['filter_class'] == filter1]['efficiency'].max()
            ],
            filter2: [
                df[df['filter_class'] == filter2]['efficiency'].mean(),
                df[df['filter_class'] == filter2]['efficiency'].median(),
                df[df['filter_class'] == filter2]['efficiency'].std(),
                df[df['filter_class'] == filter2]['efficiency'].min(),
                df[df['filter_class'] == filter2]['efficiency'].max()
            ]
        })
        st.write(comparison_df)
        
    else:  # Report Generator
        st.title("📑 Report Generator")
        
        # Report configuration
        st.subheader("Configure Report")
        filter_type = st.selectbox("Select Filter Type", df['filter_class'].unique())
        include_sections = st.multiselect(
            "Select sections to include",
            ["Basic Statistics", "Performance Analysis", "Model Predictions", "Maintenance Recommendations"],
            default=["Basic Statistics", "Performance Analysis"]
        )
        
        if st.button("Generate Report"):
            filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type]
            
            # Create report components
            report_data = {
                "Filter Type": filter_type,
                "Analysis Date": datetime.now().strftime("%Y-%m-%d"),
                "Sections": {}
            }
            
            if "Basic Statistics" in include_sections:
                report_data["Sections"]["Basic Statistics"] = {
                    "Sample Size": len(filter_data),
                    "Mean Efficiency": filter_data['efficiency'].mean(),
                    "Median Efficiency": filter_data['efficiency'].median(),
                    "Efficiency Range": [
                        filter_data['efficiency'].min(),
                        filter_data['efficiency'].max()
                    ]
                }
            
            if "Performance Analysis" in include_sections:
                scaler = RobustScaler()
                model_results = get_model_predictions(filter_data, features, scaler)
                report_data["Sections"]["Performance Analysis"] = {
                    "Model Results": {
                        name: results['metrics']
                        for name, results in model_results.items()
                    }
                }
            
            # Generate report
            report_json = json.dumps(report_data, indent=2)
            st.download_button(
                label="Download Report (JSON)",
                data=report_json,
                file_name=f"air_filter_report_{filter_type}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
            
            # Display report preview
            st.subheader("Report Preview")
            st.json(report_data)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please make sure all required files and dependencies are available.")
