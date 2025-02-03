import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Air Filter Performance Analysis",
    page_icon="🔍",
    layout="wide"
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
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🔍 Air Filter Performance Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of air filter performance using machine learning.
Explore filter efficiency, predict maintenance needs, and optimize replacement schedules.
""")

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

# Load data
try:
    df = load_data()
    df_enhanced = create_features(df)
    
    # Define features for modeling
    features = [
        'filter_age_days', 'load_factor', 'pressure_drop_pa',
        'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week', 'month',
        'pm_ratio', 'pressure_efficiency_ratio', 'load_age_ratio'
    ]
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Analysis", "Model Predictions", "Maintenance Recommendations"]
    )
    
    if page == "Overview":
        st.header("📊 Data Overview")
        
        # Display basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Summary")
            st.write(f"Total Records: {len(df):,}")
            st.write(f"Filter Types: {', '.join(df['filter_class'].unique())}")
            st.write(f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        
        with col2:
            st.subheader("Average Efficiency by Filter Type")
            efficiency_stats = df.groupby('filter_class')['efficiency'].mean().round(3)
            st.write(efficiency_stats)
        
        # Overall efficiency distribution
        st.subheader("Efficiency Distribution by Filter Type")
        fig = px.box(df, x='filter_class', y='efficiency', 
                    color='filter_class',
                    title='Efficiency Distribution by Filter Type')
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Data Analysis":
        st.header("🔎 Detailed Data Analysis")
        
        # Filter selection
        filter_type = st.selectbox("Select Filter Type", df['filter_class'].unique())
        
        # Filter data
        filter_data = df[df['filter_class'] == filter_type]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Efficiency vs Age
            fig = px.scatter(filter_data, 
                           x='filter_age_days', 
                           y='efficiency',
                           title=f'Efficiency vs Age for {filter_type}',
                           trendline="lowess")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pressure Drop vs Efficiency
            fig = px.scatter(filter_data,
                           x='pressure_drop_pa',
                           y='efficiency',
                           title=f'Efficiency vs Pressure Drop for {filter_type}',
                           trendline="lowess")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = filter_data.select_dtypes(include=[np.number]).columns
        corr = filter_data[numeric_cols].corr()
        fig = px.imshow(corr, 
                       title=f'Correlation Heatmap for {filter_type}',
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig, use_container_width=True)
        
    elif page == "Model Predictions":
        st.header("🤖 Model Predictions")
        
        # Filter selection
        filter_type = st.selectbox("Select Filter Type", df['filter_class'].unique())
        
        # Train model for selected filter
        filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type].copy()
        
        X = filter_data[features]
        y = filter_data['efficiency']
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Plot actual vs predicted
        fig = px.scatter(x=y_test, y=y_pred,
                        title=f'Actual vs Predicted Efficiency for {filter_type}',
                        labels={'x': 'Actual Efficiency', 'y': 'Predicted Efficiency'})
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines',
                                name='Perfect Prediction',
                                line=dict(dash='dash', color='red')))
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig = px.bar(importance, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title=f'Feature Importance for {filter_type}')
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Maintenance Recommendations
        st.header("🔧 Maintenance Recommendations")
        
        # Filter selection
        filter_type = st.selectbox("Select Filter Type", df['filter_class'].unique())
        
        # Train model and make predictions
        filter_data = df_enhanced[df_enhanced['filter_class'] == filter_type].copy()
        
        X = filter_data[features]
        y = filter_data['efficiency']
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Create prediction data
        max_age = 60
        test_ages = np.arange(0, max_age)
        
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
        
        # Scale features and predict
        test_data_scaled = scaler.transform(test_data[features])
        predicted_efficiencies = model.predict(test_data_scaled)
        
        # Plot efficiency prediction
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_ages, 
                                y=predicted_efficiencies,
                                mode='lines',
                                name='Predicted Efficiency'))
        fig.add_hline(y=predicted_efficiencies[0] * 0.95,
                     line_dash="dash",
                     line_color="red",
                     annotation_text="95% of Initial Efficiency")
        fig.update_layout(title=f'Predicted Efficiency Over Time for {filter_type}',
                         xaxis_title='Filter Age (days)',
                         yaxis_title='Predicted Efficiency')
        st.plotly_chart(fig, use_container_width=True)
        
        # Find replacement point
        initial_efficiency = predicted_efficiencies[0]
        threshold = initial_efficiency * 0.95
        replacement_age = test_ages[predicted_efficiencies < threshold][0] if any(predicted_efficiencies < threshold) else max_age
        
        # Display recommendations
        st.subheader("Maintenance Schedule")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Initial Efficiency", f"{initial_efficiency:.3f}")
        
        with col2:
            st.metric("Recommended Replacement Age", f"{replacement_age} days")
        
        with col3:
            st.metric("Efficiency at Replacement", f"{predicted_efficiencies[replacement_age]:.3f}")
        
        # Additional recommendations
        st.subheader("Key Recommendations")
        st.markdown(f"""
        1. **Regular Maintenance**: Replace {filter_type} after {replacement_age} days of use
        2. **Performance Monitoring**: Check efficiency when it reaches {replacement_age-5} days
        3. **Early Warning**: Consider replacement if efficiency drops below {initial_efficiency:.3f}
        """)
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Please make sure all required files and dependencies are available.")
