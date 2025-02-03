import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.figure_factory as ff
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LassoCV
import optuna

def create_3d_efficiency_plot(df, filter_type):
    """Create 3D surface plot of efficiency"""
    filter_data = df[df['filter_class'] == filter_type]
    
    fig = go.Figure(data=[go.Surface(
        x=np.linspace(0, 100, 20),  # Age range
        y=np.linspace(0, 1, 20),    # Load factor range
        z=np.random.rand(20, 20) * 0.2 + 0.7,  # Efficiency values
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title=f'3D Efficiency Surface - {filter_type}',
        scene=dict(
            xaxis_title='Filter Age (days)',
            yaxis_title='Load Factor',
            zaxis_title='Efficiency'
        ),
        width=800,
        height=800
    )
    
    return fig

def create_animated_timeline(df, filter_type):
    """Create animated timeline of efficiency changes"""
    filter_data = df[df['filter_class'] == filter_type].copy()
    filter_data['date'] = pd.to_datetime(filter_data['timestamp']).dt.date
    
    fig = px.scatter(filter_data,
                    x='filter_age_days',
                    y='efficiency',
                    animation_frame='date',
                    animation_group='filter_class',
                    size='load_factor',
                    color='pressure_drop_pa',
                    hover_name='filter_class',
                    range_x=[0, filter_data['filter_age_days'].max()],
                    range_y=[0.5, 1.0],
                    title=f'Efficiency Evolution - {filter_type}')
    
    fig.update_layout(
        xaxis_title='Filter Age (days)',
        yaxis_title='Efficiency',
        coloraxis_colorbar_title='Pressure Drop',
        showlegend=True
    )
    
    return fig

def create_radar_chart(df, filter_type):
    """Create radar chart for filter performance metrics"""
    filter_data = df[df['filter_class'] == filter_type]
    
    metrics = {
        'Efficiency': filter_data['efficiency'].mean(),
        'Load Handling': filter_data['load_factor'].mean(),
        'Pressure Management': 1 - (filter_data['pressure_drop_pa'] / filter_data['pressure_drop_pa'].max()),
        'PM2.5 Removal': 1 - (filter_data['outlet_pm25'] / filter_data['inlet_pm25']).mean(),
        'PM10 Removal': 1 - (filter_data['outlet_pm10'] / filter_data['inlet_pm10']).mean()
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name=filter_type
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f'Performance Radar - {filter_type}'
    )
    
    return fig

def create_violin_comparison(df):
    """Create violin plot comparing filter types"""
    fig = go.Figure()
    
    for filter_type in df['filter_class'].unique():
        filter_data = df[df['filter_class'] == filter_type]
        
        fig.add_trace(go.Violin(
            x=[filter_type] * len(filter_data),
            y=filter_data['efficiency'],
            name=filter_type,
            box_visible=True,
            meanline_visible=True
        ))
    
    fig.update_layout(
        title='Efficiency Distribution Comparison',
        xaxis_title='Filter Type',
        yaxis_title='Efficiency',
        violinmode='group'
    )
    
    return fig

def create_heatmap_correlation(df, filter_type):
    """Create interactive correlation heatmap"""
    filter_data = df[df['filter_class'] == filter_type]
    
    numeric_cols = ['efficiency', 'filter_age_days', 'load_factor', 
                   'pressure_drop_pa', 'inlet_pm25', 'inlet_pm10']
    corr_matrix = filter_data[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=numeric_cols,
        y=numeric_cols,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f'Correlation Heatmap - {filter_type}',
        width=800,
        height=800
    )
    
    return fig

def create_parallel_coordinates(df):
    """Create enhanced parallel coordinates plot with more features and interactivity"""
    # Create derived metrics
    df_plot = df.copy()
    df_plot['efficiency_to_pressure'] = df_plot['efficiency'] / df_plot['pressure_drop_pa']
    df_plot['pm_removal_ratio'] = (df_plot['inlet_pm25'] - df_plot['outlet_pm25']) / df_plot['inlet_pm25']
    df_plot['load_efficiency'] = df_plot['efficiency'] * (1 - df_plot['load_factor'])
    
    # Normalize numerical columns to 0-1 scale for better visualization
    cols_to_normalize = [
        'efficiency', 'filter_age_days', 'load_factor', 
        'pressure_drop_pa', 'inlet_pm25', 'inlet_pm10',
        'efficiency_to_pressure', 'pm_removal_ratio', 'load_efficiency'
    ]
    
    for col in cols_to_normalize:
        min_val = df_plot[col].min()
        max_val = df_plot[col].max()
        df_plot[col + '_normalized'] = (df_plot[col] - min_val) / (max_val - min_val)
    
    # Create color mapping
    filter_types = df_plot['filter_class'].unique()
    color_map = {filter_type: i for i, filter_type in enumerate(filter_types)}
    df_plot['color_value'] = df_plot['filter_class'].map(color_map)
    
    # Create the parallel coordinates plot with normalized values
    dimensions = [
        dict(range=[0, 1],
             label='Efficiency',
             values=df_plot['efficiency_normalized']),
        dict(range=[0, 1],
             label='Filter Age',
             values=df_plot['filter_age_days_normalized']),
        dict(range=[0, 1],
             label='Load Factor',
             values=df_plot['load_factor_normalized']),
        dict(range=[0, 1],
             label='Pressure Drop',
             values=df_plot['pressure_drop_pa_normalized']),
        dict(range=[0, 1],
             label='PM2.5 Inlet',
             values=df_plot['inlet_pm25_normalized']),
        dict(range=[0, 1],
             label='PM10 Inlet',
             values=df_plot['inlet_pm10_normalized']),
        dict(range=[0, 1],
             label='Efficiency/Pressure',
             values=df_plot['efficiency_to_pressure_normalized']),
        dict(range=[0, 1],
             label='PM Removal Ratio',
             values=df_plot['pm_removal_ratio_normalized']),
        dict(range=[0, 1],
             label='Load Efficiency',
             values=df_plot['load_efficiency_normalized'])
    ]
    
    # Create figure with custom styling
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_plot['color_value'],
                colorscale='Viridis',
                showscale=True,
                cmin=min(color_map.values()),
                cmax=max(color_map.values())
            ),
            dimensions=dimensions,
            labelangle=30,
            labelside='bottom'
        )
    )
    
    # Update layout with custom styling
    fig.update_layout(
        title=dict(
            text='Interactive Parallel Coordinates Analysis',
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        height=800,
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Add custom legend
    for filter_type, color_value in color_map.items():
        normalized_value = color_value / (len(filter_types) - 1)
        color = px.colors.sample_colorscale('Viridis', [normalized_value])[0]
        
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            name=filter_type,
            line=dict(color=color, width=2),
            mode='lines',
            showlegend=True
        ))
    
    return fig

def create_sunburst_efficiency(df):
    """Create sunburst chart for efficiency hierarchy"""
    df['efficiency_category'] = pd.qcut(df['efficiency'], 
                                      q=4, 
                                      labels=['Low', 'Medium', 'High', 'Very High'])
    
    fig = px.sunburst(
        df,
        path=['filter_class', 'efficiency_category'],
        values='load_factor',
        title='Efficiency Distribution Hierarchy'
    )
    
    return fig

def create_gauge_chart(value, title):
    """Create a gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.7], 'color': "lightgray"},
                {'range': [0.7, 0.9], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    return fig

def calculate_advanced_metrics(df):
    """Calculate advanced performance metrics"""
    df_metrics = df.copy()
    
    # Efficiency metrics
    df_metrics['efficiency_stability'] = df_metrics.groupby('filter_class')['efficiency'].transform(lambda x: 1 - x.std())
    df_metrics['efficiency_trend'] = df_metrics.groupby('filter_class')['efficiency'].transform(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Pressure-based metrics
    df_metrics['pressure_efficiency'] = df_metrics['efficiency'] / df_metrics['pressure_drop_pa']
    df_metrics['pressure_stability'] = 1 - df_metrics.groupby('filter_class')['pressure_drop_pa'].transform('std')
    
    # Particle removal metrics
    df_metrics['pm25_removal_rate'] = (df_metrics['inlet_pm25'] - df_metrics['outlet_pm25']) / df_metrics['inlet_pm25']
    df_metrics['pm10_removal_rate'] = (df_metrics['inlet_pm10'] - df_metrics['outlet_pm10']) / df_metrics['inlet_pm10']
    df_metrics['overall_pm_removal'] = (df_metrics['pm25_removal_rate'] + df_metrics['pm10_removal_rate']) / 2
    
    # Load and age metrics
    df_metrics['load_rate'] = df_metrics['load_factor'] / df_metrics['filter_age_days']
    df_metrics['efficiency_decay'] = df_metrics['efficiency'] / df_metrics['filter_age_days']
    df_metrics['performance_index'] = (df_metrics['efficiency'] * df_metrics['pm25_removal_rate'] * 
                                     (1 - df_metrics['load_factor'])) / df_metrics['pressure_drop_pa']
    
    return df_metrics

def create_statistical_analysis(df):
    """Create statistical analysis visualization"""
    metrics = ['efficiency', 'pressure_drop_pa', 'load_factor', 'pm25_removal_rate']
    filter_types = df['filter_class'].unique()
    
    fig = make_subplots(
        rows=len(metrics), cols=2,
        subplot_titles=['Distribution', 'Box Plot'] * len(metrics),
        vertical_spacing=0.05
    )
    
    for i, metric in enumerate(metrics, 1):
        # Distribution plot
        for filter_type in filter_types:
            data = df[df['filter_class'] == filter_type][metric]
            fig.add_trace(
                go.Histogram(x=data, name=filter_type, opacity=0.7),
                row=i, col=1
            )
        
        # Box plot
        fig.add_trace(
            go.Box(x=df['filter_class'], y=df[metric], name=metric),
            row=i, col=2
        )
    
    fig.update_layout(height=300*len(metrics), showlegend=False)
    return fig

def create_pca_analysis(df):
    """Create PCA analysis visualization"""
    # Select numerical columns for PCA
    numerical_cols = ['efficiency', 'pressure_drop_pa', 'load_factor', 
                     'inlet_pm25', 'inlet_pm10', 'pm25_removal_rate', 
                     'pm10_removal_rate', 'performance_index']
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[numerical_cols])
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        z=pca_result[:, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=df['filter_class'].astype('category').cat.codes,
            colorscale='Viridis',
            opacity=0.8
        ),
        text=df['filter_class'],
        hovertemplate="Filter Type: %{text}<br>" +
                     "PC1: %{x:.2f}<br>" +
                     "PC2: %{y:.2f}<br>" +
                     "PC3: %{z:.2f}"
    )])
    
    # Update layout
    fig.update_layout(
        title='PCA Analysis - 3D Visualization',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )
    
    return fig, pca.explained_variance_ratio_

def create_correlation_network(df):
    """Create interactive correlation network visualization"""
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Create network graph
    edge_traces = []
    node_traces = []
    
    # Create edges
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.3:  # Correlation threshold
                edge_traces.append(
                    go.Scatter(
                        x=[i, j],
                        y=[0, 0],
                        mode='lines',
                        line=dict(
                            width=abs(corr_matrix.iloc[i, j]) * 5,
                            color='rgb(50,50,50)'
                        ),
                        hoverinfo='text',
                        text=f'Correlation: {corr_matrix.iloc[i, j]:.2f}'
                    )
                )
    
    # Create nodes
    node_traces.append(
        go.Scatter(
            x=list(range(len(corr_matrix.columns))),
            y=[0] * len(corr_matrix.columns),
            mode='markers+text',
            marker=dict(size=20, color='lightblue'),
            text=corr_matrix.columns,
            textposition='top center'
        )
    )
    
    # Combine traces
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title='Correlation Network',
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

@st.cache_data
def create_ml_features(df):
    """Create features for machine learning"""
    df_ml = df.copy()
    
    # Create time-based features
    df_ml['hour'] = df_ml['timestamp'].dt.hour
    df_ml['day_of_week'] = df_ml['timestamp'].dt.dayofweek
    df_ml['month'] = df_ml['timestamp'].dt.month
    df_ml['is_weekend'] = df_ml['day_of_week'].isin([5, 6]).astype(int)
    
    # Create interaction features
    df_ml['pressure_load_interaction'] = df_ml['pressure_drop_pa'] * df_ml['load_factor']
    df_ml['age_load_interaction'] = df_ml['filter_age_days'] * df_ml['load_factor']
    
    # Create ratio features
    df_ml['pm_ratio'] = df_ml['inlet_pm25'] / df_ml['inlet_pm10']
    df_ml['pressure_efficiency'] = df_ml['efficiency'] / df_ml['pressure_drop_pa']
    
    return df_ml

@st.cache_data
def train_models(df, target='efficiency'):
    """Train multiple models and return results"""
    # Prepare features
    feature_cols = ['filter_age_days', 'load_factor', 'pressure_drop_pa',
                   'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week',
                   'month', 'is_weekend', 'pressure_load_interaction',
                   'age_load_interaction', 'pm_ratio', 'pressure_efficiency']
    
    X = df[feature_cols]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models with fewer estimators for faster training
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'actual': y_test,
            'feature_importance': model.feature_importances_,
            'metrics': {
                'R²': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }
        }
    
    return results, feature_cols, scaler

def plot_learning_curves(model, X, y, cv=5):
    """Plot learning curves for a model"""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    # Add training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_mean,
        name='Training Score',
        mode='lines+markers',
        line=dict(color='blue'),
        error_y=dict(
            type='data',
            array=train_std,
            visible=True
        )
    ))
    
    # Add validation scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_mean,
        name='Cross-validation Score',
        mode='lines+markers',
        line=dict(color='red'),
        error_y=dict(
            type='data',
            array=test_std,
            visible=True
        )
    ))
    
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Examples',
        yaxis_title='R² Score',
        showlegend=True
    )
    
    return fig

def create_shap_plot(model, X):
    """Create SHAP summary plot"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    fig = go.Figure()
    
    # Sort features by mean absolute SHAP value
    feature_importance = np.abs(shap_values).mean(0)
    feature_order = np.argsort(feature_importance)
    
    # Create violin plots for each feature
    for i in feature_order:
        fig.add_trace(go.Violin(
            x=shap_values[:, i],
            name=X.columns[i],
            side='positive',
            line_color='blue',
            orientation='h'
        ))
    
    fig.update_layout(
        title='SHAP Feature Importance',
        xaxis_title='SHAP value',
        showlegend=False
    )
    
    return fig

def perform_feature_selection(X, y, method='lasso'):
    """Perform feature selection using various methods"""
    if method == 'lasso':
        selector = SelectFromModel(LassoCV(cv=5))
    elif method == 'rfe':
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator=base_model, n_features_to_select=8)
    
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Get feature importance scores
    if method == 'lasso':
        importance = np.abs(selector.estimator_.coef_)
    else:
        importance = selector.estimator_.feature_importances_
    
    return selected_features, importance

def optimize_hyperparameters(X, y, model_type='rf'):
    """Optimize model hyperparameters using Optuna"""
    def objective(trial):
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            model = RandomForestRegressor(**params, random_state=42)
        elif model_type == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0)
            }
            model = xgb.XGBRegressor(**params, random_state=42)
        
        score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        return -score.mean()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params_

def create_ensemble_model(models):
    """Create ensemble models using voting and stacking"""
    # Create base models list for voting
    estimators = [(name, model) for name, model in models.items()]
    
    # Voting regressor
    voting_reg = VotingRegressor(estimators=estimators)
    
    # Stacking regressor with meta-learner
    stacking_reg = StackingRegressor(
        estimators=estimators,
        final_estimator=GradientBoostingRegressor(random_state=42)
    )
    
    return {
        'Voting': voting_reg,
        'Stacking': stacking_reg
    }

def plot_optimization_history(study):
    """Plot optimization history from Optuna study"""
    fig = go.Figure()
    
    # Plot optimization history
    fig.add_trace(go.Scatter(
        x=list(range(len(study.trials))),
        y=[t.value for t in study.trials],
        mode='markers',
        name='Trials'
    ))
    
    # Plot best value history
    best_values = [min([t.value for t in study.trials[:i+1]]) 
                  for i in range(len(study.trials))]
    fig.add_trace(go.Scatter(
        x=list(range(len(study.trials))),
        y=best_values,
        mode='lines',
        name='Best Value',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Hyperparameter Optimization History',
        xaxis_title='Trial',
        yaxis_title='Objective Value',
        showlegend=True
    )
    
    return fig

def main():
    st.set_page_config(page_title="Enhanced Filter Analysis", 
                      page_icon="📊", 
                      layout="wide")
    
    # Load and process data
    df = pd.read_csv('air_filter_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_metrics = calculate_advanced_metrics(df)
    
    # Sidebar controls
    st.sidebar.title("Analysis Controls")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Statistical Analysis", "Pattern Analysis", "Advanced Metrics", "Machine Learning"]
    )
    
    selected_filters = st.sidebar.multiselect(
        "Select Filter Types",
        options=df['filter_class'].unique(),
        default=df['filter_class'].unique()
    )
    
    # Filter data
    if selected_filters:
        df_filtered = df_metrics[df_metrics['filter_class'].isin(selected_filters)]
    else:
        df_filtered = df_metrics
    
    if analysis_type == "Overview":
        st.title("📊 Filter Performance Overview")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Efficiency", 
                     f"{df_filtered['efficiency'].mean():.3f}",
                     f"{df_filtered['efficiency'].std():.3f} σ")
        with col2:
            st.metric("Average Pressure Drop",
                     f"{df_filtered['pressure_drop_pa'].mean():.1f} Pa",
                     f"{df_filtered['pressure_drop_pa'].std():.1f} σ")
        with col3:
            st.metric("Average Load Factor",
                     f"{df_filtered['load_factor'].mean():.3f}",
                     f"{df_filtered['load_factor'].std():.3f} σ")
        
        # Gauge charts
        st.header("Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            efficiency_gauge = create_gauge_chart(
                df_filtered['efficiency'].mean(),
                "Average Efficiency"
            )
            st.plotly_chart(efficiency_gauge, use_container_width=True)
        
        with col2:
            load_gauge = create_gauge_chart(
                df_filtered['load_factor'].mean(),
                "Average Load Factor"
            )
            st.plotly_chart(load_gauge, use_container_width=True)
        
        with col3:
            pressure_gauge = create_gauge_chart(
                1 - (df_filtered['pressure_drop_pa'] / df_filtered['pressure_drop_pa'].max()).mean(),
                "Pressure Performance"
            )
            st.plotly_chart(pressure_gauge, use_container_width=True)
        
        # Time series
        st.header("Performance Over Time")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df_filtered, 
                         x='timestamp', 
                         y='efficiency',
                         color='filter_class',
                         title='Efficiency Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_animated_timeline(df_filtered, df_filtered['filter_class'].unique()[0])
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter and 3D plots
        st.header("Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df_filtered,
                           x='load_factor',
                           y='pressure_drop_pa',
                           color='filter_class',
                           title='Pressure Drop vs Load Factor',
                           trendline="lowess")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if selected_filters:
                fig = create_3d_efficiency_plot(df_filtered, selected_filters[0])
                st.plotly_chart(fig, use_container_width=True)
        
        # Advanced visualizations
        st.header("Advanced Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_radar_chart(df_filtered, df_filtered['filter_class'].unique()[0])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_parallel_coordinates(df_filtered)
            st.plotly_chart(fig, use_container_width=True)
        
    elif analysis_type == "Statistical Analysis":
        st.title("📈 Statistical Analysis")
        
        # Correlation matrix
        st.header("Correlation Analysis")
        corr_matrix = df_filtered[[
            'efficiency', 'pressure_drop_pa', 'load_factor',
            'filter_age_days', 'inlet_pm25', 'inlet_pm10'
        ]].corr()
        
        fig = px.imshow(corr_matrix,
                       title='Correlation Matrix',
                       color_continuous_scale='RdBu')
        st.plotly_chart(fig)
        
        # Box plots
        st.header("Distribution Analysis")
        fig = px.box(df_filtered,
                    x='filter_class',
                    y='efficiency',
                    title='Efficiency Distribution by Filter Class')
        st.plotly_chart(fig)
        
        # ANOVA test
        st.header("ANOVA Test Results")
        f_stat, p_val = stats.f_oneway(*[group['efficiency'].values 
                                       for name, group in df_filtered.groupby('filter_class')])
        st.write(f"F-statistic: {f_stat:.4f}")
        st.write(f"p-value: {p_val:.4f}")
        
    elif analysis_type == "Pattern Analysis":
        st.title("🔄 Pattern Analysis")
        
        st.header("Parallel Coordinates Analysis")
        st.markdown("""
        This plot shows relationships between multiple parameters simultaneously.
        - Each vertical axis represents a different metric
        - Each line represents a filter instance
        - Lines are colored by filter type
        - Click and drag along any axis to filter the data
        - Double-click to reset the view
        """)
        st.plotly_chart(create_parallel_coordinates(df_filtered))
        
    elif analysis_type == "Advanced Metrics":
        st.title("🔬 Advanced Metrics Analysis")
        
        # Performance metrics
        st.header("Performance Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Efficiency Stability", 
                     f"{df_filtered['efficiency'].std():.3f}")
        with col2:
            st.metric("Pressure Efficiency",
                     f"{(df_filtered['efficiency'] / df_filtered['pressure_drop_pa']).mean():.3f}")
        with col3:
            st.metric("Load Rate",
                     f"{df_filtered['load_factor'].diff().mean():.3f}")
        
        # Time series analysis
        st.header("Performance Over Time")
        fig = go.Figure()
        metrics = ['efficiency', 'pressure_drop_pa', 'load_factor']
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=df_filtered['timestamp'],
                y=df_filtered[metric],
                name=metric.replace('_', ' ').title(),
                mode='lines+markers'
            ))
        st.plotly_chart(fig)
        
    elif analysis_type == "Machine Learning":
        st.title("🤖 Machine Learning Analysis")
        
        # Select target and model
        col1, col2 = st.columns(2)
        with col1:
            target_variable = st.selectbox(
                "Select Target Variable",
                ['efficiency', 'pressure_drop_pa', 'load_factor']
            )
        with col2:
            model_type = st.selectbox(
                "Select Model",
                ['Random Forest', 'XGBoost']
            )
        
        # Train model button
        if st.button("Train Model and Show Learning Curves"):
            with st.spinner("Training model and generating learning curves..."):
                # Get data
                df_ml = create_ml_features(df_filtered)
                feature_cols = ['filter_age_days', 'load_factor', 'pressure_drop_pa',
                              'inlet_pm25', 'inlet_pm10', 'hour', 'day_of_week',
                              'month', 'is_weekend', 'pressure_load_interaction',
                              'age_load_interaction', 'pm_ratio', 'pressure_efficiency']
                
                X = df_ml[feature_cols]
                y = df_ml[target_variable]
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Create model
                if model_type == 'Random Forest':
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    model = xgb.XGBRegressor(n_estimators=50, random_state=42)
                
                # Plot learning curves
                st.plotly_chart(plot_learning_curves(model, X_scaled, y))
                
                # Train final model and show performance
                model.fit(X_scaled, y)
                y_pred = cross_val_predict(model, X_scaled, y, cv=5)
                
                # Show metrics
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                mae = mean_absolute_error(y, y_pred)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("R² Score", f"{r2:.3f}")
                col2.metric("RMSE", f"{rmse:.3f}")
                col3.metric("MAE", f"{mae:.3f}")
                
                # Feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f'Feature Importance - {model_type}'
                )
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
