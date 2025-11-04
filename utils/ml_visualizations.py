import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
SHAP_AVAILABLE = False
try:
    import importlib.util as importlib_util
    SHAP_AVAILABLE = importlib_util.find_spec("shap") is not None
except Exception:
    SHAP_AVAILABLE = False

class MLVisualizationUtils:
    """
    Advanced visualization utilities for ML model analysis
    """
    
    def __init__(self):
        pass
    
    def plot_confusion_matrix(self, y_true, y_pred, labels):
        """
        Create interactive confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig = px.imshow(cm, 
                       x=labels, 
                       y=labels,
                       color_continuous_scale='Blues',
                       title="Risk Classification Confusion Matrix")
        
        fig.update_layout(
            xaxis_title="Predicted Risk Level",
            yaxis_title="Actual Risk Level"
        )
        
        return fig
    
    def plot_feature_correlations(self, data, features):
        """
        Create correlation heatmap of features
        """
        corr_matrix = data[features].corr()
        
        fig = px.imshow(corr_matrix,
                       x=features,
                       y=features,
                       color_continuous_scale='RdBu',
                       title="Feature Correlation Matrix")
        
        fig.update_layout(height=600)
        return fig
    
    def plot_population_distribution(self, data):
        """
        Plot population distributions by species and risk level
        """
        fig = px.box(data, 
                    x='species_name', 
                    y='population_count',
                    color='risk_level',
                    title="Population Distribution by Species and Risk Level")
        
        fig.update_layout(
            xaxis_title="Species",
            yaxis_title="Population Count",
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_time_series_decomposition(self, data, species_name):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        species_data = data[data['species_name'] == species_name].copy()
        species_data['date'] = pd.to_datetime(species_data['date'])
        species_data = species_data.sort_values('date')
        
        # Simple moving averages for trend
        species_data['trend'] = species_data['population_count'].rolling(window=12, center=True).mean()
        species_data['seasonal'] = species_data['population_count'] - species_data['trend']
        species_data['residual'] = species_data['population_count'] - species_data['trend'] - species_data['seasonal']
        
        fig = make_subplots(rows=4, cols=1,
                           subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                           vertical_spacing=0.08)
        
        # Original data
        fig.add_trace(go.Scatter(x=species_data['date'], 
                               y=species_data['population_count'],
                               name='Population'), row=1, col=1)
        
        # Trend
        fig.add_trace(go.Scatter(x=species_data['date'], 
                               y=species_data['trend'],
                               name='Trend'), row=2, col=1)
        
        # Seasonal
        fig.add_trace(go.Scatter(x=species_data['date'], 
                               y=species_data['seasonal'],
                               name='Seasonal'), row=3, col=1)
        
        # Residual
        fig.add_trace(go.Scatter(x=species_data['date'], 
                               y=species_data['residual'],
                               name='Residual'), row=4, col=1)
        
        fig.update_layout(height=800, 
                         title=f"Time Series Decomposition: {species_name}",
                         showlegend=False)
        
        return fig
    
    def plot_prediction_accuracy(self, actual, predicted, species_names):
        """
        Scatter plot of actual vs predicted values
        """
        fig = go.Figure()
        
        for i, species in enumerate(species_names):
            species_mask = np.array([True] * len(actual))  # Simplified for demo
            
            fig.add_trace(go.Scatter(
                x=actual[species_mask],
                y=predicted[species_mask],
                mode='markers',
                name=species,
                opacity=0.7
            ))
        
        # Add perfect prediction line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Actual vs Predicted Population",
            xaxis_title="Actual Population",
            yaxis_title="Predicted Population",
            height=500
        )
        
        return fig
    
    def plot_residuals_analysis(self, residuals, predictions):
        """
        Residual analysis plots
        """
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Residuals vs Fitted', 'Residuals Distribution'))
        
        # Residuals vs Fitted
        fig.add_trace(go.Scatter(x=predictions, y=residuals,
                               mode='markers', name='Residuals'), 
                     row=1, col=1)
        
        # Add horizontal line at y=0
        fig.add_shape(type="line", x0=min(predictions), x1=max(predictions), 
                     y0=0, y1=0, line=dict(color="red", dash="dash"),
                     row=1, col=1)
        
        # Residuals histogram
        fig.add_trace(go.Histogram(x=residuals, name='Distribution'), 
                     row=1, col=2)
        
        fig.update_layout(height=400, title="Residual Analysis")
        return fig
    
    def plot_learning_curves(self, train_scores, val_scores, train_sizes):
        """
        Plot learning curves to assess model performance
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_scores.mean(axis=1),
            mode='lines+markers',
            name='Training Score',
            error_y=dict(array=train_scores.std(axis=1))
        ))
        
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_scores.mean(axis=1),
            mode='lines+markers',
            name='Validation Score',
            error_y=dict(array=val_scores.std(axis=1))
        ))
        
        fig.update_layout(
            title="Learning Curves",
            xaxis_title="Training Set Size",
            yaxis_title="Model Score",
            height=400
        )
        
        return fig
    
    def plot_feature_distribution(self, data, feature, by_risk=True):
        """
        Plot distribution of a specific feature
        """
        if by_risk:
            fig = px.histogram(data, 
                             x=feature, 
                             color='risk_level',
                             title=f"Distribution of {feature} by Risk Level",
                             marginal='rug')
        else:
            fig = px.histogram(data, 
                             x=feature,
                             title=f"Distribution of {feature}")
        
        return fig
    
    def plot_species_comparison_radar(self, data):
        """
        Radar chart comparing species characteristics
        """
        # Aggregate data by species
        species_stats = data.groupby('species_name').agg({
            'population_count': 'mean',
            'temperature': 'mean',
            'rainfall': 'mean',
            'human_disturbance': 'mean',
            'habitat_quality': 'mean' if 'habitat_quality' in data.columns else lambda x: 0.8
        }).reset_index()
        
        # Normalize to 0-1 scale
        numeric_cols = ['population_count', 'temperature', 'rainfall', 'human_disturbance', 'habitat_quality']
        for col in numeric_cols:
            if col in species_stats.columns:
                species_stats[f'{col}_norm'] = (species_stats[col] - species_stats[col].min()) / (species_stats[col].max() - species_stats[col].min())
        
        fig = go.Figure()
        
        categories = ['Population', 'Temperature', 'Rainfall', 'Human Impact', 'Habitat Quality']
        
        for _, row in species_stats.iterrows():
            values = [
                row.get('population_count_norm', 0),
                row.get('temperature_norm', 0),
                row.get('rainfall_norm', 0),
                row.get('human_disturbance_norm', 0),
                row.get('habitat_quality_norm', 0.8)
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=row['species_name']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Species Characteristics Comparison"
        )
        
        return fig
    
    def create_prediction_dashboard(self, data, predictions, actual):
        """
        Create comprehensive prediction dashboard
        """
        # Calculate metrics
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual) ** 2))
        r2 = 1 - (np.sum((actual - predictions) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
        
        # Create dashboard layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        with col2:
            st.metric("Root Mean Square Error", f"{rmse:.2f}")
        with col3:
            st.metric("R² Score", f"{r2:.3f}")
        
        # Prediction accuracy plot
        accuracy_fig = self.plot_prediction_accuracy(actual, predictions, 
                                                   data['species_name'].unique())
        st.plotly_chart(accuracy_fig, use_container_width=True)
        
        # Residuals analysis
        residuals = actual - predictions
        residual_fig = self.plot_residuals_analysis(residuals, predictions)
        st.plotly_chart(residual_fig, use_container_width=True)
        
        return mae, rmse, r2

def create_shap_explanation(model, X_sample):
    """
    Create SHAP explanations for model predictions
    Note: This requires SHAP to be installed separately
    """
    if not SHAP_AVAILABLE:
        return None
        
    try:
        import importlib
        shap = importlib.import_module("shap")  # type: ignore
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)
        
        # Create waterfall plot
        fig = shap.plots.waterfall(shap_values[0], show=False)
        return fig
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None
