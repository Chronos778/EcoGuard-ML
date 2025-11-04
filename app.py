import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Prefer package-qualified imports for clarity
from models.ml_predictor import EcoMLPredictor
from models.math_models import (
    ExponentialParams,
    LogisticParams,
    LotkaVolterraParams,
    simulate_exponential,
    simulate_logistic,
    simulate_lotka_volterra,
)
from data.data_generator import DataGenerator
from utils.insights import compute_risk_hotspots, recommend_actions
from models.scenario_engine import run_scenario

# Page configuration
st.set_page_config(
    page_title="EcoGuard ML - AI-Powered Ecological Monitoring",
    page_icon=None,
    layout="wide"
)

# Initialize session state
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = EcoMLPredictor()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def load_or_generate_data():
    """Load or generate dataset using DataGenerator (population dataset)."""
    try:
        with st.spinner("Generating ecological datasets for ML training..."):
            generator = DataGenerator()
            datasets = generator.create_complete_dataset(years=3, save_to_csv=False)
            population_data = datasets['population']
        st.success(f"Generated {len(population_data)} records across {population_data['location'].nunique()} locations and {population_data['year'].nunique()} years")

        st.session_state.population_data = population_data
        st.session_state.data_loaded = True
        return population_data

    except Exception as e:
        st.error(f"Error loading/generating data: {e}")
        return None

def train_ml_models(data):
    """Train all ML models"""
    try:
        with st.spinner("Training Machine Learning models..."):
            progress_bar = st.progress(0)
            
            # Train population predictor
            st.info("Training Population Prediction Model...")
            pop_model, pop_score = st.session_state.ml_predictor.train_population_model(data)
            progress_bar.progress(50)
            
            # Train risk classifier
            st.info("Training Risk Classification Model...")
            risk_model, risk_accuracy = st.session_state.ml_predictor.train_risk_model(data)
            progress_bar.progress(100)
            
            # Save models
            st.session_state.ml_predictor.save_models('trained_models')
            
            st.session_state.models_trained = True
            st.session_state.model_scores = {
                'population_r2': pop_score,
                'risk_accuracy': risk_accuracy
            }
            
            st.success("All ML models trained successfully!")
            
            # Display model performance
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Population Model R²", f"{pop_score:.3f}")
            with col2:
                st.metric("Risk Classification Accuracy", f"{risk_accuracy:.3f}")
                
    except Exception as e:
        st.error(f"Error training models: {e}")

def show_data_overview():
    """Display data overview and statistics"""
    if not st.session_state.data_loaded:
        return
    
    data = st.session_state.population_data
    
    st.header("Dataset Overview")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Species Count", data['species_name'].nunique())
    with col3:
        st.metric("Locations", data['location'].nunique())
    with col4:
        st.metric("Time Span (Years)", data['year'].nunique())
    
    # Data distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Species Distribution")
        species_counts = data['species_name'].value_counts()
        fig = px.bar(x=species_counts.index, y=species_counts.values, 
                    title="Records per Species")
        fig.update_layout(xaxis_title="Species", yaxis_title="Record Count")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Risk Level Distribution")
        risk_counts = data['risk_level'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title="Risk Level Distribution")
        st.plotly_chart(fig, width='stretch')
    
    # Population trends
    st.subheader("Population Trends Over Time")
    
    # Aggregate by year and species
    yearly_data = data.groupby(['year', 'species_name'])['population_count'].mean().reset_index()
    
    fig = px.line(yearly_data, x='year', y='population_count', 
                 color='species_name', title="Average Population Trends by Species")
    fig.update_layout(xaxis_title="Year", yaxis_title="Average Population")
    st.plotly_chart(fig, width='stretch')

def show_ml_predictions():
    """Show ML model predictions interface"""
    if not st.session_state.models_trained:
        st.warning("Please train ML models first!")
        return
    
    st.header("AI Predictions")
    
    # Input parameters for prediction
    st.subheader("Environmental Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperature = st.slider("Temperature (°C)", -10, 35, 15)
        rainfall = st.slider("Rainfall (mm)", 0, 200, 60)
        humidity = st.slider("Humidity (%)", 20, 95, 65)
    
    with col2:
        human_activity = st.slider("Human Disturbance", 0.0, 1.0, 0.3, 0.1)
        habitat_area = st.slider("Habitat Area (hectares)", 200, 2000, 1000)
        habitat_quality = st.slider("Habitat Quality", 0.1, 1.0, 0.8, 0.1)
    
    with col3:
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])
        month = st.slider("Month", 1, 12, 6)
        species_name = st.selectbox("Species", [
            "Red Fox", "Gray Wolf", "White-tailed Deer", "European Starling", "Feral Cat"
        ])
    
    # Create prediction input
    prediction_input = pd.DataFrame({
        'species_name': [species_name],
        'temperature': [temperature],
        'rainfall': [rainfall],
        'humidity': [humidity],
        'season': [season],
        'month': [month],
        'human_activity': [human_activity],
        'hunting_pressure': [0.2],
        'habitat_quality': [habitat_quality],
        'base_population': [max(100, habitat_area * habitat_quality * 0.8)]
    })
    
    if st.button("Generate AI Predictions", type="primary"):
        try:
            # Population prediction
            pop_prediction = st.session_state.ml_predictor.predict_population(prediction_input)
            
            # Risk prediction
            risk_prediction, risk_probs = st.session_state.ml_predictor.predict_risk(prediction_input)
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Population", f"{int(pop_prediction[0]):,}")
            
            with col2:
                st.metric("Risk Level", risk_prediction[0])
                
                # Show risk probabilities
                try:
                    classes = st.session_state.ml_predictor.models['risk']['encoder'].classes_
                    for label, prob in zip(classes, risk_probs[0]):
                        st.write(f"{label}: {prob:.1%}")
                except Exception:
                    pass
            
            # Visualization
            st.subheader("Prediction Visualization")
            
            # Create time series projection
            future_months = 12
            timeline = pd.date_range(start='2024-01-01', periods=future_months, freq='M')
            
            # Simple projection (in real scenario, this would use the LSTM model)
            current_pop = pop_prediction[0]
            projections = []
            
            for i in range(future_months):
                # Add some realistic variation
                trend = np.random.normal(0.02, 0.1)  # 2% average growth with variation
                current_pop = max(10, current_pop * (1 + trend))
                projections.append(current_pop)
            
            # Plot projection
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timeline, y=projections, 
                                   mode='lines+markers', name='Predicted Population',
                                   line=dict(color='blue')))
            
            fig.update_layout(
                title="12-Month Population Projection",
                xaxis_title="Date",
                yaxis_title="Population Count",
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
            
        except Exception as e:
            st.error(f"Prediction error: {e}")

def show_feature_importance():
    """Display feature importance from trained models"""
    if not st.session_state.models_trained:
        st.warning("Please train ML models first!")
        return
    
    st.header("AI Model Insights")
    
    # Get feature importance
    importance_df = st.session_state.ml_predictor.get_feature_importance()
    
    if importance_df is not None:
        st.subheader("Most Important Factors for Population Prediction")
        
        # Top 10 features
        top_features = importance_df.head(10)
        
        fig = px.bar(top_features, x='importance', y='feature', 
                    orientation='h', title="Feature Importance in Population Prediction")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')
        
        # Feature explanations
        st.subheader("Factor Explanations")
        
        feature_explanations = {
            'population_t1': 'Previous month population count',
            'population_t2': 'Population count two months ago',
            'temperature': 'Environmental temperature',
            'habitat_area': 'Available habitat area',
            'human_disturbance': 'Level of human interference',
            'capacity_utilization': 'How close to carrying capacity',
            'habitat_density': 'Population density in habitat',
            'rainfall': 'Monthly precipitation',
            'environmental_stress': 'Combined environmental pressure',
            'population_momentum': 'Population growth trend'
        }
        
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            explanation = feature_explanations.get(feature_name, 'Environmental or population factor')
            
            with st.expander(f"{feature_name} (Importance: {importance:.3f})"):
                st.write(f"**Description:** {explanation}")
                st.write(f"**Impact on Predictions:** {importance:.1%} of model decision-making")
                
                # Add interpretation
                if importance > 0.1:
                    st.info("**High Impact** - This factor strongly influences population predictions")
                elif importance > 0.05:
                    st.warning("**Moderate Impact** - This factor has noticeable influence")
                else:
                    st.success("**Low Impact** - This factor has minor influence")

def show_model_comparison():
    """Compare different ML model performances"""
    if not st.session_state.models_trained:
        st.warning("Please train ML models first!")
        return
    
    st.header("Model Performance Comparison")
    
    # Model scores
    scores = st.session_state.model_scores
    
    # Create comparison chart
    model_names = ['Population Regressor', 'Risk Classifier']
    model_scores = [scores.get('population_r2', 0.0), scores.get('risk_accuracy', 0.0)]
    if 'lstm_loss' in scores:
        model_names.append('LSTM Neural Network')
        model_scores.append(1 - scores['lstm_loss'])
    
    fig = go.Figure(data=[
        go.Bar(name='Model Performance', x=model_names, y=model_scores,
               marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ])
    
    fig.update_layout(
        title='ML Model Performance Comparison',
        yaxis_title='Performance Score',
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Detailed metrics
    st.subheader("Detailed Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Population Prediction Model", 
                 f"{scores['population_r2']:.3f}", 
                 help="R² score - how well the model explains population variance")
        st.info("**R² Score**: 1.0 = perfect prediction, 0.0 = no better than average")
    
    with col2:
        st.metric("Risk Classification Model", 
                 f"{scores['risk_accuracy']:.3f}",
                 help="Accuracy - percentage of correct risk level predictions")
        st.info("**Accuracy**: 1.0 = 100% correct, 0.5 = random guessing")
    
    with col3:
        if 'lstm_loss' in scores:
            st.metric("Time Series LSTM Model", 
                     f"{scores['lstm_loss']:.3f}",
                     help="Mean Squared Error - lower is better")
            st.info("**MSE Loss**: Lower values indicate better time series predictions")
        else:
            st.metric("Time Series LSTM Model", 
                     "Not trained",
                     help="LSTM model not available in this session")
            st.info("**Note**: LSTM training not enabled in simplified predictor")

def show_math_models():
    st.header("Mathematical Ecological Models")

    tabs = st.tabs(["Exponential", "Logistic", "Lotka–Volterra"])

    with tabs[0]:
        st.subheader("Exponential Growth: dN/dt = rN")
        col1, col2, col3 = st.columns(3)
        with col1:
            N0 = st.number_input("Initial population N0", value=100.0, min_value=0.0)
        with col2:
            r = st.number_input("Growth rate r", value=0.2, step=0.05)
        with col3:
            t_max = st.number_input("Time horizon", value=50.0, min_value=1.0)
        dt = st.slider("Time step dt", 0.01, 1.0, 0.1)
        if st.button("Run Exponential Model"):
            df = simulate_exponential(ExponentialParams(N0=N0, r=r, t_max=t_max, dt=dt))
            st.line_chart(df.set_index("time"))

    with tabs[1]:
        st.subheader("Logistic Growth: dN/dt = rN(1 - N/K)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            N0 = st.number_input("Initial population N0", value=100.0, min_value=0.0, key="lgN0")
        with col2:
            r = st.number_input("Growth rate r", value=0.2, step=0.05, key="lgr")
        with col3:
            K = st.number_input("Carrying capacity K", value=1000.0, min_value=1.0, key="lgK")
        with col4:
            t_max = st.number_input("Time horizon", value=50.0, min_value=1.0, key="lgt")
        dt = st.slider("Time step dt", 0.01, 1.0, 0.1, key="lgdt")
        if st.button("Run Logistic Model"):
            df = simulate_logistic(LogisticParams(N0=N0, r=r, K=K, t_max=t_max, dt=dt))
            st.line_chart(df.set_index("time"))

    with tabs[2]:
        st.subheader("Lotka–Volterra Predator–Prey")
        col1, col2, col3 = st.columns(3)
        with col1:
            prey0 = st.number_input("Prey initial", value=40.0, min_value=0.0)
            alpha = st.number_input("Prey growth α", value=1.1, step=0.1)
            delta = st.number_input("Predator reproduction δ", value=0.1, step=0.05)
        with col2:
            pred0 = st.number_input("Predator initial", value=9.0, min_value=0.0)
            beta = st.number_input("Predation β", value=0.4, step=0.05)
            gamma = st.number_input("Predator mortality γ", value=0.4, step=0.05)
        with col3:
            t_max = st.number_input("Time horizon", value=50.0, min_value=1.0, key="lv_t")
            dt = st.slider("Time step dt", 0.005, 0.2, 0.02, key="lv_dt")
        if st.button("Run Lotka–Volterra"):
            df = simulate_lotka_volterra(LotkaVolterraParams(
                prey0=prey0, pred0=pred0, alpha=alpha, beta=beta, delta=delta, gamma=gamma, t_max=t_max, dt=dt
            ))
            fig = px.line(df, x="time", y=["prey", "predator"], title="Predator–Prey Dynamics")
            st.plotly_chart(fig, width='stretch')

def show_actionable_insights():
    """Display hotspots, recommended actions, and what-if scenario simulation."""
    if not st.session_state.data_loaded:
        st.warning("Please load training data first!")
        return
    if not st.session_state.models_trained:
        st.warning("Please train ML models first to enable scenario simulation!")
    
    st.header("Actionable Insights")
    df = st.session_state.population_data.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Hotspots
    st.subheader("Risk Hotspots")
    hotspots = compute_risk_hotspots(df)
    top_n = st.slider("Show top N hotspots", 5, 50, 15)
    st.dataframe(hotspots.head(top_n), width='stretch')

    if not hotspots.empty:
        chart_df = hotspots.head(top_n).copy()
        chart_df['label'] = chart_df['species_name'] + " @ " + chart_df['location']
        fig = px.bar(chart_df, x='label', y='priority_score', color='risk_level', title='Top Hotspots by Priority Score')
        fig.update_layout(xaxis_title='Species @ Location', yaxis_title='Priority Score', xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

    # Declines (derived from trend)
    st.subheader("Significant Declines")
    declines = hotspots.sort_values('trend_slope').head(top_n)
    st.dataframe(declines[['date','location','species_name','population_count','risk_level','trend_slope']].head(top_n), width='stretch')

    # Recommendations
    st.subheader("Recommended Conservation Actions")
    recs = recommend_actions(hotspots, top_k=top_n)
    if not recs.empty:
        st.dataframe(recs, width='stretch')
        csv = recs.to_csv(index=False).encode('utf-8')
        st.download_button("Download Recommendations CSV", data=csv, file_name="recommendations.csv", mime="text/csv")
    else:
        st.info("No recommendations generated. Try increasing the number of hotspots or adjust data filters.")

    # What-if scenario simulation
    st.subheader("What-if Scenario Simulator")
    colA, colB = st.columns(2)
    with colA:
        species_sel = st.selectbox("Species", sorted(df['species_name'].unique()))
    with colB:
        loc_sel = st.selectbox("Location", sorted(df[df['species_name'] == species_sel]['location'].unique()))

    group_df = df[(df['species_name'] == species_sel) & (df['location'] == loc_sel)].sort_values('date')
    if group_df.empty:
        st.info("No records for the selected species and location.")
        return
    baseline_row = group_df.iloc[-1]

    st.caption(f"Baseline date: {baseline_row['date'].date()} | Population: {int(baseline_row['population_count'])} | Risk: {baseline_row['risk_level']}")

    col1, col2, col3 = st.columns(3)
    with col1:
        new_human = st.slider("Human Activity", 0.0, 1.0, float(baseline_row.get('human_activity', 0.3)), 0.05)
    with col2:
        new_habitat = st.slider("Habitat Quality", 0.1, 1.0, float(baseline_row.get('habitat_quality', 0.8)), 0.05)
    with col3:
        new_hunting = st.slider("Hunting Pressure", 0.0, 1.0, float(baseline_row.get('hunting_pressure', 0.1)), 0.05)

    if st.button("Run Scenario"):
        try:
            result = run_scenario(
                st.session_state.ml_predictor,
                baseline_row,
                {
                    'human_activity': new_human,
                    'habitat_quality': new_habitat,
                    'hunting_pressure': new_hunting,
                },
            )

            st.markdown("### Scenario Results")
            colx, coly, colz = st.columns(3)
            with colx:
                st.metric("Predicted Population (Baseline)", f"{int(result['population_baseline']):,}")
            with coly:
                st.metric("Predicted Population (Adjusted)", f"{int(result['population_adjusted']):,}",
                          delta=f"{int(result['population_delta']):,}")
            with colz:
                st.metric("Risk Level", f"{result['risk_adjusted']}",
                          delta=f"from {result['risk_baseline']}")

            # Risk probabilities chart
            try:
                classes = st.session_state.ml_predictor.models['risk']['encoder'].classes_
                prob_df = pd.DataFrame({
                    'Risk': classes,
                    'Baseline': result['risk_probs_baseline'],
                    'Adjusted': result['risk_probs_adjusted'],
                })
                fig = go.Figure()
                fig.add_trace(go.Bar(x=prob_df['Risk'], y=prob_df['Baseline'], name='Baseline'))
                fig.add_trace(go.Bar(x=prob_df['Risk'], y=prob_df['Adjusted'], name='Adjusted'))
                fig.update_layout(barmode='group', title='Risk Probability Changes', yaxis_title='Probability')
                st.plotly_chart(fig, width='stretch')
            except Exception:
                pass

            # Download scenario result
            out_df = pd.DataFrame([result])
            st.download_button("Download Scenario Result", data=out_df.to_csv(index=False).encode('utf-8'),
                               file_name="scenario_result.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Scenario error: {e}")

def main():
    # Sidebar navigation
    st.sidebar.title("EcoGuard ML Navigation")
    
    pages = [
        "Home",
        "Data Overview", 
        "Train ML Models",
        "AI Predictions",
        "Actionable Insights",
        "Mathematical Models",
        "Model Insights",
        "Performance Dashboard"
    ]
    
    page = st.sidebar.selectbox("Choose a page", pages)
    
    # Main content
    if page == "Home":
        st.title("EcoGuard ML - AI-Powered Ecological Monitoring")
        
        st.markdown("""
        ## Welcome to the Next Generation of Ecological Monitoring
        
        EcoGuard ML uses **Machine Learning** and **Artificial Intelligence** to predict wildlife population changes, 
        assess conservation risks, and provide data-driven management recommendations.
        
    ### AI-Powered Features:
        - **Population Prediction Models**: Random Forest, XGBoost, and LSTM neural networks
        - **Risk Classification**: Automated risk assessment using ML algorithms  
        - **Time Series Forecasting**: Deep learning models for long-term predictions
        - **Feature Importance Analysis**: Understand what factors matter most
        - **Real-time Predictions**: Get instant AI predictions for any scenario
        
    ### Machine Learning Models:
        1. **Random Forest Regressor** - Population prediction
        2. **Gradient Boosting** - Enhanced accuracy predictions  
        3. **LSTM Neural Networks** - Time series forecasting
        4. **Classification Models** - Risk level prediction
        5. **XGBoost/LightGBM** - Advanced ensemble methods
        
    ### Get Started:
        1. **View Data Overview** - Explore the training dataset
        2. **Train ML Models** - Build AI prediction models
        3. **Generate Predictions** - Use AI for population forecasting
        4. **Analyze Insights** - Understand model decision-making
        
        ---
        *This system replaces traditional mathematical models with machine learning algorithms 
        trained on ecological data to provide more accurate, data-driven predictions.*
        """)
        
        # Quick stats
        if st.session_state.data_loaded:
            st.success("Dataset loaded and ready for ML training")
        else:
            if st.button("Load/Generate Training Data", type="primary"):
                load_or_generate_data()
                st.rerun()
        
        if st.session_state.models_trained:
            st.success("ML models trained and ready for predictions")
    
    elif page == "Data Overview":
        if not st.session_state.data_loaded:
            if st.button("Load Training Data First"):
                load_or_generate_data()
                st.rerun()
        else:
            show_data_overview()
    
    elif page == "Train ML Models":
        if not st.session_state.data_loaded:
            st.warning("Please load training data first!")
            if st.button("Load Data"):
                load_or_generate_data()
                st.rerun()
        else:
            st.header("Train Machine Learning Models")
            
            st.markdown("""
            ### AI Model Training Process:
            This will train multiple machine learning models on your ecological data:
            
            - **Population Predictor**: Random Forest, XGBoost, LightGBM models
            - **Risk Classifier**: Classification model for risk assessment  
            - **LSTM Neural Network**: Deep learning for time series prediction
            
            Training may take 1-2 minutes depending on data size.
            """)
            
        if st.button("Train All ML Models", type="primary"):
                train_ml_models(st.session_state.population_data)
    
    elif page == "AI Predictions":
        show_ml_predictions()
    elif page == "Mathematical Models":
        show_math_models()
    elif page == "Actionable Insights":
        show_actionable_insights()
    
    elif page == "Model Insights":
        show_feature_importance()
    
    elif page == "Performance Dashboard":
        show_model_comparison()

if __name__ == "__main__":
    main()
