# 🌿 EcoGuard ML Project Structure

## Overview
EcoGuard ML is organized as a modular Streamlit application with separate components for data generation, machine learning, mathematical modeling, and visualization.

## Directory Structure

```
EcoGuard-ML/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── setup_check.py                  # Installation verification script
├── LICENSE                         # MIT License
├── README.md                       # Main documentation
├── CONTRIBUTING.md                 # Contribution guidelines
├── .gitignore                      # Git ignore rules
│
├── data/                           # Data generation and storage
│   ├── data_generator.py          # Synthetic ecological data generator
│   ├── ecological_population_data.csv
│   ├── environmental_data.csv
│   ├── species_interactions.csv
│   └── conservation_actions.csv
│
├── models/                         # Machine learning models
│   ├── ml_predictor.py            # Core ML predictor (scikit-learn, XGBoost, etc.)
│   ├── ml_predictor_gpu.py        # GPU-accelerated version
│   ├── ml_predictor_clean.py      # Simplified ML implementation
│   ├── math_models.py             # Mathematical ecological models
│   └── scenario_engine.py         # What-if scenario simulator
│
├── utils/                          # Utility functions
│   ├── insights.py                # Risk hotspots and recommendations
│   └── ml_visualizations.py       # Plotting and visualization helpers
│
├── trained_models/                 # Saved ML models (created at runtime)
│   ├── *.pkl                      # Scikit-learn models
│   ├── *.h5 / *.keras             # TensorFlow/Keras models
│   └── metadata.json              # Model metadata
│
└── demos/                          # Demo and setup scripts
    ├── demo_gpu.py                # GPU acceleration demo
    ├── gpu_setup.py               # GPU configuration helper
    └── test_app.py                # Basic app testing script
```

## Core Components

### 1. Main Application (`app.py`)
- Streamlit interface with multiple pages
- Navigation: Home, Data Overview, Train Models, Predictions, Insights, etc.
- Session state management
- Real-time interactive visualizations

### 2. Data Generation (`data/data_generator.py`)
- Creates synthetic ecological datasets
- Generates population trends, environmental factors, species interactions
- Configurable parameters (years, species count, etc.)
- CSV export functionality

### 3. ML Models (`models/ml_predictor.py`)
- **Population Predictors**: Random Forest, XGBoost, LightGBM, LSTM
- **Risk Classifiers**: Multi-class conservation risk assessment
- **Feature Engineering**: Automated feature creation
- **Model Persistence**: Save/load trained models
- **Hyperparameter Tuning**: Optuna-based optimization

### 4. Mathematical Models (`models/math_models.py`)
- Exponential growth: dN/dt = rN
- Logistic growth: dN/dt = rN(1 - N/K)
- Lotka-Volterra predator-prey dynamics
- Useful for educational comparison with ML models

### 5. Insights Engine (`utils/insights.py`)
- Risk hotspot detection
- Population decline alerts
- Conservation action recommendations
- Cost-benefit analysis

### 6. Scenario Engine (`models/scenario_engine.py`)
- What-if analysis using trained models
- Test impact of changing environmental factors
- Compare baseline vs intervention scenarios

## Data Flow

```
User Input
    ↓
Data Generator → Synthetic Dataset
    ↓
ML Predictor → Trained Models
    ↓
Prediction Engine → Population/Risk Predictions
    ↓
Insights Engine → Actionable Recommendations
    ↓
Streamlit UI → Interactive Visualizations
```

## Key Features by File

### app.py
- Multi-page navigation
- Data loading and display
- Model training interface
- Prediction forms
- Visualization dashboards

### data_generator.py
- `create_complete_dataset()` - Generate all datasets
- `generate_population_data()` - Population time series
- `generate_environmental_data()` - Weather, habitat, disturbance
- `generate_species_interactions()` - Predator-prey, competition

### ml_predictor.py
- `train_population_model()` - Train regression models
- `train_risk_model()` - Train classification models
- `predict_population()` - Make population predictions
- `predict_risk()` - Assess conservation risk
- `get_feature_importance()` - Explain predictions

### math_models.py
- `simulate_exponential()` - Exponential growth model
- `simulate_logistic()` - Logistic growth model
- `simulate_lotka_volterra()` - Predator-prey dynamics

### insights.py
- `compute_risk_hotspots()` - Identify high-risk locations
- `recommend_actions()` - Suggest conservation interventions
- `detect_population_declines()` - Alert on significant drops

## Configuration

### Dependencies (requirements.txt)
- **Core**: streamlit, pandas, numpy, scikit-learn
- **Visualization**: plotly, matplotlib, seaborn
- **ML**: xgboost, lightgbm, tensorflow (optional)
- **Optimization**: optuna (optional)
- **Explainability**: shap (optional)

### Model Files (trained_models/)
- Created automatically when training models
- Can be pre-trained using: `python models/ml_predictor.py`
- Includes metadata for model versioning

## Running Different Components

### Full Application
```bash
streamlit run app.py
```

### Data Generation Only
```bash
python data/data_generator.py
```

### Pre-train Models
```bash
python models/ml_predictor.py
```

### GPU Demo
```bash
python demo_gpu.py
```

### Setup Verification
```bash
python setup_check.py
```

## Customization

### Adding New Models
1. Add model class to `models/ml_predictor.py`
2. Register in training function
3. Update UI in `app.py`

### Adding New Features
1. Update data generator with new columns
2. Add feature engineering in predictor
3. Update visualizations as needed

### New Pages
1. Create page section in `app.py`
2. Add navigation menu item
3. Implement page logic and UI

## Development Tips

- Use virtual environment for isolated dependencies
- Test changes with small datasets first
- Check console for Streamlit logs
- Use `st.cache_data` for expensive operations
- Save models frequently during development

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style
- Testing procedures
- Pull request process
- Issue reporting
