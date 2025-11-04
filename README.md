# 🌿 EcoGuard ML — AI-Powered Ecological Monitoring System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io)
[![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-orange)](https://tensorflow.org)

## 📋 Overview

EcoGuard ML is an advanced ecological monitoring system powered by machine learning. It replaces traditional formula-based models with data-driven algorithms to predict wildlife population changes, assess conservation risks, and provide actionable insights for conservation professionals and researchers.

### 🎯 Project Vision

This project bridges the gap between traditional ecological modeling and modern AI capabilities. By leveraging machine learning on ecological datasets, EcoGuard ML provides:

- **Data-Driven Predictions**: Learn patterns from historical data rather than relying solely on theoretical models
- **Scalable Monitoring**: Handle large-scale ecological datasets across multiple species and locations
- **Actionable Intelligence**: Generate specific, prioritized recommendations for conservation interventions
- **Explainable AI**: Understand why models make certain predictions through feature importance analysis
- **Interactive Exploration**: Test "what-if" scenarios to evaluate conservation strategies before implementation

### 🌍 Real-World Applications

- **Wildlife Conservation**: Monitor endangered species populations and predict extinction risks
- **Habitat Management**: Assess the impact of environmental changes on ecosystem health
- **Invasive Species Control**: Predict spread patterns and evaluate control strategies
- **Protected Area Planning**: Identify high-risk locations requiring immediate attention
- **Policy Decision Support**: Provide evidence-based recommendations for conservation funding
- **Climate Change Analysis**: Model population responses to changing environmental conditions

### ✨ Key Features

- 🤖 **AI-Powered Predictions** - Multiple ML models including Random Forest, XGBoost, LightGBM, and LSTM
- 📊 **Interactive Dashboard** - Real-time visualizations and analytics
- 🎯 **Risk Assessment** - Automated conservation risk classification
- 📈 **Time Series Forecasting** - 12-month population projections
- 💡 **Actionable Insights** - Hotspot detection and management recommendations
- 🔍 **Explainable AI** - Feature importance and decision transparency
- 🎲 **What-If Scenarios** - Test different conservation strategies

## Machine Learning Models

### Population Prediction Models

- Random Forest Regressor — Ensemble method for accurate population predictions
- XGBoost — Gradient boosting for enhanced accuracy
- LightGBM — Fast gradient boosting with high performance
- LSTM Neural Networks — Deep learning for time series forecasting

### Classification Models

- Risk Assessment Classifier — Automatically categorize conservation risk levels
- Species Status Predictor — ML-based conservation status evaluation

### Advanced Features

- Feature Importance Analysis — Understand what factors drive predictions
- Time Series Forecasting — Multi-step ahead population projections
- Automated Model Selection — Best performing model chosen automatically
- Real-time Predictions — Instant AI predictions for any scenario

## 📊 Dataset

The system includes a comprehensive synthetic ecological dataset designed to simulate real-world conservation scenarios. The data generator creates realistic patterns including:

### Generated Data Components

- **Species Population Data** — Historical population counts over time with realistic growth/decline patterns
- **Environmental Factors** — Temperature, rainfall, habitat quality, and human disturbance levels
- **Species Interactions** — Predator-prey relationships, competition dynamics, and symbiotic effects
- **Conservation Actions** — Management interventions (habitat restoration, hunting restrictions, etc.) and their outcomes
- **Spatial Information** — Location-based data for geographic analysis and hotspot detection
- **Temporal Trends** — Multi-year time series with seasonal variations and long-term trends

### Data Characteristics

- **Realistic Correlations**: Environmental factors influence population changes
- **Stochastic Variations**: Natural randomness in population dynamics
- **Multiple Species**: Support for diverse species with different life histories
- **Configurable Scale**: Generate datasets from months to decades
- **Risk Categories**: Automatic labeling of conservation status (Low, Medium, High, Critical)

The synthetic data allows users to experiment with ML models without requiring real ecological datasets, while still learning patterns applicable to actual conservation work.

## Installation and Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12 recommended
- Git (for cloning the repository)

### 1) Clone the repository

```bash
git clone https://github.com/Chronos778/EcoGuard-ML.git
cd EcoGuard-ML
```

### 2) Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by policy:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) (Optional) Pre-train models

You can optionally pre-train models before running the app:

```bash
python models/ml_predictor.py
```

This creates a `trained_models` folder. You can also train models directly in the app interface.

### 5) Run the application

```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

To run on a specific port:

```bash
streamlit run app.py --server.port 8504
```

Then open `http://localhost:8504` in your browser.

Stop the server with `Ctrl + C`.

### Verify Installation

Run the setup check script to verify all dependencies:

```bash
python setup_check.py
```

### Troubleshooting

**Port already in use:**
```bash
streamlit run app.py --server.port 8505
```

**XGBoost/LightGBM installation errors:**
- These packages are optional for advanced features
- The app will work with scikit-learn models only
- Try installing them separately: `pip install xgboost lightgbm`

**Import or module errors:**
- Ensure virtual environment is activated
- Verify you're in the project root directory
- Reinstall dependencies: `pip install -r requirements.txt`

**No training data available:**
- Click "Load/Generate Training Data" button in the app
- Synthetic ecological data will be generated automatically
- Pre-generate data: `python data/data_generator.py`

**TensorFlow/GPU issues:**
- TensorFlow is optional for LSTM models
- The app works with scikit-learn models if TensorFlow isn't available
- For GPU support, install CUDA-enabled TensorFlow separately

## Key Features

### 1. Data Overview Dashboard

- Comprehensive dataset statistics
- Species distribution analysis
- Population trend visualizations
- Risk level distributions

### 2. ML Model Training

- Train multiple ML models simultaneously
- Automatic model comparison and selection
- Performance metrics and validation
- Model persistence and loading

### 3. AI Predictions Interface

- Interactive parameter selection
- Real-time population predictions
- Risk level classification
- 12-month population projections

### 4. Model Insights and Explainability

- Feature importance analysis
- Factor impact explanations
- Model decision transparency
- Performance comparisons

### 5. Advanced Analytics

- Time series decomposition
- Residual analysis
- Learning curves
- Prediction accuracy metrics

## Machine Learning Workflow

### 1. Data Generation

```python
from data.data_generator import DataGenerator
generator = DataGenerator()
datasets = generator.create_complete_dataset(years=5)
data = datasets['population']
```

### 2. Model Training

```python
from models.ml_predictor import EcoMLPredictor
predictor = EcoMLPredictor()

# Train models
pop_model, score = predictor.train_population_model(data)
risk_model, accuracy = predictor.train_risk_model(data)
```

### 3. Making Predictions

```python
# Predict population
population_pred = predictor.predict_population(current_data)

# Predict risk level
risk_pred, risk_probs = predictor.predict_risk(current_data)
```

## Model Performance

Expected performance metrics:

- Population Prediction R²: 0.85–0.95
- Risk Classification Accuracy: 0.88–0.95
- LSTM Time Series Loss: < 0.1 MSE

## User Interface

### Navigation Menu

- Home — Introduction and quick start
- Data Overview — Dataset exploration and statistics
- Train ML Models — Model training interface
- AI Predictions — Interactive prediction tool
- Actionable Insights — Hotspots, declines, recommended actions, and what‑if scenarios
- Mathematical Models — Exponential, Logistic, and Lotka–Volterra simulations
- Model Insights — Feature importance and explanations
- Performance Dashboard — Model comparison and metrics

### Actionable Insights

- Risk Hotspots: combines current risk level and recent population trend into a priority score
- Significant Declines: highlights species/locations with sharp negative trends
- Recommended Actions: heuristic recommendations with estimated cost/effect and ROI; export CSV
- What‑if Scenario Simulator: adjust `human_activity`, `habitat_quality`, and `hunting_pressure` to see predicted population delta and risk probability changes

Note: the Scenario Simulator requires models trained in the current session (train on “Train ML Models” page, or pre-train via `models/ml_predictor.py`).

### Mathematical Models

- Exponential Growth: dN/dt = rN
- Logistic Growth: dN/dt = rN(1 − N/K)
- Lotka–Volterra Predator–Prey dynamics

These classic models are provided for intuition-building alongside the ML system.

## Technical Architecture

```text
EcoGuard-ML/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── models/
│   ├── ml_predictor.py       # ML model implementations
│   ├── scenario_engine.py    # What‑if scenario simulation using trained ML
│   └── math_models.py        # Exponential/Logistic/Lotka–Volterra simulations
├── data/
│   ├── data_generator.py     # Synthetic data generation
│   └── *.csv                 # Generated datasets
├── utils/
│   └── insights.py           # Hotspots, declines, and action recommendations
└── trained_models/           # Saved ML models
   ├── *.pkl                 # Scikit-learn models
   └── *.h5 / *.keras        # TensorFlow/Keras models (if used)
```

## AI vs Traditional Models

| Aspect | Traditional Models | EcoGuard ML |
|--------|--------------------|-------------|
| Approach | Mathematical equations | Data-driven learning |
| Adaptability | Fixed parameters | Learns from data |
| Accuracy | Theory-based | Evidence-based |
| Complexity | Limited interactions | Complex patterns |
| Updates | Manual adjustment | Automatic retraining |
| Predictions | Formula-based | Pattern recognition |

## 🔬 Technical Highlights

### Advanced ML Techniques

**Feature Engineering**
- Population momentum calculations (rate of change analysis)
- Environmental stress indices (composite environmental factors)
- Habitat density metrics (carrying capacity utilization)
- Capacity utilization ratios (population vs. habitat limits)
- Interaction terms (species competition and predation effects)
- Temporal features (seasonality, trends, cyclical patterns)

**Model Ensemble**
- Multiple algorithms trained and compared automatically
- Best model automatically selected based on validation performance
- Cross-validation for robust performance estimates
- Hyperparameter optimization using Optuna (optional)
- Model stacking and weighted averaging support
- Automatic fallback to simpler models if advanced packages unavailable

**Explainable AI**
- Feature importance rankings (understand what drives predictions)
- SHAP values for prediction explanations (optional, if installed)
- Model transparency through interpretable features
- Decision factors analysis for risk classifications
- Visual explanations of model decisions

### Performance Optimization

- **Efficient Data Processing**: Vectorized operations with NumPy/Pandas
- **Model Caching**: Save/load trained models to avoid retraining
- **GPU Acceleration**: Optional GPU support for XGBoost, LightGBM, and TensorFlow
- **Lazy Loading**: Load models only when needed
- **Batch Predictions**: Process multiple scenarios efficiently

### Extensibility

- **Modular Architecture**: Easy to add new models or features
- **Plugin System**: Custom data generators and predictors
- **Configuration Files**: Adjust model parameters without code changes
- **API-Ready**: Models can be exposed via REST API for integration
- **Export Options**: Save predictions and recommendations as CSV/JSON

## Quick Start Guide

1. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

2. **Generate training data:** Click "Load/Generate Training Data" button
3. **Train ML models:** Navigate to "Train ML Models" page
4. **Make predictions:** Go to "AI Predictions" page
5. **Analyze results:** Explore "Model Insights" and "Actionable Insights"

## 🛠️ Technology Stack

### Core Dependencies

- **streamlit** (>= 1.28.0) — Interactive web application framework
- **pandas** (>= 2.0.0) — Data manipulation and analysis
- **numpy** (>= 1.24.0) — Numerical computing and array operations
- **scikit-learn** (>= 1.3.0) — Machine learning library (Random Forest, preprocessing)
- **plotly** (>= 5.15.0) — Interactive visualizations and charts
- **matplotlib** (>= 3.7.0) — Static plotting and figure generation
- **seaborn** (>= 0.12.0) — Statistical data visualization
- **joblib** (>= 1.3.0) — Model serialization and persistence

### Optional Advanced Features

- **tensorflow** (>= 2.13.0) — Deep learning framework for LSTM models
- **xgboost** (>= 1.7.0) — Gradient boosting framework
- **lightgbm** (>= 4.0.0) — Fast gradient boosting with high performance
- **optuna** (>= 3.3.0) — Hyperparameter optimization
- **shap** (>= 0.42.0) — Model explainability and interpretability

### Development Tools

- **Python 3.10+** — Modern Python with type hints and performance improvements
- **Git** — Version control
- **Virtual Environment** — Isolated dependency management

All dependencies are specified in `requirements.txt` for easy installation.

## 💡 Best Practices

### For Conservation Professionals

1. **Start with Data Exploration**: Review the Data Overview dashboard to understand patterns
2. **Train Multiple Models**: Compare different algorithms to find the best fit for your data
3. **Validate Predictions**: Cross-reference ML predictions with field observations
4. **Use What-If Scenarios**: Test conservation strategies before implementation
5. **Export Results**: Download recommendations as CSV for reporting and planning

### For Researchers

1. **Customize Data Generation**: Modify `data_generator.py` to match your study system
2. **Experiment with Features**: Add domain-specific features to improve predictions
3. **Tune Hyperparameters**: Use the optional Optuna integration for optimization
4. **Document Experiments**: Track model versions and performance metrics
5. **Contribute Improvements**: Share enhancements via pull requests

### For Developers

1. **Follow PEP 8**: Code style guidelines for Python
2. **Use Type Hints**: Improve code clarity and catch errors early
3. **Test Changes**: Verify models train and predict correctly after modifications
4. **Document Functions**: Add docstrings to new functions and classes
5. **Check Dependencies**: Ensure compatibility with specified package versions

## 🔍 How It Works

### Step-by-Step Workflow

1. **Data Collection/Generation**
   - Real data: Import CSV files with population, environmental, and interaction data
   - Synthetic data: Use built-in generator to create realistic ecological datasets

2. **Feature Engineering**
   - Automatic calculation of derived features (momentum, stress indices, etc.)
   - Handling of missing data and outliers
   - Normalization and scaling for ML models

3. **Model Training**
   - Multiple algorithms trained in parallel
   - Automatic hyperparameter tuning (optional)
   - Cross-validation for performance estimation
   - Best model selection based on validation metrics

4. **Prediction & Analysis**
   - Population forecasting for individual species
   - Risk classification (Low/Medium/High/Critical)
   - Feature importance analysis
   - Uncertainty quantification

5. **Actionable Insights**
   - Hotspot identification (high-risk areas)
   - Decline detection (species needing attention)
   - Conservation recommendations with cost-benefit analysis
   - What-if scenario testing

6. **Reporting & Export**
   - Interactive visualizations
   - Downloadable CSV reports
   - Model performance metrics
   - Explanation of predictions

## 📈 Performance Metrics

### Model Accuracy

Expected performance on synthetic datasets:
- **Population Prediction R²**: 0.85–0.95 (explains 85-95% of variance)
- **Risk Classification Accuracy**: 0.88–0.95 (88-95% correct classifications)
- **LSTM Time Series MSE**: < 0.1 (low prediction error)

### Computational Performance

- **Data Generation**: ~1-5 seconds for 1000 records
- **Model Training**: ~10-30 seconds for basic models, ~2-5 minutes with optimization
- **Predictions**: < 1 second for single predictions, ~5-10 seconds for batch predictions
- **Dashboard Loading**: ~2-3 seconds initial load

Performance may vary based on dataset size and hardware specifications.

## Support


## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by real-world ecological monitoring needs
- Built with modern ML and web technologies
- Designed for conservation professionals and researchers

## 📚 Additional Resources

- [Project Structure Documentation](PROJECT_STRUCTURE.md)
- [Setup Verification Tool](setup_check.py)
- [Demo Scripts](demos/)

## 📧 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check existing issues and documentation
- Review the troubleshooting section

---

**EcoGuard ML** — Bringing AI to Wildlife Conservation 🌿

---
