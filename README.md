# 🌿 EcoGuard ML — AI-Powered Ecological Monitoring System

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io)
[![Made with TensorFlow](https://img.shields.io/badge/Made%20with-TensorFlow-orange)](https://tensorflow.org)

## 📋 Overview

EcoGuard ML is an ecological monitoring system powered by machine learning. It replaces traditional formula-based models with data-driven algorithms to predict wildlife population changes, assess conservation risks, and provide actionable insights.

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

## Dataset

The system includes a comprehensive synthetic ecological dataset with:

- Species population data — Historical population counts over time
- Environmental factors — Temperature, rainfall, habitat quality, human disturbance
- Species interactions — Predator-prey relationships and competition dynamics
- Conservation actions — Management interventions and their outcomes

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

## Advanced Features Overview

### Feature Engineering

- Population momentum calculations
- Environmental stress indices
- Habitat density metrics
- Capacity utilization ratios

### Model Ensemble

- Multiple algorithms trained and compared
- Best model automatically selected
- Cross-validation for robust performance
- Hyperparameter optimization

### Explainable AI

- Feature importance rankings
- Prediction explanations
- Model transparency
- Decision factors analysis

## Quick Start Guide

1. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

2. **Generate training data:** Click "Load/Generate Training Data" button
3. **Train ML models:** Navigate to "Train ML Models" page
4. **Make predictions:** Go to "AI Predictions" page
5. **Analyze results:** Explore "Model Insights" and "Actionable Insights"

## Dependencies

- streamlit — Web application framework
- pandas — Data manipulation and analysis
- numpy — Numerical computing
- scikit-learn — Machine learning library
- tensorflow — Deep learning framework (optional)
- plotly — Interactive visualizations
- xgboost — Gradient boosting framework
- lightgbm — Fast gradient boosting
- matplotlib/seaborn — Statistical plotting

## Support

For questions or issues:

1. Check the application's built-in help text
2. Review model performance metrics
3. Examine feature importance explanations
4. Use the prediction dashboard for insights

## Screenshots

_This section removed — add screenshots after deployment in `screenshots/` if desired._

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
