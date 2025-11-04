"""
EcoGuard ML - Simple and Clean Machine Learning Population Predictor
A streamlined ML system for ecological population prediction
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib
import os

# Optional GPU-accelerated libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class EcoMLPredictor:
    """Simple and clean ML predictor for ecological data.

    Supports two dataset schemas:
    - Synthetic schema from generate_sample_data(): uses 'current_population' as target.
    - DataGenerator population dataset: uses 'population_count' as target with environmental features.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        print("EcoGuard ML Predictor Initialized")
        if XGBOOST_AVAILABLE:
            print("XGBoost available")
        if LIGHTGBM_AVAILABLE:
            print("LightGBM available")
    
    def generate_sample_data(self, n_samples=1000):
        """Generate sample ecological data for testing"""
        print(f"Generating {n_samples} sample ecological records...")
        
        np.random.seed(42)
        data = []
        
        species_names = ['Red Fox', 'Gray Wolf', 'White-tailed Deer', 'Brown Bear', 'Mountain Lion']
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        
        for i in range(n_samples):
            # Environmental factors
            temperature = np.random.normal(15, 8)
            rainfall = np.random.exponential(50)
            season = np.random.choice(seasons)
            habitat_area = np.random.normal(1000, 200)
            human_disturbance = np.random.uniform(0, 1)
            
            # Species info
            species_id = np.random.randint(0, len(species_names))
            species_name = species_names[species_id]
            
            # Population dynamics
            carrying_capacity = max(100, habitat_area * np.random.uniform(0.8, 1.2))
            base_pop = np.random.uniform(50, carrying_capacity * 0.8)
            
            # Previous populations
            pop_t2 = max(10, base_pop + np.random.normal(0, base_pop * 0.1))
            pop_t1 = max(10, pop_t2 + np.random.normal(0, pop_t2 * 0.15))
            
            # Current population (target variable)
            env_factor = 1 - (human_disturbance * 0.3)
            temp_factor = 1 - abs(temperature - 20) / 40
            rain_factor = min(1, rainfall / 100)
            
            current_pop = max(5, pop_t1 * env_factor * temp_factor * rain_factor + 
                            np.random.normal(0, pop_t1 * 0.1))
            
            # Risk assessment
            pop_trend = (current_pop - pop_t2) / pop_t2 if pop_t2 > 0 else 0
            if current_pop < carrying_capacity * 0.2 or pop_trend < -0.3:
                risk_level = 'High'
            elif current_pop < carrying_capacity * 0.5 or pop_trend < -0.1:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            data.append({
                'species_id': species_id,
                'species_name': species_name,
                'temperature': round(temperature, 2),
                'rainfall': round(max(0, rainfall), 2),
                'season': season,
                'habitat_area': round(max(50, habitat_area), 2),
                'human_disturbance': round(human_disturbance, 3),
                'population_t2': int(pop_t2),
                'population_t1': int(pop_t1),
                'current_population': int(current_pop),
                'carrying_capacity': int(carrying_capacity),
                'risk_level': risk_level
            })
        
        df = pd.DataFrame(data)
        print(f"Generated data: {df.shape[0]} records, {df.shape[1]} columns")
        return df
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models from either supported dataset schema."""
        df = data.copy()

        # Ensure season encoder exists
        if 'season' in df.columns:
            if 'season' not in self.encoders:
                self.encoders['season'] = LabelEncoder()
                self.encoders['season'].fit(['Spring', 'Summer', 'Fall', 'Winter'])
            df['season_encoded'] = self.encoders['season'].transform(df['season'])

        # Path A: DataGenerator population dataset
        if 'population_count' in df.columns:
            # Map species_name to id for stability
            if 'species_name' in df.columns:
                if 'species' not in self.encoders:
                    self.encoders['species'] = LabelEncoder()
                    self.encoders['species'].fit(df['species_name'].unique())
                df['species_id'] = self.encoders['species'].transform(df['species_name'])
            else:
                df['species_id'] = 0

            # Derived features
            if 'temperature' in df.columns:
                df['temp_deviation'] = abs(df['temperature'] - 15)
            if 'base_population' in df.columns and 'population_count' in df.columns:
                df['capacity_utilization'] = df['population_count'] / np.maximum(1, df['base_population'])
            if 'habitat_quality' in df.columns and 'population_count' in df.columns:
                df['habitat_density'] = df['population_count'] / np.maximum(0.1, df.get('habitat_quality', 1.0))

            # Select columns available in the population dataset
            cols = []
            candidates = [
                'species_id', 'temperature', 'rainfall', 'humidity', 'season_encoded', 'month',
                'human_activity', 'hunting_pressure', 'habitat_quality', 'base_population',
                'population_change_rate', 'temp_deviation', 'capacity_utilization', 'habitat_density'
            ]
            for c in candidates:
                if c in df.columns:
                    cols.append(c)
            self.feature_columns = cols
            return df[cols]

        # Path B: Synthetic schema from generate_sample_data()
        # Create derived features if source columns exist
        if {'population_t1', 'population_t2'}.issubset(df.columns):
            df['population_change'] = (df['population_t1'] - df['population_t2']) / df['population_t2']
            df['habitat_density'] = df['population_t1'] / np.maximum(1.0, df.get('habitat_area', 1.0))
            if 'carrying_capacity' in df.columns:
                df['capacity_utilization'] = df['population_t1'] / np.maximum(1.0, df['carrying_capacity'])
        if 'temperature' in df.columns:
            df['temp_deviation'] = abs(df['temperature'] - 15)

        # Select feature columns
        columns_pref = [
            'species_id', 'temperature', 'rainfall', 'season_encoded', 
            'habitat_area', 'human_disturbance', 'population_t1', 'population_t2',
            'population_change', 'habitat_density', 'capacity_utilization', 
            'temp_deviation', 'carrying_capacity'
        ]
        self.feature_columns = [c for c in columns_pref if c in df.columns]
        return df[self.feature_columns]
    
    def train_population_model(self, data):
        """Train population prediction model"""
        print("Training population prediction models...")
        
        X = self.prepare_features(data)
        # Choose target dynamically
        if 'current_population' in data.columns:
            y = data['current_population']
        elif 'population_count' in data.columns:
            y = data['population_count']
        else:
            raise ValueError("No target column found: expected 'current_population' or 'population_count'")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scalers['population'] = StandardScaler()
        X_train_scaled = self.scalers['population'].fit_transform(X_train)
        X_test_scaled = self.scalers['population'].transform(X_test)
        
        # Try different models
        models_to_try = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'LinearRegression': LinearRegression()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_to_try['XGBoost'] = xgb.XGBRegressor(  # type: ignore[name-defined]
                n_estimators=100, random_state=42, n_jobs=-1
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models_to_try['LightGBM'] = lgb.LGBMRegressor(  # type: ignore[name-defined]
                n_estimators=100, random_state=42, verbose=-1, n_jobs=-1
            )
        
        best_model = None
        best_score = -np.inf
        best_name = ""
        
        for name, model in models_to_try.items():
            # Train model
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate score
            score = r2_score(y_test, y_pred)
            
            if score > best_score:
                best_model = model
                best_score = score
                best_name = name
        
        # Save best model
        self.models['population'] = {
            'model': best_model,
            'name': best_name,
            'score': best_score,
            'use_scaling': best_name == 'LinearRegression',
            'features': list(self.feature_columns)
        }
        print(f"Best model: {best_name} (R² = {best_score:.4f})")
        return best_model, best_score
    
    def train_risk_model(self, data):
        """Train risk classification model"""
        print("Training risk classification model...")
        
        X = self.prepare_features(data)
        y = data['risk_level']
        
        # Encode risk levels
        if 'risk' not in self.encoders:
            self.encoders['risk'] = LabelEncoder()
        
        y_encoded = self.encoders['risk'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Scale features
        if 'risk' not in self.scalers:
            self.scalers['risk'] = StandardScaler()
        
        X_train_scaled = self.scalers['risk'].fit_transform(X_train)
        X_test_scaled = self.scalers['risk'].transform(X_test)
        
        # Train Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        self.models['risk'] = {
            'model': model,
            'accuracy': accuracy,
            'encoder': self.encoders['risk'],
            'features': list(self.feature_columns)
        }
        print(f"Risk model accuracy: {accuracy:.4f}")
        return model, accuracy
    
    def predict_population(self, data):
        """Predict population for new data"""
        if 'population' not in self.models:
            raise ValueError("Population model not trained yet!")
        
        model_info = self.models['population']
        model = model_info['model']
        use_scaling = model_info['use_scaling']
        feat_cols = model_info.get('features', self.feature_columns)
        
        # Prepare features
        X = self.prepare_features(data)
        X = X.reindex(columns=feat_cols, fill_value=0)
        
        # Scale if needed
        if use_scaling:
            X_scaled = self.scalers['population'].transform(X)
            predictions = model.predict(X_scaled)
        else:
            predictions = model.predict(X)
        
        return predictions
    
    def predict_risk(self, data):
        """Predict risk levels for new data"""
        if 'risk' not in self.models:
            raise ValueError("Risk model not trained yet!")
        
        model_info = self.models['risk']
        model = model_info['model']
        encoder = model_info['encoder']
        feat_cols = model_info.get('features', self.feature_columns)
        
        # Prepare features
        X = self.prepare_features(data)
        X = X.reindex(columns=feat_cols, fill_value=0)
        X_scaled = self.scalers['risk'].transform(X)
        
        # Predict
        predictions_encoded = model.predict(X_scaled)
        predictions = encoder.inverse_transform(predictions_encoded)
        probabilities = model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def get_feature_importance(self):
        """Get feature importance from population model"""
        if 'population' not in self.models:
            return None
        
        model_info = self.models['population']
        model = model_info['model']
        
        if hasattr(model, 'feature_importances_'):
            # Ensure lengths match
            feat_cols = model_info.get('features', self.feature_columns)
            importances = model.feature_importances_
            
            if len(feat_cols) != len(importances):
                print(f"Warning: Feature count mismatch ({len(feat_cols)} vs {len(importances)})")
                return None
            
            importance_df = pd.DataFrame({
                'feature': feat_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None
    
    def save_models(self, directory):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        joblib.dump(self.models, os.path.join(directory, 'models.pkl'))
        joblib.dump(self.scalers, os.path.join(directory, 'scalers.pkl'))
        joblib.dump(self.encoders, os.path.join(directory, 'encoders.pkl'))
        joblib.dump(self.feature_columns, os.path.join(directory, 'features.pkl'))
        print(f"Models saved to {directory}")
    
    def load_models(self, directory):
        """Load trained models"""
        try:
            self.models = joblib.load(os.path.join(directory, 'models.pkl'))
            self.scalers = joblib.load(os.path.join(directory, 'scalers.pkl'))
            self.encoders = joblib.load(os.path.join(directory, 'encoders.pkl'))
            self.feature_columns = joblib.load(os.path.join(directory, 'features.pkl'))
            
            print(f"Models loaded from {directory}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

if __name__ == "__main__":
    # Test the ML predictor
    predictor = EcoMLPredictor()
    # Prefer using DataGenerator dataset when available
    try:
        from data.data_generator import DataGenerator
        gen = DataGenerator()
        datasets = gen.create_complete_dataset(years=2, save_to_csv=False)
        df = datasets['population']
    except Exception:
        df = predictor.generate_sample_data(n_samples=1000)
    predictor.train_population_model(df)
    predictor.train_risk_model(df)
    print("EcoGuard ML Predictor ready!")