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
    """Simple and clean ML predictor for ecological data"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        print("🌿 EcoGuard ML Predictor Initialized")
        if XGBOOST_AVAILABLE:
            print("✅ XGBoost available")
        if LIGHTGBM_AVAILABLE:
            print("✅ LightGBM available")
    
    def generate_sample_data(self, n_samples=1000):
        """Generate sample ecological data for testing"""
        print(f"📊 Generating {n_samples} sample ecological records...")
        
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
        print(f"✅ Generated data: {df.shape[0]} records, {df.shape[1]} columns")
        return df
    
    def prepare_features(self, data):
        """Prepare features for ML models"""
        df = data.copy()
        
        # Encode categorical variables
        if 'season' not in self.encoders:
            self.encoders['season'] = LabelEncoder()
            self.encoders['season'].fit(['Spring', 'Summer', 'Fall', 'Winter'])
        
        df['season_encoded'] = self.encoders['season'].transform(df['season'])
        
        # Create derived features
        df['population_change'] = (df['population_t1'] - df['population_t2']) / df['population_t2']
        df['habitat_density'] = df['population_t1'] / df['habitat_area']
        df['capacity_utilization'] = df['population_t1'] / df['carrying_capacity']
        df['temp_deviation'] = abs(df['temperature'] - 15)
        
        # Select feature columns
        self.feature_columns = [
            'species_id', 'temperature', 'rainfall', 'season_encoded', 
            'habitat_area', 'human_disturbance', 'population_t1', 'population_t2',
            'population_change', 'habitat_density', 'capacity_utilization', 
            'temp_deviation', 'carrying_capacity'
        ]
        
        return df[self.feature_columns]
    
    def train_population_model(self, data):
        """Train population prediction model"""
        print("🤖 Training population prediction models...")
        
        X = self.prepare_features(data)
        y = data['current_population']
        
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
            models_to_try['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models_to_try['LightGBM'] = lgb.LGBMRegressor(
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
            'use_scaling': best_name == 'LinearRegression'
        }
        
        print(f"✅ Best model: {best_name} (R² = {best_score:.4f})")
        return best_model, best_score
    
    def train_risk_model(self, data):
        """Train risk classification model"""
        print("⚠️  Training risk classification model...")
        
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
            'encoder': self.encoders['risk']
        }
        
        print(f"✅ Risk model accuracy: {accuracy:.4f}")
        return model, accuracy
    
    def predict_population(self, data):
        """Predict population for new data"""
        if 'population' not in self.models:
            raise ValueError("Population model not trained yet!")
        
        model_info = self.models['population']
        model = model_info['model']
        use_scaling = model_info['use_scaling']
        
        # Prepare features
        X = self.prepare_features(data)
        
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
        
        # Prepare features
        X = self.prepare_features(data)
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
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
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
        
        print(f"💾 Models saved to {directory}")
    
    def load_models(self, directory):
        """Load trained models"""
        try:
            self.models = joblib.load(os.path.join(directory, 'models.pkl'))
            self.scalers = joblib.load(os.path.join(directory, 'scalers.pkl'))
            self.encoders = joblib.load(os.path.join(directory, 'encoders.pkl'))
            self.feature_columns = joblib.load(os.path.join(directory, 'features.pkl'))
            
            print(f"📂 Models loaded from {directory}")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False

if __name__ == "__main__":
    # Test the ML predictor
    predictor = EcoMLPredictor()
    data = predictor.generate_sample_data(n_samples=1000)
    predictor.train_population_model(data)
    predictor.train_risk_model(data)
    print("✅ EcoGuard ML Predictor ready!")
