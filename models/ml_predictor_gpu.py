import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class MLPopulationPredictor:
    """
    Machine Learning-based population prediction system for ecological monitoring
    with GPU acceleration support
    """
    
    def __init__(self, use_gpu=True):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.cv_metrics = {}
        self.use_gpu = use_gpu
        
        # Configure GPU settings
        self._configure_gpu()
        
        # Reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        try:
            import cupy as cp
            self.cupy_available = True
            print("✅ CuPy available for GPU-accelerated NumPy operations")
        except ImportError:
            self.cupy_available = False
            print("ℹ️  CuPy not available - using CPU for NumPy operations")
        except Exception:
            self.cupy_available = False
    
    def _configure_gpu(self):
        """Configure GPU settings for TensorFlow and other libraries"""
        if self.use_gpu:
            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to avoid allocating all GPU memory
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    print(f"✅ Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
                    print("✅ TensorFlow GPU acceleration enabled")
                    
                    # Set GPU as default device
                    with tf.device('/GPU:0'):
                        # Test GPU functionality
                        test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                        result = tf.reduce_sum(test_tensor)
                        print(f"✅ GPU test successful: {result.numpy()}")
                        
                except RuntimeError as e:
                    print(f"⚠️  GPU setup error: {e}")
                    self.use_gpu = False
            else:
                print("ℹ️  No GPU found - using CPU")
                self.use_gpu = False
        else:
            print("ℹ️  GPU disabled by user - using CPU")

    class _EnsembleRegressor:
        """Simple averaging ensemble for regressors."""
        def __init__(self, models):
            self.models = models
        
        def predict(self, X):
            preds = [model.predict(X) for model in self.models]
            return np.mean(np.column_stack(preds), axis=1)
        
    def generate_synthetic_data(self, n_samples=2000, n_species=5):
        """
        Generate synthetic ecological data for training ML models with GPU acceleration where possible
        """
        np.random.seed(42)
        
        data = []
        species_names = ['Red Fox', 'Gray Wolf', 'White-tailed Deer', 'Brown Bear', 'Mountain Lion']
        
        # Use CuPy for GPU-accelerated random number generation if available
        if self.cupy_available:
            try:
                import cupy as cp
                cp.random.seed(42)
                
                # Generate data on GPU
                temperatures = cp.random.normal(15, 8, n_samples)
                rainfalls = cp.random.exponential(50, n_samples)
                habitat_areas = cp.random.normal(1000, 200, n_samples)
                human_disturbances = cp.random.uniform(0, 1, n_samples)
                
                # Transfer back to CPU for DataFrame creation
                temperatures = cp.asnumpy(temperatures)
                rainfalls = cp.asnumpy(rainfalls)
                habitat_areas = cp.asnumpy(habitat_areas)
                human_disturbances = cp.asnumpy(human_disturbances)
                
                print("✅ Using GPU-accelerated data generation")
                
            except Exception as e:
                print(f"⚠️  GPU data generation failed, using CPU: {e}")
                self.cupy_available = False
        
        for i in range(n_samples):
            if not self.cupy_available:
                temperature = np.random.normal(15, 8)  # Celsius
                rainfall = np.random.exponential(50)   # mm
                habitat_area = np.random.normal(1000, 200)  # hectares
                human_disturbance = np.random.uniform(0, 1)
            else:
                temperature = temperatures[i]
                rainfall = rainfalls[i] 
                habitat_area = habitat_areas[i]
                human_disturbance = human_disturbances[i]
            
            season = np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'])
            species_id = np.random.randint(0, n_species)
            species_name = species_names[species_id]
            month = np.random.randint(1, 13)
            year = np.random.randint(2020, 2025)
            
            # Calculate carrying capacity based on habitat
            carrying_capacity = max(100, habitat_area * np.random.uniform(0.8, 1.2))
            
            # Generate population dynamics with some ecological realism
            base_pop = np.random.uniform(50, carrying_capacity * 0.8)
            pop_t2 = max(10, base_pop + np.random.normal(0, base_pop * 0.1))
            pop_t1 = max(10, pop_t2 + np.random.normal(0, pop_t2 * 0.15))
            
            # Current population influenced by environmental factors
            env_factor = 1 - (human_disturbance * 0.3)
            temp_factor = 1 - abs(temperature - 20) / 40  # Optimal around 20C
            rain_factor = min(1, rainfall / 100)  # Too little rain is bad
            
            current_pop = max(5, pop_t1 * env_factor * temp_factor * rain_factor + 
                            np.random.normal(0, pop_t1 * 0.1))
            
            # Risk level based on population trend and environmental stress
            pop_trend = (current_pop - pop_t2) / pop_t2
            if current_pop < carrying_capacity * 0.2 or pop_trend < -0.3:
                risk_level = 'High'
            elif current_pop < carrying_capacity * 0.5 or pop_trend < -0.1:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            data.append({
                'species_id': species_id,
                'species_name': species_name,
                'temperature': temperature,
                'rainfall': max(0, rainfall),
                'season': season,
                'month': month,
                'year': year,
                'habitat_area': max(50, habitat_area),
                'human_disturbance': human_disturbance,
                'population_t2': int(pop_t2),
                'population_t1': int(pop_t1),
                'current_population': int(current_pop),
                'carrying_capacity': int(carrying_capacity),
                'risk_level': risk_level
            })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, data):
        """
        Feature engineering for ML models
        """
        df = data.copy()
        
        # Encode categorical variables
        if 'season' not in self.label_encoders:
            self.label_encoders['season'] = LabelEncoder()
            self.label_encoders['season'].fit(['Spring', 'Summer', 'Fall', 'Winter'])
        
        df['season_encoded'] = self.label_encoders['season'].transform(df['season'])
        
        # Create derived features
        df['population_change_t1'] = (df['population_t1'] - df['population_t2']) / df['population_t2']
        df['population_momentum'] = df['population_t1'] / df['population_t2']
        df['habitat_density'] = df['population_t1'] / df['habitat_area']
        df['capacity_utilization'] = df['population_t1'] / df['carrying_capacity']
        df['environmental_stress'] = df['human_disturbance'] * (1 - df['habitat_area'] / 1500)
        df['temp_deviation'] = abs(df['temperature'] - 15)  # Deviation from optimal
        df['rainfall_category'] = pd.cut(df['rainfall'], bins=[0, 30, 70, 150, np.inf], labels=[0, 1, 2, 3])
        
        # Select features for modeling
        feature_cols = [
            'species_id', 'temperature', 'rainfall', 'season_encoded', 'habitat_area',
            'human_disturbance', 'population_t1', 'population_t2', 'population_change_t1',
            'population_momentum', 'habitat_density', 'capacity_utilization',
            'environmental_stress', 'temp_deviation', 'month', 'carrying_capacity'
        ]
        
        self.feature_columns = feature_cols
        return df[feature_cols]
    
    def train_population_predictor(self, data, enable_tuning=True, cv_folds=5, ensemble=True):
        """
        Train ML model to predict population changes with GPU acceleration
        """
        print("Training Population Prediction Models with GPU acceleration...")

        X = self.prepare_features(data)
        y = data['current_population']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        self.scalers['population'] = StandardScaler()
        X_train_scaled = self.scalers['population'].fit_transform(X_train)
        X_test_scaled = self.scalers['population'].transform(X_test)

        # Configure GPU-accelerated models when available
        base_models = {}
        
        # Standard CPU models
        base_models['RandomForest'] = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        )
        base_models['GradientBoosting'] = GradientBoostingRegressor(random_state=42)
        
        # XGBoost with GPU support
        if self.use_gpu:
            try:
                base_models['XGBoost'] = xgb.XGBRegressor(
                    random_state=42,
                    n_estimators=400,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    tree_method='hist',  # GPU compatible
                    device='cuda:0' if tf.config.list_physical_devices('GPU') else 'cpu',
                    n_jobs=-1
                )
                print("✅ XGBoost configured for GPU acceleration")
            except Exception as e:
                print(f"⚠️  XGBoost GPU setup failed, using CPU: {e}")
                base_models['XGBoost'] = xgb.XGBRegressor(
                    random_state=42, n_estimators=400, learning_rate=0.05, 
                    subsample=0.9, colsample_bytree=0.9, n_jobs=-1
                )
        else:
            base_models['XGBoost'] = xgb.XGBRegressor(
                random_state=42, n_estimators=400, learning_rate=0.05, 
                subsample=0.9, colsample_bytree=0.9, n_jobs=-1
            )
        
        # LightGBM with GPU support
        if self.use_gpu:
            try:
                base_models['LightGBM'] = lgb.LGBMRegressor(
                    random_state=42,
                    n_estimators=600,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    device='gpu',
                    gpu_platform_id=0,
                    gpu_device_id=0,
                    verbose=-1
                )
                print("✅ LightGBM configured for GPU acceleration")
            except Exception as e:
                print(f"⚠️  LightGBM GPU setup failed, using CPU: {e}")
                base_models['LightGBM'] = lgb.LGBMRegressor(
                    random_state=42, n_estimators=600, learning_rate=0.05, 
                    subsample=0.9, colsample_bytree=0.9, verbose=-1
                )
        else:
            base_models['LightGBM'] = lgb.LGBMRegressor(
                random_state=42, n_estimators=600, learning_rate=0.05, 
                subsample=0.9, colsample_bytree=0.9, verbose=-1
            )

        best_model = None
        best_score = -np.inf
        best_model_name = "RandomForest"
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.cv_metrics = {}

        fitted_models = {}
        for name, base_model in base_models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=cv, scoring='r2', n_jobs=-1)
            
            # Fit on full training set
            Xtr, Xte = X_train_scaled, X_test_scaled
            model = base_model
            model.fit(Xtr, y_train)
            score = model.score(Xte, y_test)
            
            fitted_models[name] = model
            self.cv_metrics[name] = {
                'cv_mean_r2': float(np.mean(cv_scores)),
                'cv_std_r2': float(np.std(cv_scores))
            }
            
            print(f"{name} R² Score: {score:.4f}")
            
            if score > best_score:
                best_model = model
                best_score = score
                best_model_name = name

        # Optional ensemble of top-2 CV models
        ensemble_members = None
        if ensemble and len(self.cv_metrics) >= 2:
            # Sort by CV performance and select top 2
            sorted_models = sorted(self.cv_metrics.items(), 
                                 key=lambda x: x[1]['cv_mean_r2'], reverse=True)
            top_2_names = [name for name, _ in sorted_models[:2]]
            
            # Create ensemble
            ensemble_models = [fitted_models[name] for name in top_2_names]
            ensemble_model = self._EnsembleRegressor(ensemble_models)
            
            # Test ensemble performance
            ens_pred = ensemble_model.predict(X_test_scaled)
            ensemble_score = r2_score(y_test, ens_pred)
            
            if ensemble_score > best_score:
                best_model = ensemble_model
                best_score = ensemble_score
                best_model_name = f"Ensemble({','.join(top_2_names)})"
                ensemble_members = top_2_names

        self.models['population'] = {
            'model': best_model,
            'name': best_model_name,
            'score': float(best_score),
            'features': self.feature_columns,
            'cv': self.cv_metrics,
            'ensemble_members': ensemble_members
        }

        print(f"Best model: {best_model_name} (R² = {best_score:.4f})")
        return best_model, best_score
    
    def train_risk_classifier(self, data):
        """
        Train ML model to classify risk levels
        """
        print("Training Risk Classification Model...")
        
        X = self.prepare_features(data)
        y = data['risk_level']
        
        # Encode risk levels
        if 'risk' not in self.label_encoders:
            self.label_encoders['risk'] = LabelEncoder()
        
        y_encoded = self.label_encoders['risk'].fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Scale features
        if 'risk' not in self.scalers:
            self.scalers['risk'] = StandardScaler()
        
        X_train_scaled = self.scalers['risk'].fit_transform(X_train)
        X_test_scaled = self.scalers['risk'].transform(X_test)

        # Calibrated classifier for better probabilities
        base_rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
        model = CalibratedClassifierCV(estimator=base_rf, method='sigmoid', cv=3)
        model.fit(X_train_scaled, y_train)

        accuracy = model.score(X_test_scaled, y_test)
        bal_acc = balanced_accuracy_score(y_test, model.predict(X_test_scaled))

        self.models['risk'] = {
            'model': model,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(bal_acc),
            'features': self.feature_columns
        }

        print(f"Risk Classification Accuracy: {accuracy:.4f}")
        return model, accuracy
    
    def create_lstm_model(self, sequence_length=12, n_features=5):
        """
        Create LSTM model for time series prediction with GPU support
        """
        # Use GPU if available
        device_name = '/GPU:0' if self.use_gpu and tf.config.list_physical_devices('GPU') else '/CPU:0'
        
        with tf.device(device_name):
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            # Use mixed precision for faster training on modern GPUs
            if self.use_gpu and tf.config.list_physical_devices('GPU'):
                try:
                    from tensorflow.keras import mixed_precision
                    policy = mixed_precision.Policy('mixed_float16')
                    mixed_precision.set_global_policy(policy)
                    print("✅ Mixed precision enabled for faster GPU training")
                except:
                    pass
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            print(f"✅ LSTM model created on {device_name}")
        
        return model
    
    def prepare_time_series_data(self, data, sequence_length=12):
        """
        Prepare data for LSTM time series prediction
        """
        # Sort by species and time
        data_sorted = data.sort_values(['species_id', 'year', 'month'])
        
        sequences = []
        targets = []
        
        for species_id in data_sorted['species_id'].unique():
            species_data = data_sorted[data_sorted['species_id'] == species_id]
            
            # Create features for time series (simplified)
            time_features = species_data[['temperature', 'rainfall', 'habitat_area', 
                                        'human_disturbance', 'current_population']].values
            
            # Create sequences
            for i in range(len(time_features) - sequence_length):
                sequences.append(time_features[i:i+sequence_length])
                targets.append(time_features[i+sequence_length, -1])  # current_population
        
        return np.array(sequences), np.array(targets)
    
    def train_lstm_model(self, data):
        """
        Train LSTM model for time series prediction with GPU acceleration
        """
        print("Training LSTM Time Series Model with GPU acceleration...")
        
        X, y = self.prepare_time_series_data(data)
        
        if len(X) < 50:
            print("⚠️  Not enough sequential data for LSTM training")
            return None, float('inf')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the data
        if 'lstm_feature' not in self.scalers:
            self.scalers['lstm_feature'] = MinMaxScaler()
            self.scalers['lstm_target'] = MinMaxScaler()
        
        # Reshape for scaling
        original_shape = X_train.shape
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # Scale features
        X_train_scaled = self.scalers['lstm_feature'].fit_transform(X_train_reshaped)
        X_test_scaled = self.scalers['lstm_feature'].transform(X_test_reshaped)
        
        # Reshape back
        X_train_scaled = X_train_scaled.reshape(original_shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Scale targets
        y_train_scaled = self.scalers['lstm_target'].fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scalers['lstm_target'].transform(y_test.reshape(-1, 1)).flatten()
        
        # Create and train LSTM model
        lstm_model = self.create_lstm_model(
            sequence_length=X.shape[1], 
            n_features=X.shape[2]
        )
        
        # Train with early stopping and GPU optimization
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
            history = lstm_model.fit(
                X_train_scaled, y_train_scaled,
                epochs=50,
                batch_size=32 if not self.use_gpu else 128,  # Larger batch for GPU
                validation_data=(X_test_scaled, y_test_scaled),
                callbacks=callbacks,
                verbose=0
            )
        
        # Evaluate
        y_pred_scaled = lstm_model.predict(X_test_scaled)
        y_pred = self.scalers['lstm_target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        
        self.models['lstm'] = {
            'model': lstm_model,
            'rmse': rmse,
            'history': history.history
        }
        
        test_loss = lstm_model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        print(f"LSTM Model Test Loss (scaled): {test_loss[0]:.4f} | RMSE (original scale): {rmse:.2f}")
        
        return lstm_model, rmse
    
    def predict_population(self, current_data, model_type='population'):
        """
        Predict population using trained models
        """
        if model_type not in self.models:
            print(f"Model {model_type} not trained yet")
            return np.array([0])  # Default fallback
        
        model_info = self.models[model_type]
        model = model_info['model']
        
        try:
            # Prepare features
            X = self.prepare_features(current_data)
            
            if model_type == 'population':
                # Scale features
                X_scaled = self.scalers['population'].transform(X)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X)
                
            return predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            if len(current_data) > 0 and 'population_t1' in current_data.columns:
                # Fallback to previous population
                predictions = np.array([X.iloc[0]['population_t1']])  # Fallback
            else:
                predictions = np.array([100])  # Default
            return predictions
    
    def predict_risk(self, current_data):
        """
        Predict risk levels using trained classifier
        """
        if 'risk' not in self.models:
            return ['Medium'], [0.5]
        
        model_info = self.models['risk']
        model = model_info['model']
        
        try:
            X = self.prepare_features(current_data)
            X_scaled = self.scalers['risk'].transform(X)
            
            # Get predictions and probabilities
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
            # Convert back to labels
            risk_labels = self.label_encoders['risk'].inverse_transform(predictions)
            
            return risk_labels, probabilities.max(axis=1)
        except Exception as e:
            print(f"Risk prediction error: {e}")
            return ['Medium'], [0.5]
    
    def get_feature_importance(self, model_type='population'):
        """
        Get feature importance from trained models
        """
        if model_type not in self.models:
            return None
        
        model_info = self.models[model_type]
        model = model_info['model']
        
        # Check if model has feature importance
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_values = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
        else:
            print(f"Model {model_info['name']} doesn't support feature importance")
            return None
        
        # Create DataFrame with feature importance
        return pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_values
        }).sort_values('importance', ascending=False)
    
    def save_models(self, directory):
        """
        Save all trained models and scalers
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save sklearn models and scalers
        for model_name, model_info in self.models.items():
            if model_name != 'lstm':  # Handle LSTM separately
                joblib.dump(model_info, os.path.join(directory, f"{model_name}_model.pkl"))
        
        # Save scalers and encoders
        joblib.dump(self.scalers, os.path.join(directory, "scalers.pkl"))
        joblib.dump(self.label_encoders, os.path.join(directory, "encoders.pkl"))
        
        # Save LSTM model if it exists
        if 'lstm' in self.models:
            self.models['lstm']['model'].save(os.path.join(directory, "lstm_model.h5"))
            # Save LSTM-specific scalers separately
            joblib.dump({
                'lstm_feature': self.scalers.get('lstm_feature'),
                'lstm_target': self.scalers.get('lstm_target')
            }, os.path.join(directory, "lstm_scalers.pkl"))
        
        print(f"Models saved to {directory}")
    
    def load_models(self, directory):
        """
        Load all trained models and scalers
        """
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return False
        
        try:
            # Load scalers and encoders
            self.scalers = joblib.load(os.path.join(directory, "scalers.pkl"))
            self.label_encoders = joblib.load(os.path.join(directory, "encoders.pkl"))
            
            # Load sklearn models
            for model_file in os.listdir(directory):
                if model_file.endswith("_model.pkl"):
                    model_name = model_file.replace("_model.pkl", "")
                    self.models[model_name] = joblib.load(os.path.join(directory, model_file))
            
            # Load LSTM model if it exists
            lstm_path = os.path.join(directory, "lstm_model.h5")
            if os.path.exists(lstm_path):
                from tensorflow.keras.models import load_model
                lstm_model = load_model(lstm_path)
                
                # Load LSTM scalers
                lstm_scalers = joblib.load(os.path.join(directory, "lstm_scalers.pkl"))
                self.scalers.update(lstm_scalers)
                
                self.models['lstm'] = {
                    'model': lstm_model,
                    'rmse': 0.0  # Will be updated if needed
                }
            
            print(f"Models loaded from {directory}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

if __name__ == "__main__":
    # Test the GPU-accelerated ML predictor
    print("🚀 Testing EcoGuard ML with GPU Acceleration")
    print("=" * 60)
    
    # Initialize with GPU support
    predictor = MLPopulationPredictor(use_gpu=True)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic ecological data...")
    data = predictor.generate_synthetic_data(n_samples=5000, n_species=5)
    print(f"Generated {len(data)} records for {data['species_name'].nunique()} species")
    print(f"Data shape: {data.shape}")
    print(f"Species: {list(data['species_name'].unique())}")
    
    # Train models
    print("\n🤖 Training ML models...")
    pop_model, pop_score = predictor.train_population_predictor(data, enable_tuning=False)
    risk_model, risk_accuracy = predictor.train_risk_classifier(data)
    lstm_model, lstm_rmse = predictor.train_lstm_model(data)
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    sample_data = data.head(3)
    pop_predictions = predictor.predict_population(sample_data)
    risk_predictions, risk_probs = predictor.predict_risk(sample_data)
    
    print(f"Population predictions: {pop_predictions}")
    print(f"Risk predictions: {risk_predictions}")
    
    # Save models
    print("\n💾 Saving models...")
    predictor.save_models("trained_models")
    
    print("\n✅ GPU-accelerated EcoGuard ML system ready!")
