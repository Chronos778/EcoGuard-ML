#!/usr/bin/env python3
"""
EcoGuard ML - GPU Accelerated Demo
This script demonstrates the GPU-accelerated machine learning capabilities
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from models.ml_predictor_gpu import MLPopulationPredictor

def demo_gpu_acceleration():
    """Demonstrate GPU vs CPU performance differences"""
    print("🌿 EcoGuard ML - GPU Acceleration Demo")
    print("=" * 60)
    
    # Initialize both GPU and CPU versions
    print("\n🔧 Initializing ML Predictors...")
    gpu_predictor = MLPopulationPredictor(use_gpu=True)
    cpu_predictor = MLPopulationPredictor(use_gpu=False)
    
    # Generate larger dataset for meaningful performance comparison
    print("\n📊 Generating Large Ecological Dataset...")
    sample_sizes = [1000, 2000, 5000, 10000]
    results = {
        'sample_size': [],
        'cpu_time': [],
        'gpu_time': [],
        'cpu_accuracy': [],
        'gpu_accuracy': []
    }
    
    for sample_size in sample_sizes:
        print(f"\nTesting with {sample_size} samples...")
        
        # Generate data
        data = gpu_predictor.generate_synthetic_data(n_samples=sample_size, n_species=3)
        
        # CPU Training
        print("  🖥️  Training on CPU...")
        start_time = time.time()
        cpu_model, cpu_score = cpu_predictor.train_population_predictor(data, enable_tuning=False, ensemble=False)
        cpu_time = time.time() - start_time
        
        # GPU Training (mainly XGBoost and LightGBM with GPU support)
        print("  🚀 Training with GPU acceleration...")
        start_time = time.time()
        gpu_model, gpu_score = gpu_predictor.train_population_predictor(data, enable_tuning=False, ensemble=True)
        gpu_time = time.time() - start_time
        
        # Store results
        results['sample_size'].append(sample_size)
        results['cpu_time'].append(cpu_time)
        results['gpu_time'].append(gpu_time)
        results['cpu_accuracy'].append(cpu_score)
        results['gpu_accuracy'].append(gpu_score)
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1
        print(f"    CPU Time: {cpu_time:.2f}s | GPU Time: {gpu_time:.2f}s")
        print(f"    CPU R²: {cpu_score:.4f} | GPU R²: {gpu_score:.4f}")
        print(f"    Speedup: {speedup:.2f}x")
    
    return results

def demo_real_world_predictions():
    """Demonstrate real-world prediction scenarios"""
    print("\n\n🔮 Real-World Prediction Scenarios")
    print("=" * 50)
    
    predictor = MLPopulationPredictor(use_gpu=True)
    
    # Generate training data
    print("Training on comprehensive ecological dataset...")
    training_data = predictor.generate_synthetic_data(n_samples=3000, n_species=5)
    
    # Train all models
    predictor.train_population_predictor(training_data, enable_tuning=False)
    predictor.train_risk_classifier(training_data)
    predictor.train_lstm_model(training_data)
    
    # Create realistic prediction scenarios
    scenarios = [
        {
            'name': 'Climate Change Impact',
            'description': 'Rising temperatures and reduced rainfall',
            'data': {
                'species_id': [0],
                'temperature': [25],  # Higher than normal
                'rainfall': [30],     # Lower than normal
                'season': ['Summer'],
                'habitat_area': [800],
                'human_disturbance': [0.6],  # High disturbance
                'population_t1': [150],
                'population_t2': [180],
                'month': [7],
                'carrying_capacity': [500]
            }
        },
        {
            'name': 'Conservation Success',
            'description': 'Protected area with optimal conditions',
            'data': {
                'species_id': [2],
                'temperature': [18],  # Optimal
                'rainfall': [80],     # Good rainfall
                'season': ['Spring'],
                'habitat_area': [1500], # Large habitat
                'human_disturbance': [0.1],  # Low disturbance
                'population_t1': [300],
                'population_t2': [280],
                'month': [4],
                'carrying_capacity': [800]
            }
        },
        {
            'name': 'Urban Encroachment',
            'description': 'Habitat fragmentation and human pressure',
            'data': {
                'species_id': [1],
                'temperature': [22],
                'rainfall': [45],
                'season': ['Fall'],
                'habitat_area': [400],  # Small habitat
                'human_disturbance': [0.8],  # Very high disturbance
                'population_t1': [80],
                'population_t2': [95],
                'month': [10],
                'carrying_capacity': [200]
            }
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        
        # Create DataFrame
        test_data = pd.DataFrame(scenario['data'])
        
        # Make predictions
        pop_pred = predictor.predict_population(test_data)
        risk_pred, risk_prob = predictor.predict_risk(test_data)
        
        # Calculate trends
        current_pop = scenario['data']['population_t1'][0]
        predicted_pop = int(pop_pred[0])
        population_change = ((predicted_pop - current_pop) / current_pop) * 100
        
        print(f"   📊 Current Population: {current_pop}")
        print(f"   🔮 Predicted Population: {predicted_pop}")
        print(f"   📈 Population Change: {population_change:+.1f}%")
        print(f"   ⚠️  Risk Level: {risk_pred[0]} (confidence: {risk_prob[0]:.2f})")
        
        # Provide recommendations
        if risk_pred[0] == 'High':
            print("   🚨 URGENT ACTION NEEDED!")
            print("      - Implement immediate conservation measures")
            print("      - Reduce human disturbance in habitat")
            print("      - Consider supplemental feeding programs")
        elif risk_pred[0] == 'Medium':
            print("   ⚠️  MONITORING REQUIRED")
            print("      - Increase observation frequency")
            print("      - Prepare contingency plans")
            print("      - Monitor environmental changes")
        else:
            print("   ✅ STABLE POPULATION")
            print("      - Continue current management practices")
            print("      - Regular monitoring sufficient")

def demo_feature_importance():
    """Demonstrate feature importance analysis"""
    print("\n\n🔍 Feature Importance Analysis")
    print("=" * 40)
    
    predictor = MLPopulationPredictor(use_gpu=True)
    
    # Generate and train on data
    data = predictor.generate_synthetic_data(n_samples=2000, n_species=3)
    predictor.train_population_predictor(data, enable_tuning=False, ensemble=False)
    
    # Get feature importance
    importance = predictor.get_feature_importance('population')
    
    if importance is not None:
        print("Top 10 Most Important Features for Population Prediction:")
        print("-" * 60)
        for i, (idx, row) in enumerate(importance.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")
        
        # Provide insights
        print("\n💡 Key Insights:")
        top_feature = importance.iloc[0]['feature']
        print(f"   • {top_feature} is the most critical factor")
        
        if 'population_t1' in importance.head(3)['feature'].values:
            print("   • Previous population is a strong predictor (as expected)")
        
        if 'human_disturbance' in importance.head(5)['feature'].values:
            print("   • Human disturbance significantly impacts populations")
            
        if 'habitat_area' in importance.head(5)['feature'].values:
            print("   • Habitat area is crucial for population sustainability")
    else:
        print("Feature importance not available for ensemble models")
        print("Individual model performances:")
        cv_metrics = predictor.cv_metrics
        for model_name, metrics in cv_metrics.items():
            print(f"   {model_name}: R² = {metrics['cv_mean_r2']:.4f} ± {metrics['cv_std_r2']:.4f}")

def main():
    """Main demo function"""
    print("Starting EcoGuard ML GPU Demo...")
    
    try:
        # Performance comparison
        results = demo_gpu_acceleration()
        
        # Real-world predictions
        demo_real_world_predictions()
        
        # Feature analysis
        demo_feature_importance()
        
        print("\n" + "=" * 60)
        print("🎉 EcoGuard ML GPU Demo Complete!")
        print("✅ GPU acceleration successfully demonstrated")
        print("✅ Real-world scenarios analyzed")
        print("✅ Feature importance calculated")
        print("=" * 60)
        
        # Performance summary
        print("\n📊 Performance Summary:")
        total_samples = sum(results['sample_size'])
        avg_cpu_time = sum(results['cpu_time']) / len(results['cpu_time'])
        avg_gpu_time = sum(results['gpu_time']) / len(results['gpu_time'])
        avg_speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 1
        
        print(f"   Total samples processed: {total_samples:,}")
        print(f"   Average CPU time: {avg_cpu_time:.2f}s")
        print(f"   Average GPU time: {avg_gpu_time:.2f}s")
        print(f"   Average speedup: {avg_speedup:.2f}x")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
