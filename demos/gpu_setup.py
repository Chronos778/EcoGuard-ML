#!/usr/bin/env python3
"""
GPU Setup and Configuration Script for EcoGuard ML
This script helps configure GPU acceleration for machine learning models
"""

import sys
import os
import subprocess
import warnings
warnings.filterwarnings('ignore')

def check_gpu_hardware():
    """Check if GPU hardware is available"""
    print("🔍 Checking GPU Hardware...")
    
    try:
        import platform
        system = platform.system()
        
        if system == "Windows":
            # Check for NVIDIA GPU on Windows
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("✅ NVIDIA GPU detected:")
                    # Parse nvidia-smi output for GPU info
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Quadro' in line:
                            gpu_info = line.strip()
                            if gpu_info:
                                print(f"   - {gpu_info}")
                    return True, "nvidia"
                else:
                    print("ℹ️  nvidia-smi not found or no NVIDIA GPU")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("ℹ️  nvidia-smi not available")
        
        # Check for AMD GPU (basic check)
        try:
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                if 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    print("✅ AMD GPU detected (limited ML support)")
                    return True, "amd"
        except:
            pass
            
        print("ℹ️  No dedicated GPU detected or drivers not installed")
        return False, "none"
        
    except Exception as e:
        print(f"⚠️  GPU detection failed: {e}")
        return False, "none"

def check_tensorflow_gpu():
    """Check TensorFlow GPU support"""
    print("\n🧠 Checking TensorFlow GPU Support...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check GPU devices
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow can see {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                
            # Test GPU functionality
            try:
                with tf.device('/GPU:0'):
                    test = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    result = tf.reduce_sum(test)
                    print(f"✅ GPU computation test: {result.numpy()}")
                return True
            except Exception as e:
                print(f"⚠️  GPU computation test failed: {e}")
                return False
        else:
            print("ℹ️  No GPU devices found by TensorFlow")
            
            # Check for CUDA
            try:
                cuda_version = tf.test.is_built_with_cuda()
                if cuda_version:
                    print("ℹ️  TensorFlow was built with CUDA support")
                else:
                    print("ℹ️  TensorFlow was NOT built with CUDA support")
            except:
                pass
                
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ TensorFlow GPU check failed: {e}")
        return False

def check_xgboost_gpu():
    """Check XGBoost GPU support"""
    print("\n🌳 Checking XGBoost GPU Support...")
    
    try:
        import xgboost as xgb
        print(f"✅ XGBoost version: {xgb.__version__}")
        
        # Test GPU training
        try:
            # Create a simple test dataset
            import numpy as np
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=100, n_features=10, random_state=42)
            
            # Try GPU training
            model = xgb.XGBRegressor(tree_method='hist', device='cuda:0', n_estimators=10)
            model.fit(X, y)
            print("✅ XGBoost GPU training successful")
            return True
            
        except Exception as e:
            print(f"⚠️  XGBoost GPU test failed: {e}")
            print("ℹ️  XGBoost will use CPU")
            return False
            
    except ImportError:
        print("❌ XGBoost not installed")
        return False

def check_lightgbm_gpu():
    """Check LightGBM GPU support"""
    print("\n🌟 Checking LightGBM GPU Support...")
    
    try:
        import lightgbm as lgb
        print(f"✅ LightGBM version: {lgb.__version__}")
        
        # Test GPU training
        try:
            import numpy as np
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=100, n_features=10, random_state=42)
            
            # Try GPU training
            model = lgb.LGBMRegressor(device='gpu', n_estimators=10, verbose=-1)
            model.fit(X, y)
            print("✅ LightGBM GPU training successful")
            return True
            
        except Exception as e:
            print(f"⚠️  LightGBM GPU test failed: {e}")
            print("ℹ️  LightGBM will use CPU")
            return False
            
    except ImportError:
        print("❌ LightGBM not installed")
        return False

def check_cupy():
    """Check CuPy for GPU-accelerated NumPy"""
    print("\n🚀 Checking CuPy Support...")
    
    try:
        import cupy as cp
        print(f"✅ CuPy version: {cp.__version__}")
        
        # Test GPU arrays
        try:
            gpu_array = cp.array([1, 2, 3, 4, 5])
            result = cp.sum(gpu_array)
            print(f"✅ CuPy GPU computation: {result}")
            return True
        except Exception as e:
            print(f"⚠️  CuPy test failed: {e}")
            return False
            
    except ImportError:
        print("ℹ️  CuPy not installed - GPU-accelerated NumPy not available")
        return False

def install_gpu_packages():
    """Install GPU-accelerated packages"""
    print("\n📦 GPU Package Installation Guide")
    print("=" * 50)
    
    gpu_present, gpu_type = check_gpu_hardware()
    
    if not gpu_present:
        print("⚠️  No GPU detected. GPU acceleration not recommended.")
        return
    
    if gpu_type == "nvidia":
        print("🔧 NVIDIA GPU detected. Installation recommendations:")
        print("\n1. Install CUDA Toolkit (if not already installed):")
        print("   - Download from: https://developer.nvidia.com/cuda-downloads")
        print("   - Recommended: CUDA 11.8 or 12.x")
        
        print("\n2. Install cuDNN (if not already installed):")
        print("   - Download from: https://developer.nvidia.com/cudnn")
        print("   - Follow NVIDIA installation guide")
        
        print("\n3. Install GPU-accelerated Python packages:")
        print("   pip install tensorflow[and-cuda]  # TensorFlow with GPU")
        print("   pip install cupy-cuda11x  # or cupy-cuda12x depending on CUDA version")
        print("   pip install xgboost[gpu]  # XGBoost with GPU support")
        
        print("\n4. For LightGBM GPU support:")
        print("   pip uninstall lightgbm")
        print("   pip install lightgbm --install-option=--gpu")
        
        print("\n5. Verify installation by running this script again")
        
    elif gpu_type == "amd":
        print("🔧 AMD GPU detected:")
        print("   - Limited ML acceleration support")
        print("   - Consider ROCm for AMD GPU support (Linux)")
        print("   - Most ML frameworks prioritize NVIDIA CUDA")
    
    print("\nAlternatively, run the automated installer:")
    print("python install_gpu_support.py")

def create_gpu_installer():
    """Create automated GPU support installer"""
    installer_code = '''#!/usr/bin/env python3
"""
Automated GPU Support Installer for EcoGuard ML
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 Installing GPU Support for EcoGuard ML...")
    
    packages = [
        "tensorflow[and-cuda]",
        "cupy-cuda11x", 
        "xgboost",
    ]
    
    for package in packages:
        print(f"\\nInstalling {package}...")
        success = install_package(package)
        if success:
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
    
    # Special handling for LightGBM GPU
    print("\\nInstalling LightGBM with GPU support...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "lightgbm", "-y"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "--install-option=--gpu"])
        print("✅ LightGBM GPU installed successfully")
    except:
        print("⚠️  LightGBM GPU installation failed, using CPU version")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    
    print("\\n✅ GPU support installation complete!")
    print("Run 'python gpu_setup.py' to verify installation")

if __name__ == "__main__":
    main()
'''
    
    with open('install_gpu_support.py', 'w') as f:
        f.write(installer_code)
    
    print("✅ Created install_gpu_support.py")

def test_gpu_performance():
    """Test GPU vs CPU performance"""
    print("\n⚡ GPU Performance Test")
    print("=" * 30)
    
    try:
        import time
        import numpy as np
        import tensorflow as tf
        
        # Test data
        size = 5000
        X = np.random.randn(size, 100).astype(np.float32)
        y = np.random.randn(size, 1).astype(np.float32)
        
        # CPU test
        print("Testing CPU performance...")
        start_time = time.time()
        with tf.device('/CPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=5, verbose=0, batch_size=32)
        cpu_time = time.time() - start_time
        print(f"CPU training time: {cpu_time:.2f} seconds")
        
        # GPU test
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("Testing GPU performance...")
            start_time = time.time()
            with tf.device('/GPU:0'):
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=5, verbose=0, batch_size=128)  # Larger batch for GPU
            gpu_time = time.time() - start_time
            print(f"GPU training time: {gpu_time:.2f} seconds")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        else:
            print("No GPU available for performance test")
            
    except Exception as e:
        print(f"Performance test failed: {e}")

def main():
    """Main GPU setup function"""
    print("🌿 EcoGuard ML - GPU Setup & Configuration")
    print("=" * 60)
    
    # Hardware check
    gpu_present, gpu_type = check_gpu_hardware()
    
    # Software checks
    tf_gpu = check_tensorflow_gpu()
    xgb_gpu = check_xgboost_gpu()
    lgb_gpu = check_lightgbm_gpu()
    cupy_available = check_cupy()
    
    # Summary
    print("\\n📊 GPU Support Summary")
    print("=" * 30)
    print(f"GPU Hardware: {'✅' if gpu_present else '❌'} ({gpu_type})")
    print(f"TensorFlow GPU: {'✅' if tf_gpu else '❌'}")
    print(f"XGBoost GPU: {'✅' if xgb_gpu else '❌'}")
    print(f"LightGBM GPU: {'✅' if lgb_gpu else '❌'}")
    print(f"CuPy: {'✅' if cupy_available else '❌'}")
    
    gpu_ready = tf_gpu or xgb_gpu or lgb_gpu
    
    if gpu_ready:
        print("\\n🎉 GPU acceleration is available!")
        print("Your EcoGuard ML system can use GPU acceleration.")
        
        # Performance test
        test_gpu_performance()
        
    else:
        print("\\n⚠️  GPU acceleration not fully configured")
        if gpu_present:
            print("GPU hardware detected but software not configured.")
            install_gpu_packages()
            create_gpu_installer()
        else:
            print("No GPU hardware detected. Using CPU is still fast for most tasks.")
    
    print("\\n🚀 Ready to run EcoGuard ML!")
    print("Use: python models/ml_predictor_gpu.py")

if __name__ == "__main__":
    main()
