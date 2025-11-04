#!/usr/bin/env python3
"""
Setup script to verify EcoGuard ML installation and dependencies
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.10+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is not None:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name} ({version})")
            return True
        except:
            print(f"⚠️  {package_name} (installed but import failed)")
            return False
    else:
        print(f"❌ {package_name} (not installed)")
        return False

def main():
    """Main setup verification"""
    print("=" * 60)
    print("🌿 EcoGuard ML - Installation Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    if not check_python_version():
        print("\n⚠️  Please upgrade Python to version 3.10 or higher")
        return False
    
    print()
    print("📦 Checking required packages...")
    print()
    
    # Core packages
    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('plotly', 'plotly'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('joblib', 'joblib'),
    ]
    
    # Optional packages
    optional_packages = [
        ('tensorflow', 'tensorflow'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('optuna', 'optuna'),
        ('shap', 'shap'),
    ]
    
    all_installed = True
    
    # Check required packages
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print()
    print("📦 Checking optional packages (for advanced features)...")
    print()
    
    # Check optional packages
    optional_count = 0
    for package_name, import_name in optional_packages:
        if check_package(package_name, import_name):
            optional_count += 1
    
    print()
    print("=" * 60)
    
    if all_installed:
        print("✅ All required packages are installed!")
        print(f"✅ {optional_count}/{len(optional_packages)} optional packages installed")
        print()
        print("🚀 Ready to run: streamlit run app.py")
        return True
    else:
        print("❌ Some required packages are missing")
        print()
        print("📥 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
