#!/bin/bash
# Quick start script for EcoGuard ML (Linux/macOS)

echo "================================================"
echo "  EcoGuard ML - Quick Start"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        echo "Please ensure Python 3.10+ is installed"
        exit 1
    fi
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "Installing dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "Installation failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "Running setup verification..."
python setup_check.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Setup verification failed. Please fix the issues above."
    exit 1
fi

echo ""
echo "================================================"
echo "  Starting EcoGuard ML..."
echo "================================================"
echo ""
echo "The application will open in your browser"
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
