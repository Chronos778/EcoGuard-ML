@echo off
REM Quick start script for EcoGuard ML (Windows)

echo ================================================
echo   EcoGuard ML - Quick Start
echo ================================================
echo.

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment
        echo Please ensure Python 3.10+ is installed
        pause
        exit /b 1
    )
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo Installation failed. Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo Running setup verification...
python setup_check.py

if errorlevel 1 (
    echo.
    echo Setup verification failed. Please fix the issues above.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Starting EcoGuard ML...
echo ================================================
echo.
echo The application will open in your browser
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py

pause
