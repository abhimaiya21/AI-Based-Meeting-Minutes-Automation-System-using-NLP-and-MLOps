@echo off
title Final AI Meeting System Demo
echo ==========================================
echo       STARTING FINAL MLOPS PIPELINE
echo ==========================================

:: --- PHASE 1: CONTINUOUS INTEGRATION (CI) ---
echo [STEP 1] Setting up Environment...
call venv\Scripts\activate

echo.
echo [STEP 2] Running Automated Tests (CI)...
:: Set path so tests can import main.py
set PYTHONPATH=.
:: Ensure correct httpx version to prevent pytest crashing
pip install pytest httpx==0.27.0 > nul

:: Run the tests
python -m pytest tests/

:: Check if tests failed
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [FAIL] CRITICAL: Tests Failed! Deployment aborted.
    exit /b %ERRORLEVEL%
)
echo [PASS] All Tests Passed. System is stable.

:: --- PHASE 2: CONTINUOUS MONITORING (CM) ---
echo.
echo [STEP 3] Generating Governance Report (Monitoring)...
python src/monitor.py
IF EXIST drift_report.html (
    echo [PASS] Data Drift Report generated successfully.
) ELSE (
    echo [WARNING] Report generation skipped (first run/no reference data).
)

:: --- PHASE 3: CONTINUOUS DEPLOYMENT (CD) ---
echo.
echo [STEP 4] Full System Deployment (CD)...
echo ==========================================
echo   PIPELINE SUCCESSFUL - LAUNCHING DEMO
echo ==========================================

echo 1. Starting Backend API (FastAPI + Gemini) hidden in background...
:: 'start /B' runs it without blocking this script
start /B /MIN cmd /c "python main.py"

echo 2. Waiting 5 seconds for API to initialize properly...
timeout /t 5 >nul

echo 3. Starting Frontend Dashboard (Streamlit)...
echo    The browser should open automatically.
echo.
echo ===> CLOSE THIS WINDOW TO STOP THE ENTIRE SYSTEM <===
echo.
streamlit run app.py