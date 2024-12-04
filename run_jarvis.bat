@echo off
echo Starting Jarvis...

REM Check if Anaconda is installed
if not exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    echo Error: Anaconda not found in the expected location.
    echo Please install Anaconda or update the path in this script.
    pause
    exit /b 1
)

REM Activate conda environment
call C:\ProgramData\anaconda3\Scripts\activate.bat jarvis
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'jarvis'
    echo Creating environment...
    call conda create -n jarvis python=3.9 -y
    call C:\ProgramData\anaconda3\Scripts\activate.bat jarvis
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run Jarvis
echo Running Jarvis...
python test_voice.py

if errorlevel 1 (
    echo Error: Failed to run Jarvis. Please check the error messages above.
    pause
    exit /b 1
)

pause
