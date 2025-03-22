@echo off
setlocal

echo Jarvis AI Assistant - Digital Ocean Deployment Script
echo ======================================================

REM Set the path to doctl
set DOCTL_PATH=C:\Program Files\doctl\doctl.exe

REM Check if doctl exists at the specified path
if not exist "%DOCTL_PATH%" (
    echo Error: doctl not found at %DOCTL_PATH%
    echo Please ensure doctl is installed at the correct location.
    exit /b 1
)

REM Check if user is authenticated with doctl
"%DOCTL_PATH%" account get >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: You are not authenticated with Digital Ocean.
    echo Please run '%DOCTL_PATH% auth init' and follow the instructions.
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found.
    echo Creating a template .env file. Please edit it with your API keys.
    echo WIT_EN_KEY=your_wit_key_here > .env
    echo GOOGLE_API_KEY=your_google_key_here >> .env
    echo A template .env file has been created. Please edit it with your API keys and run this script again.
    exit /b 1
)

REM Check if .do directory exists
if not exist .do (
    mkdir .do
)

REM Check if .do/app.yaml exists
if not exist .do\app.yaml (
    echo Creating .do/app.yaml file...
    echo name: jarvis-ai-assistant > .do\app.yaml
    echo region: fra >> .do\app.yaml
    echo services: >> .do\app.yaml
    echo - name: jarvis-api >> .do\app.yaml
    echo   dockerfile_path: Dockerfile >> .do\app.yaml
    echo   source_dir: / >> .do\app.yaml
    echo   http_port: 8080 >> .do\app.yaml
    echo   instance_count: 1 >> .do\app.yaml
    echo   instance_size_slug: basic-xs >> .do\app.yaml
    echo   routes: >> .do\app.yaml
    echo   - path: / >> .do\app.yaml
    echo   envs: >> .do\app.yaml
    echo   - key: WIT_EN_KEY >> .do\app.yaml
    echo     scope: RUN_TIME >> .do\app.yaml
    echo     value: ${WIT_EN_KEY} >> .do\app.yaml
    echo   - key: GOOGLE_API_KEY >> .do\app.yaml
    echo     scope: RUN_TIME >> .do\app.yaml
    echo     value: ${GOOGLE_API_KEY} >> .do\app.yaml
)

echo Please select a deployment option:
echo 1) Deploy to Digital Ocean App Platform
echo 2) Deploy to a Digital Ocean Droplet
set /p choice=Enter your choice (1 or 2): 

if "%choice%"=="1" (
    echo Deploying to Digital Ocean App Platform...
    echo Deploying app...
    "%DOCTL_PATH%" apps create --spec .do\app.yaml
    echo Deployment initiated!
    echo You can check the status in the Digital Ocean dashboard.
    goto :end
)

if "%choice%"=="2" (
    echo Preparing for Droplet deployment...
    echo Creating deployment package...
    powershell -Command "Compress-Archive -Path .\* -DestinationPath .\jarvis-deployment.zip -Force"
    echo Deployment package created: jarvis-deployment.zip
    echo.
    echo To deploy to a Droplet, follow these steps:
    echo 1. Create a Droplet in the Digital Ocean dashboard
    echo 2. Upload the deployment package to your Droplet:
    echo    scp jarvis-deployment.zip root@your-droplet-ip:~/
    echo 3. SSH into your Droplet:
    echo    ssh root@your-droplet-ip
    echo 4. Extract and deploy:
    echo    mkdir -p jarvis ^&^& unzip jarvis-deployment.zip -d jarvis
    echo    cd jarvis ^&^& docker-compose up -d
    goto :end
)

echo Invalid choice. Exiting.
exit /b 1

:end
echo Done!
endlocal
