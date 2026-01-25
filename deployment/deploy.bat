@echo off
REM Deployment script for SentiSight (Windows)

echo ========================================
echo SentiSight Deployment Script
echo ========================================

REM Check if .env exists
if not exist .env (
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo WARNING: Please update .env with your configuration
    exit /b 1
)

REM Build and start services
echo Building Docker images...
docker-compose build

echo Starting services...
docker-compose up -d

echo Waiting for services to be healthy...
timeout /t 10 /nobreak > nul

REM Check health
echo Checking API health...
curl -f http://localhost:8000/health

echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo.
echo Services:
echo   - API:        http://localhost:8000
echo   - API Docs:   http://localhost:8000/docs
echo   - Frontend:   http://localhost:8501
echo   - Database:   localhost:5432
echo.
echo View logs:
echo   docker-compose logs -f
echo.
echo Stop services:
echo   docker-compose down
echo ========================================
