@echo off
title FastAPI Credit Risk Backend
echo ========================================================
echo   Launching FastAPI Credit Risk Scoring Service
echo ========================================================
echo.
python -m uvicorn api.main:app --reload --port 8000
pause
