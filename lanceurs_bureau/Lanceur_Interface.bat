@echo off
chcp 65001 >nul
title Lancement Interface Geofoncier
echo Lancement de l'interface Geofoncier... Ne fermez pas cette fenetre !
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m streamlit run app_validation.py --server.address 0.0.0.0
) else (
    py -m streamlit run app_validation.py --server.address 0.0.0.0
)
pause
