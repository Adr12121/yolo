@echo off
chcp 65001 >nul
title Lancement Pipeline OCR
echo Lancement du Pipeline OCR en cours...
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe main.py
) else (
    py main.py
)
pause

