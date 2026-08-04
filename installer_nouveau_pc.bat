@echo off
chcp 65001 >nul
title Installation du projet sur le nouvel ordinateur
color 0B

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║    INSTALLATION DU PROJET PIPELINE OCR & VALIDATION     ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

:: 1. Verification de Python
echo [1/5] Verification de Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH systeme.
    echo          Veuillez installer Python 3.10 ou superieur depuis python.org.
    echo          N'oubliez pas de cocher "Add Python to PATH" pendant l'installation.
    echo.
    pause
    exit /b 1
)
echo       Python detecte avec succes.
echo.

:: 2. Creation de l'environnement virtuel .venv
echo [2/5] Verification / Creation de l'environnement virtuel (.venv)...
if not exist ".venv\Scripts\python.exe" (
    echo       Creation du dossier .venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR] Impossible de creer l'environnement virtuel.
        pause
        exit /b 1
    )
    echo       Environnement virtuel cree.
) else (
    echo       Environnement virtuel .venv deja existant.
)
echo.

:: 3. Installation des dependances Python
echo [3/5] Installation des bibliotheques Python depuis requirements.txt...
.venv\Scripts\python.exe -m pip install --upgrade pip >nul 2>&1
.venv\Scripts\pip.exe install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Une erreur est survenue lors de l'installation des dependances.
    pause
    exit /b 1
)
echo       Dependances installees avec succes.
echo.

:: 4. Creation des dossiers requis
echo [4/5] Verification des dossiers requis (inputs, outputs, runs)...
if not exist "inputs" mkdir inputs
if not exist "outputs" mkdir outputs
if not exist "runs" mkdir runs
echo       Dossiers de travail prets.
echo.

:: 5. Verification du fichier .env
echo [5/5] Verification du fichier de configuration (.env)...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo [ATTENTION] Fichier .env cree a partir de .env.example.
        echo             Pensez a renseigner vos cles d'API Geofoncier dans le fichier .env !
    ) else (
        echo [ATTENTION] Aucun fichier .env ou .env.example trouve.
    )
) else (
    echo       Fichier .env detecte.
)
echo.

echo ╔══════════════════════════════════════════════════════════╗
echo ║                 INSTALLATION TERMINEE !                  ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo RAPPEL POUR LE DÉPLOIEMENT VIA GIT :
echo 1. Si les fichiers de modeles (.pt, .mlmodel) ou de donnees lourdes (.json)
echo    ne sont pas sur Git (ignores par .gitignore), pensez a les copier
echo    manuellement dans ce dossier.
echo 2. Pensez a verifier votre fichier .env pour les cles d'API Geofoncier.
echo.
echo Pour lancer l'interface de validation :
echo    .venv\Scripts\python.exe -m streamlit run app_validation.py
echo.
echo Pour lancer le pipeline complet :
echo    1_lancer_pipeline.bat
echo.
pause
