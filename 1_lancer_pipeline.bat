@echo off
chcp 65001 >nul
title Pipeline OCR Cadastral — Lancement
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       PIPELINE OCR CADASTRAL  —  Extraction Auto        ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

:: ── Répertoire du script ───────────────────────────────────────────────
cd /d "%~dp0"

:: ── Vérification du venv ───────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [ERREUR] Environnement .venv introuvable.
    echo          Creez-le avec :  python -m venv .venv
    echo          Puis :           .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

set PYTHON=.venv\Scripts\python.exe
set PIP=.venv\Scripts\pip.exe

:: ── Etape 0 : Dependances manquantes (rapide, skip si deja ok) ─────────
echo [1/4] Verification des dependances...
%PIP% install -q pydantic rapidfuzz requests 2>nul
echo       OK.
echo.

:: ── Etape 1 : Patches automatiques ────────────────────────────────────
:: Si patch_classifier2.py existe, on l'applique UNE FOIS puis on le supprime
if exist "patch_classifier2.py" (
    echo [2/4] Application des patches d'amelioration...
    %PYTHON% patch_classifier2.py
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR] Le patch a echoue. Verifiez patch_classifier2.py
        pause
        exit /b 1
    )
    echo       Patch applique avec succes.
    del /f /q "patch_classifier2.py" 2>nul
    del /f /q "patch_classifier.py"  2>nul
    echo       Fichiers de patch supprimes.
) else (
    echo [2/4] Aucun patch en attente.
)
echo.

:: ── Etape 2 : Nettoyage du dossier outputs (optionnel) ─────────────────
echo [3/4] Preparation du dossier outputs...
if not exist "outputs" mkdir outputs
if not exist "inputs"  mkdir inputs

:: Verifier qu'il y a des fichiers dans inputs
set FILE_COUNT=0
for %%f in (inputs\*.pdf inputs\*.jpg inputs\*.png inputs\*.tiff) do set /a FILE_COUNT+=1

if %FILE_COUNT% equ 0 (
    echo.
    echo [ATTENTION] Le dossier 'inputs' est VIDE.
    echo             Deposez vos fichiers PDF/JPG dans :
    echo             %~dp0inputs\
    echo.
    pause
    exit /b 0
)
echo       %FILE_COUNT% fichier(s) detecte(s) dans inputs.
echo.

:: ── Etape 3 : Lancement du pipeline ────────────────────────────────────
echo [4/4] Lancement du pipeline OCR...
echo ─────────────────────────────────────────────────────────────
echo.

:: Enregistrer la sortie dans un log horodate
set LOG_FILE=outputs\log_exec.txt
echo Execution demarree le %DATE% a %TIME% > "%LOG_FILE%"
echo. >> "%LOG_FILE%"

%PYTHON% main.py 2>&1 | tee "%LOG_FILE%"

echo.
echo ─────────────────────────────────────────────────────────────
if %ERRORLEVEL% equ 0 (
    echo [OK] Pipeline termine avec succes.
    echo      Resultats disponibles dans : %~dp0outputs\
) else (
    echo [ERREUR] Le pipeline a rencontre une erreur.
    echo          Consultez le log : %LOG_FILE%
)

echo.
echo Appuyez sur une touche pour fermer...
pause >nul
