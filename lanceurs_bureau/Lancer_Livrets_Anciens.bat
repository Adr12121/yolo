@echo off
chcp 65001 >nul
title Lancement Livrets Anciens Serret
color 0B

echo ====================================================================
echo       LANCEMENT DU VIEUX CODE - LIVRETS ANCIENS (SERRET)
echo ====================================================================
echo.
echo Ce script lance "bonne_detection_mauvaise_analyse.py" qui encadrait
echo la commune sur deux colonnes dans les vieux livrets.
echo.

:: Se deplacer dans le repertoire principal
cd /d "C:\Users\Topo_4\Documents\AT_PFE"

:: Verification du dossier inputs_livrets
if not exist "inputs_livrets" mkdir inputs_livrets
if not exist "outputs_livrets" mkdir outputs_livrets

echo [INFO] Veuillez vous assurer que vos PDF/JPG sont dans :
echo        C:\Users\Topo_4\Documents\AT_PFE\inputs_livrets
echo.

echo [INFO] Lancement du script dans l'environnement WSL principal (kraken-env)...
echo --------------------------------------------------------------------

wsl /home/topo_4/kraken-env/bin/python bonne_detection_mauvaise_analyse.py

echo --------------------------------------------------------------------
echo.
echo ====================================================================
echo Execution terminee. Appuyez sur une touche pour fermer.
pause >nul
