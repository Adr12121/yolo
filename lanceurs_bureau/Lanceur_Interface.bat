@echo off
echo Lancement de l'interface Geofoncier... Ne fermez pas cette fenetre !
cd C:\Users\Topo_4\Documents\AT_PFE\Anti\yolo
py -m streamlit run app_validation.py --server.address 0.0.0.0
pause
