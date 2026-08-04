@echo off
echo Lancement de l'OCR en cours (Ne fermez pas cette fenetre)...
wsl -e bash -c "source ~/kraken-env/bin/activate && cd /mnt/c/Users/Topo_4/Documents/AT_PFE/Anti/yolo && if [ -f patch_classifier2.py ]; then echo '[AUTO-PATCH] Application des ameliorations...' && python patch_classifier2.py && rm -f patch_classifier2.py patch_classifier.py && echo '[AUTO-PATCH] Patch applique et supprime.'; fi && pip install -q pydantic 2>/dev/null && python main.py"
echo Termine.
pause

