import plan_classifier
import json
import easyocr
import cv2

reader = easyocr.Reader(['fr'], gpu=False, verbose=False)

plans = [
    'inputs/geofoncier_dmpc_07116_000_262 (1).pdf',
    'inputs/geofoncier_dmpc_07289_000_608.pdf',
    'inputs/geofoncier_dmpc_07289_000_677.pdf',
]

for p in plans:
    print(f"\nProcessing {p}...")
    res = plan_classifier.process_plan(p, commune_db=[], models=[None, reader])
    champs = res.get('champs', {})
    n_ordre = champs.get('n_ordre', {})
    if isinstance(n_ordre, dict):
        print(f"  n_ordre = {n_ordre.get('valeur')} (methode={n_ordre.get('methode')})")
    else:
        print(f"  n_ordre = {n_ordre}")
