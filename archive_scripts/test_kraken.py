import os, sys
sys.path.append(os.path.expanduser('~/kraken-env/lib/python3.12/site-packages'))

import cv2
import numpy as np
from PIL import Image
import torch
from kraken.lib import models
from kraken import blla, rpred

print(f"PyTorch: {torch.__version__}")

model_path = os.path.expanduser('~/.local/share/htrmopo/48097fc5-ca4a-5bfa-b1c5-d33bb6838156/reichenau_lat_cat_099218.mlmodel')

# Chargement via load_any qui retourne un TorchSeqRecognizer (avec CTC intégré)
print("Chargement du modèle via models.load_any()...")
m = models.load_any(model_path)
print("Type du modèle:", type(m))
print("Modèle chargé avec succès !")

# Image de test
img = np.ones((500, 800, 3), dtype=np.uint8) * 255
cv2.putText(img, "Bonjour monde test", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)
pil_img = Image.fromarray(img).convert('L')
pil_img_rgb = Image.fromarray(img)

print("Segmentation de l'image...")
bounds = blla.segment(pil_img_rgb)

if hasattr(bounds, 'lines'):
    nb_lignes = len(bounds.lines)
else:
    nb_lignes = len(bounds.get('lines', []))

print(f"Nombre de lignes détectées : {nb_lignes}")

print("Reconnaissance du texte...")
try:
    preds_gen = rpred.rpred(network=m, im=pil_img_rgb, bounds=bounds)
    resultats = list(preds_gen)
    print(f"Nombre de prédictions : {len(resultats)}")
    for i, rec in enumerate(resultats):
        print(f"  Ligne {i+1} : '{rec.prediction}' (confiance moy: {sum(rec.confidences)/len(rec.confidences) if rec.confidences else 'N/A'})")
    print("\n=== SUCCÈS : La reconnaissance Kraken fonctionne ! ===")
except Exception as e:
    import traceback
    print("Erreur:", e)
    traceback.print_exc()
