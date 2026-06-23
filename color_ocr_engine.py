import cv2
import numpy as np
import easyocr
import re
from typing import Dict, List, Tuple

# Initialize EasyOCR reader for color module
# We can use the same language as the main module or initialize it globally
# It's better to pass the initialized reader to avoid reloading models.
# But for simplicity, we'll accept a reader or init one.
_COLOR_READER = None

def get_color_reader():
    global _COLOR_READER
    if _COLOR_READER is None:
        _COLOR_READER = easyocr.Reader(['fr'], gpu=True)
    return _COLOR_READER

def extract_color_parcels(img_bgr: np.ndarray, reader=None) -> Dict[str, List[Dict]]:
    """
    Extrait le texte rouge et vert d'une image BGR.
    Retourne un dictionnaire:
    {
       "nouvelles_parcelles": [{"valeur": "...", "bbox": [...], "conf": 0.9}],
       "anciennes_parcelles": [...]
    }
    """
    if reader is None:
        reader = get_color_reader()

    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Définition des plages de couleur (H: 0-180, S: 0-255, V: 0-255)
    # Rouge: deux plages car le rouge boucle à 180
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Vert (Elargi pour plans anciens)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Nettoyage morphologique
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Pour enlever les petits bruits isolés
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    results = {
        "nouvelles_parcelles": _run_ocr_on_mask(img_bgr, mask_red, reader),
        "anciennes_parcelles": _run_ocr_on_mask(img_bgr, mask_green, reader)
    }

    return results

def _run_ocr_on_mask(img_bgr: np.ndarray, mask: np.ndarray, reader) -> List[Dict]:
    """
    Trouve les contours dans le masque, découpe les boîtes, passe l'OCR.
    """
    # Trouver les contours du texte coloré
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    extracted = []
    # Fusionner les contours très proches pour regrouper les caractères d'un même nombre
    # On va plutôt extraire des zones un peu larges pour l'OCR
    
    # Une approche plus robuste pour EasyOCR est de masquer l'image originale
    # avec le masque (tout devient blanc ou noir sauf le texte couleur)
    # puis de donner cette image globale à EasyOCR qui se charge de la détection de boîtes.
    
    # Créer une image avec fond blanc
    color_isolated = np.full_like(img_bgr, 255)
    # Copier seulement les pixels du masque
    color_isolated[mask > 0] = img_bgr[mask > 0]
    
    # Convertir en niveaux de gris pour l'OCR
    gray_isolated = cv2.cvtColor(color_isolated, cv2.COLOR_BGR2GRAY)
    
    # Exécuter l'OCR (on force uniquement les chiffres via allowlist)
    ocr_results = reader.readtext(gray_isolated, detail=1, allowlist='0123456789')
    
    for bbox, text, conf in ocr_results:
        # Nettoyage: on ne garde strictement que les chiffres
        val_clean = re.sub(r'[^0-9]', '', text)
        # On ignore les bruits aberrants (ex: très longs ou trop incertains)
        if val_clean and 1 <= len(val_clean) <= 5 and conf >= 0.25:
            (tl, tr, br, bl) = bbox
            x1, y1 = int(tl[0]), int(tl[1])
            x2, y2 = int(br[0]), int(br[1])
            extracted.append({
                "valeur": val_clean,
                "bbox": [x1, y1, x2, y2],
                "conf": conf
            })
            
    # Trier de gauche à droite, haut en bas
    extracted.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
    return extracted
