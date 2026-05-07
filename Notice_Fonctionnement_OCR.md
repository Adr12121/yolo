# Notice de Fonctionnement : Script d'Extraction et OCR Cadastral (`main.py`)

Ce document explique en détail le fonctionnement du script Python `main.py`. Il est conçu pour traiter des plans cadastraux (DMPC, PV de bornage) en distinguant le texte imprimé du texte manuscrit afin d'en extraire les informations clés (Commune, Géomètre, Numéros de dossiers et parcelles).

---

## 1. Initialisation et Outils (Lignes 1 à 65)

Le script commence par importer les bibliothèques nécessaires à son fonctionnement.

```python
import os
import json
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from ultralytics import YOLO
import easyocr
import subprocess
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import re
from kraken import blla, rpred
from kraken.lib import models as kraken_models
import unicodedata
from spellchecker import SpellChecker
```

### Le Correcteur Orthographique Géographique (`correct_ocr_with_dict`)
L'OCR se trompe souvent sur les noms propres. Le script charge en mémoire la liste des communes (Drôme/Ardèche). Lorsqu'un texte est identifié comme une commune potentielle, il est "nettoyé" (sans ponctuation, en majuscule) mais en **gardant ses accents**. 

Il utilise ensuite l'algorithme "Fuzzy Matching" (`rapidfuzz`) pour trouver le mot du dictionnaire qui lui ressemble le plus (distance de Levenshtein).
```python
    t_clean = str(texte).strip().upper()
    t_clean = re.sub(r'[^A-ZÀ-Ÿ0-9 ]', ' ', t_clean)
    
    # ...
    result = process.extractOne(t_clean, dictionnaire, scorer=fuzz.WRatio)
    if result:
        meilleur_match, score, _ = result
        if score >= seuil:
            return meilleur_match # Retourne le mot exact du dictionnaire
```

---

## 2. Les Règles Métiers (Lignes 67 à 131)

L'IA n'a pas de contexte métier. La fonction `correct_cadastral_rules` applique des formules mathématiques (`Regex`) pour empêcher les erreurs classiques d'analyse d'image.

**Exemple : Les numéros de parcelles**
L'OCR confond fréquemment la lettre `O` avec le chiffre `0`, ou le `l` (L minuscule) avec le `1`. Le script vérifie si un petit bloc de texte contient des chiffres et des lettres ambiguës, puis force la correction.

```python
    def repl_numero(m):
        val = m.group(0)
        # Bloc de 1 à 5 caractères, avec au moins un chiffre et des lettres ambiguës
        if 1 <= len(val) <= 5 and any(c.isdigit() for c in val) and re.match(r'^[0-9OolISsB]+$', val):
            val_corrige = val.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('S', '5').replace('s', '5').replace('B', '8')
            if val_corrige.isdigit():
                return val_corrige
        return val

    # Application à tous les blocs alphanumériques
    texte_corrige = re.sub(r'\b[A-Za-zÀ-ÿ0-9]+\b', repl_numero, texte_corrige)
```

---

## 3. Prétraitement de l'Image (Lignes 213 à 307)

Avant qu'une IA ne lise quoi que ce soit, il faut préparer l'image.

### Extraction via PyMuPDF (`read_document`)
Un plan cadastral en PDF contient des textes minuscules. Le script applique d'office un "zoom mathématique" X3 avant de convertir la page en image, augmentant artificiellement les DPI (points par pouce) pour que Tesseract puisse lire l'imprimé régulier.
```python
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat)
            img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
```

### Amélioration Radiométrique (`preprocess_roi_for_ocr`)
Les vieux plans sont souvent fades ou granuleux. 
1. `cv2.bilateralFilter` nettoie le grain de la feuille sans flouter les lettres.
2. `CLAHE` (Contrast Limited Adaptive Histogram Equalization) assombrit l'encre délavée localement.
```python
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(blur)
```

---

## 4. Le Moteur Hybride (Lignes 516 à 866)

La fonction `process_document_hybrid` est le cœur du programme. Elle résout le problème principal : empêcher l'OCR manuscrit de "halluciner" sur du texte imprimé.

### A. La Passe "Imprimé" (Tesseract)
Le script confie l'intégralité de la page convertie en Noir & Blanc à Tesseract.
```python
        custom_tess_config = r'--oem 3 --psm 3 -l fra'
        tess_df = run_tesseract_windows(thresh_full, custom_tess_config, output_type='tsv')
```
Si Tesseract lit un mot avec un niveau de confiance supérieur à 50% ET que ce mot ressemble bien à une phrase ou un chiffre valide (pas une suite de caractères spéciaux), alors **ce mot est virtuellement "effacé"** (peint en blanc) sur le masque `mask_tesseract_lu`.

### B. La Passe "Manuscrit" (Kraken + TrOCR)
Le script regarde l'image où tous les textes clairs imprimés ont été gommés. Il reste les signatures, les vieilles écritures et le bruit.
Il délègue à **Kraken (`blla`)** la tâche de segmenter ces zones résiduelles en "polygones".

Pour chaque polygone manuscrit :
1. **TrOCR** extrait le texte (il est doué en français mais ne sait pas dire s'il est sûr de lui).
2. **PyLaia** (CatMuS) lit la même chose uniquement pour générer un pourcentage de confiance mathématique fiable.
```python
                pil_img_crop = Image.fromarray(roi_color_clean)
                pixel_values = processor(pil_img_crop, return_tensors="pt").pixel_values.to(device)
                generated_ids = trocr_model.generate(pixel_values, max_new_tokens=20)
                texte_extrait = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### C. La Spatialisation
Les IA renvoient des nuages de mots. Le script trie tous ces rectangles par leur hauteur (`Y`), crée une tolérance de pixels, puis les aligne de gauche à droite (`X`) pour reconstituer des "Lignes" sémantiques.

---

## 5. Intelligence Artificielle d'Extraction (Lignes 398 à 513)

### Recherche des Variables Clés (`extract_kpis_from_layout`)
Le script possède une liste de "pivots" (ex: `dossier n°`, `géomètre expert`).
Il lit les lignes reconstituées. S'il trouve un pivot, il cherche d'abord **à droite** sur la même ligne. S'il ne trouve rien, il attrape le texte situé sur **la ligne du dessous**.
```python
    pivots = {
        "commune": ["commune de", "ville de", "territoire de la commune"],
        "geometre": ["géomètre expert", "cabinet", "dessiné par"],
        ...
    }
```

### Classification du Document (`classify_document`)
Un système de "scoring" additionne les occurrences de certains mots dans l'entièreté de la page. "Arpentage" et "DMPC" donnent des points à la catégorie "Document d'Arpentage (DMPC)". S'il y a mention d'un "Ordre" (détecté en KPI), cela rajoute 2 points supplémentaires à cette catégorie, assurant la robustesse du tri final.
