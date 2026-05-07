import cv2
import numpy as np
import json
import ast
from pathlib import Path
from rapidfuzz import process, fuzz
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import warnings

warnings.filterwarnings('ignore') # Désactiver certains warnings HuggingFace dans les logs

# ==============================================================================
# 0. CONSTANTES ET ALIAS (Gestiion des Abréviations)
# ==============================================================================

# Dictionnaire d'alias spécifique aux "livrets de Fernand"
# Clé: Forme abrégée ou erronée (en minuscules pour faciliter le nettoyage)
# Valeur: Nom officiel de la commune
ALIAS_COMMUNES = {
    "alba": "Alba-la-Romaine",
    "la chapelle": "La Chapelle-sous-Aubenas",
    "st laurent baths": "Saint-Laurent-les-Bains-Laval-d'Aurelle",
    # Structure extensible : ajoute d'autres alias ici facilement.
}

# Charger un mock de base de données de communes de l'Ardèche 
# ou remplacer par la vraie base JSON
DB_COMMUNES = [
    "Alba-la-Romaine", "Aubenas", "Privas", "Juvinas", "Ucel", 
    "Vinezac", "Fons", "Saint-Prix", "La Chapelle-sous-Aubenas", 
    "Vals-les-Bains", "Le Teil"
]

# ==============================================================================
# 1. SEGMENTATION DE ZONE (OpenCV)
# ==============================================================================
def isoler_colonne_commune(image_path, x_start=None, x_end=None, width_ratio=(0.4, 0.6)):
    """
    Segmentation de zone : Isole la colonne "Commune" d'une image.
    Si un masque (x_start, x_end) est donné en dur, on l'utilise.
    Sinon, on fait un crop statistique ou basé sur un ratio de largeur (width_ratio).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {image_path}")

    h, w, _ = img.shape

    # Stratégie 1 : Crop fixe si défini
    if x_start is not None and x_end is not None:
        col_img = img[:, x_start:x_end]
        return col_img, (x_start, 0, x_end, h)

    # Stratégie 2 : Utiliser un pourcentage de largeur (ex: de 40% à 60% de l'image)
    # Dans les livrets, la colonne commune est souvent au centre gauche.
    r_start, r_end = width_ratio
    cx_start = int(w * r_start)
    cx_end = int(w * r_end)

    col_img = img[:, cx_start:cx_end]
    bbox = (cx_start, 0, cx_end, h)

    # Note: On pourrait aussi utiliser cv2.Canny et cv2.HoughLinesP 
    # pour détecter les traits verticaux exacts dessinés dans le cahier.

    return col_img, bbox

# ==============================================================================
# 2. EXTRACTION HTR (TrOCR)
# ==============================================================================
# Variables globales pour le modèle (chargement unique)
_processor = None
_model = None

def init_trocr(model_name="agomberto/trocr-large-handwritten-fr"):
    """
    Initialise le moteur HTR adapté aux écritures cursives/manuscrites.
    """
    global _processor, _model
    if _processor is None:
        print(f"[*] Chargement du modèle HTR ({model_name})...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        _model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)

def lire_texte_manuscrit(image_roi):
    """
    Applique TrOCR sur la zone isolée (ROI - Region Of Interest).
    """
    init_trocr()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TrOCR attend des images RGB
    img_rgb = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
    
    # Traitement via transformers
    pixel_values = _processor(img_rgb, return_tensors="pt").pixel_values.to(device)
    
    # Génération du texte
    generated_ids = _model.generate(
         pixel_values,
         max_new_tokens=20,
         num_beams=4,        # Beam search pour de meilleurs résultats
         early_stopping=True,
         output_scores=True,
         return_dict_in_generate=True
    )
    
    texte_lu = _processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
    
    # Calcul grossier de confiance basé sur le score de la séquence (généralement log prob)
    # Pour simuler une probabilité 0-100% :
    seq_score = float(generated_ids.sequences_scores[0].exp() * 100) if hasattr(generated_ids, 'sequences_scores') else 50.0

    return texte_lu.strip(), seq_score

# ==============================================================================
# 3 ET 4. MOTEUR DE CORRESPONDANCE ET GESTION DES ALIAS (Fuzzy Matching)
# ==============================================================================
def normaliser_chaine(texte):
    """
    Nettoie et normalise la chaîne de caractères (accents, casse, espaces).
    """
    import unicodedata
    import re
    # Enlever les accents et passer en minuscules
    texte = unicodedata.normalize('NFKD', texte).encode('ASCII', 'ignore').decode('utf-8')
    texte = texte.lower().strip()
    # Remplacer les tirets/points par des espaces pour le fuzzy match
    texte = re.sub(r'[^\w\s]', ' ', texte)
    # Compresser les espaces multiples
    texte = re.sub(r'\s+', ' ', texte)
    return texte

def valider_commune_fuzzy(texte_lu):
    """
    Compare le texte OCR à la base de données et aux alias.
    Utilise rapidfuzz (rapide et précis) pour le Fuzzy Matching.
    """
    texte_norm = normaliser_chaine(texte_lu)

    # A. Validation 1 : Gestion des ALIAS directs
    # Si la version normalisée matche exactement une clé dans notre dictionnaire d'alias
    if texte_norm in ALIAS_COMMUNES:
        return {
            "commune": ALIAS_COMMUNES[texte_norm],
            "score": 100.0,
            "methode": "Alias Exact",
            "brut": texte_lu
        }

    # B. Validation 2 : Fuzzy Matching sur la base officielle
    # On prepare une liste des communes normalisées pour la comparaison
    db_noms_norm = [normaliser_chaine(nom) for nom in DB_COMMUNES]
    
    # extractOne retourne (string_match, score, index) 
    # WRatio gère les correspondances partielles et désordonnées (utile pour les ratures)
    resultat = process.extractOne(texte_norm, db_noms_norm, scorer=fuzz.WRatio)
    
    if resultat:
        match_norm, score_fuzzy, idx = resultat
        nom_officiel = DB_COMMUNES[idx]
        return {
            "commune": nom_officiel,
            "score": round(score_fuzzy, 1),
            "methode": "Fuzzy matching",
            "brut": texte_lu
        }
    
    return {
        "commune": "Non identifiée",
        "score": 0.0,
        "methode": "Échec",
        "brut": texte_lu
    }

# ==============================================================================
# MAIN WORKFLOW : process_commune
# ==============================================================================
def process_commune(image_path, x_start=None, x_end=None):
    """
    Workflow complet :
    1. Reçoit le chemin de l'image.
    2. Découpe la colonne Commune (segmentation).
    3. Lit le texte avec TrOCR (HTR).
    4. Corrige et trouve le nom officiel (Fuzzy Matching + Alias).
    5. Retourne le résultat structuré.
    """
    print(f"\n--- Traitement de : {image_path} ---")
    try:
        # Étape 1 : Segmentation
        img_colonne, bbox = isoler_colonne_commune(image_path, x_start=x_start, x_end=x_end)
        
        # Vérifier si la zone est vide (blanche ou taille 0)
        hauteur, largeur = img_colonne.shape[:2]
        if hauteur == 0 or largeur == 0 or cv2.mean(cv2.cvtColor(img_colonne, cv2.COLOR_BGR2GRAY))[0] > 245: # Très blanc = vide
            return {"commune": "Vide", "score": 100.0, "brut": "", "methode": "Zone Vide"}

        # Étape 2 : Extraction HTR
        texte_brut, confiance_ocr = lire_texte_manuscrit(img_colonne)
        print(f"  > Lu par TrOCR : '{texte_brut}' (Confiance OCR: {confiance_ocr:.1f}%)")

        if len(texte_brut.strip()) < 2:
             return {"commune": "Illisible", "score": 0.0, "brut": texte_brut, "methode": "Texte trop court"}

        # Étapes 3 et 4 : Matching et Alias
        resultat_final = valider_commune_fuzzy(texte_brut)

        # On peut combiner la confiance OCR et le score Fuzzy pour pondérer la certitude globale
        confiance_globale = (resultat_final['score'] * 0.7) + (confiance_ocr * 0.3)

        return {
            "commune": resultat_final["commune"],
            "score": round(confiance_globale, 1),
            "score_fuzzy": resultat_final["score"],
            "methode": resultat_final["methode"],
            "brut": texte_brut,
            "bbox_colonne": bbox
        }

    except Exception as e:
        print(f"[ERREUR] Échec du pipeline pour {image_path}: {e}")
        return {"commune": "Erreur", "score": 0.0, "brut": "", "methode": "Exception"}


# Test local si exécuté directement
if __name__ == "__main__":
    # Remplacer par un chemin de test valide
    img_test = "outputs/debug_page_0.jpg"
    
    if Path(img_test).exists():
        # Exemple de découpage en fournissant des bornes horizontales brutes
        res = process_commune(img_test, x_start=250, x_end=450)
        print("Résultat final :", json.dumps(res, indent=4, ensure_ascii=False))
    else:
        print(f"Image de test introuvable ({img_test}). Pipeline prêt !")
