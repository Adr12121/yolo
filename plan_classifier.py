"""
plan_classifier.py — Extraction complète des champs par type de plan cadastral.
Types : PVa (Procès-Verbal d'arpentage), PLa (Plan de Lotissement/arpentage), DMPC
Chaque champ est retourné avec sa zone fractionnelle [x0,y0,x1,y1] pour le zoom dans app_validation.py
"""
import os, re, json, fitz, cv2
import numpy as np
import pandas as pd
import unicodedata
from typing import Dict, Any, List, Optional, Tuple

try:
    from gliner import GLiNER
    import torch
    _GLINER_AVAILABLE = True
    print("[PlanClassifier] ✅ GLiNER est installé et sera utilisé pour l'extraction sémantique.")
except ImportError:
    _GLINER_AVAILABLE = False
    print("[PlanClassifier] ⚠️ GLiNER n'est pas installé. Fallback sur les expressions régulières.")

_gliner_model = None
def get_gliner_model():
    global _gliner_model
    if _gliner_model is None and _GLINER_AVAILABLE:
        print("  [PlanClassifier] 🧠 Chargement du modèle sémantique GLiNER...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # On utilise une version small, très rapide et légère
        try:
            _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1").to(device)
            print("  [PlanClassifier] ✨ Modèle GLiNER chargé avec succès.")
        except Exception as e:
            print(f"  [PlanClassifier] ❌ Erreur lors du chargement de GLiNER : {e}")
            return None
    return _gliner_model

# ── Couleurs annotation ────────────────────────────────────────────
COLORS = {
    "commune":    (0, 180, 90),
    "n_ordre":    (30, 120, 200),
    "section":    (255, 140, 0),
    "feuille":    (0, 200, 200),
    "date":       (180, 60, 60),
    "echelle":    (0, 180, 200),
    "geometre":   (150, 0, 200),
    "signataires":(200, 80, 0),
    "parcelles":  (180, 180, 0),
    "proprietaires_anciens": (0, 100, 160),
    "proprietaires_nouveaux":(0, 160, 100),
    "indication": (100, 100, 100),
    "n_dossier":  (30, 180, 120),
}

# ── Mots-clés de recherche par champ (ordre : plus long d'abord) ──
KEYWORDS: Dict[str, List[str]] = {
    "commune": [
        "commune de", "commune:", "territoire de", "sur la commune",
        "en la commune", "c.n.e.", "commune", "territoire",
    ],
    "n_ordre": [
        "numéro d'ordre", "n° d'ordre", "numéro d arpentage",
        "n° d'arpentage", "document d'arpentage", "da n°", "da:",
        "n° dossier", "dossier n°", "affaire n°", "référence",
        "n° ordre", "numéro ordre",
    ],
    "n_dossier": [
        "dossier", "référence", "n°",
    ],
    "section": [
        "section cadastrale", "section n°", "section:", "section", "sect.",
    ],
    "feuille": [
        "feuille cadastrale", "feuille n°", "feuille:", "feuille",
    ],
    "date": [
        "établi le", "dressé le", "édité le", "signé le",
        "en date du", "fait à", "fait le", "date:",
    ],
    "echelle": [
        "à l'échelle de", "échelle:", "échelle", "echelle:", "echelle", "ech.",
    ],
    "geometre": [
        "géomètre-expert", "géomètre expert", "geometre expert",
        "cabinet de géomètre", "le soussigné géomètre",
        "géomètre:", "cabinet:", "bureau d'études",
    ],
    "signataires": [
        "certifié exact", "signataire", "signatures", "vu et approuvé",
        "le géomètre expert soussigné", "soussigné", "certifié",
    ],
    "proprietaires_anciens": [
        "anciens propriétaires", "ancien propriétaire",
        "propriétaire sortant", "vendeur", "cédant", "ci-devant",
    ],
    "proprietaires_nouveaux": [
        "nouveaux propriétaires", "nouveau propriétaire",
        "propriétaire entrant", "acquéreur", "bénéficiaire",
    ],
    "indication": [
        "piquetage contradictoire", "piquetage", "bornage contradictoire",
        "reconnaissance de limites", "nature des opérations",
        "objet du document", "objet:",
    ],
}

# ── Zones de recherche spatialisées selon le type de plan ─────────
# Format: [x0_frac, y0_frac, x1_frac, y1_frac] (fractions de la page)
ZONES_PAR_TYPE: Dict[str, Dict[str, List[float]]] = {
    "PVa": {   # Procès-Verbal d'arpentage — en-tête haut gauche
        "commune":    [0.0, 0.0, 0.6, 0.25],
        "n_ordre":    [0.0, 0.0, 0.6, 0.25],
        "section":    [0.0, 0.0, 0.6, 0.35],
        "feuille":    [0.0, 0.0, 0.6, 0.35],
        "date":       [0.0, 0.6, 1.0, 1.0],
        "echelle":    [0.5, 0.0, 1.0, 0.35],
        "geometre":   [0.4, 0.6, 1.0, 1.0],
        "signataires":[0.4, 0.6, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.2, 0.55, 0.6],
        "proprietaires_nouveaux": [0.0, 0.2, 0.55, 0.6],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],   # partout (rouge)
        "indication": [0.0, 0.0, 0.6, 0.4],
    },
    "PLa": {   # Plan de Lotissement / document d'arpentage classique
        "commune":    [0.0, 0.0, 0.65, 0.30],
        "n_ordre":    [0.0, 0.0, 0.65, 0.30],
        "section":    [0.0, 0.0, 0.65, 0.40],
        "feuille":    [0.0, 0.0, 0.65, 0.40],
        "date":       [0.0, 0.65, 1.0, 1.0],
        "echelle":    [0.5, 0.0, 1.0, 0.40],
        "geometre":   [0.4, 0.6, 1.0, 1.0],
        "signataires":[0.4, 0.6, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.25, 0.6, 0.65],
        "proprietaires_nouveaux": [0.0, 0.25, 0.6, 0.65],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 0.65, 0.4],
    },
    "DMPC": {   # Document Modificatif du Parcellaire Cadastral
        "commune":    [0.0, 0.0, 0.55, 0.20],
        "n_ordre":    [0.5, 0.0, 1.0, 0.20],
        "n_dossier":  [0.5, 0.0, 1.0, 0.20],
        "section":    [0.0, 0.0, 0.55, 0.30],
        "feuille":    [0.0, 0.0, 0.55, 0.30],
        "date":       [0.0, 0.70, 1.0, 1.0],
        "echelle":    [0.5, 0.0, 1.0, 0.30],
        "geometre":   [0.4, 0.65, 1.0, 1.0],
        "signataires":[0.4, 0.65, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.2, 0.55, 0.65],
        "proprietaires_nouveaux": [0.0, 0.2, 0.55, 0.65],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 0.55, 0.35],
    },
    "GENERIC": {   # Fallback
        "commune":    [0.0, 0.0, 0.60, 0.30],
        "n_ordre":    [0.0, 0.0, 0.60, 0.30],
        "section":    [0.0, 0.0, 0.60, 0.40],
        "feuille":    [0.0, 0.0, 0.60, 0.40],
        "date":       [0.0, 0.60, 1.0, 1.0],
        "echelle":    [0.5, 0.0, 1.0, 0.40],
        "geometre":   [0.4, 0.60, 1.0, 1.0],
        "signataires":[0.4, 0.60, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.25, 0.6, 0.65],
        "proprietaires_nouveaux": [0.0, 0.25, 0.6, 0.65],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 0.60, 0.40],
    },
}


# ── Normalisation texte ────────────────────────────────────────────
def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()

# ── Validation métier par champ ────────────────────────────────────
def _validate_field(field_type: str, text: str) -> Optional[str]:
    """Valide et nettoie la valeur d'un champ selon des règles métiers strictes."""
    val = text.strip()
    if not val or len(val) < 1:
        return None
        
    val_norm = val.lower()
    
    # Mots interdits qui indiquent qu'on a lu le label suivant au lieu de la valeur
    INTERDITS = ["lieudit", "lieu dit", "contenance", "surface", "propriétaire", 
                 "demandeur", "document", "plan de", "procès", "section", "échelle"]
                 
    for interdit in INTERDITS:
        if interdit in val_norm:
            return None

    if field_type == "section":
        # Une section c'est typiquement 1 à 4 lettres majuscules (A, ZD, AB...)
        # ou parfois un chiffre. On nettoie tout ce qui n'est pas alphanum
        clean = re.sub(r"[^A-Za-z0-9]", "", val).upper()
        if len(clean) > 0 and len(clean) <= 4:
            # Remplacement des erreurs d'OCR courantes
            clean = clean.replace("0", "O").replace("1", "I").replace("8", "B")
            return clean
        return None
        
    elif field_type == "echelle":
        # Doit ressembler à "1/500", "500", "1:1000", etc.
        m = re.search(r'(1\s*[/:,\.]\s*\d{3,4}|\d{3,4})', val)
        if m:
            return m.group(1).replace(" ", "").replace(".", "/").replace(",", "/")
        return None
        
    elif field_type in ["n_ordre", "n_dossier"]:
        # DA ou dossier (ex: 2024-015, A094147, 12345A)
        # Blacklist de mots très courants qui ne sont jamais des numéros de dossier
        BAD_WORDS = ["aupres", "avec", "pour", "dans", "fait", "le", "la", "les", "des", "du", "objet", "limite", "faire"]
        for bad in BAD_WORDS:
            if bad in val_norm:
                return None
        
        # S'il y a des unités de mesure, on rejette
        if any(u in val_norm for u in [" ha", " a ", " ca", "m2", "cm"]):
            return None 

        # On garde si ça ressemble à une ref alphanum avec au moins UN chiffre
        # car un numéro de dossier sans chiffre est suspect
        clean = re.sub(r"[^A-Za-z0-9\-\/]", "", val)
        if len(clean) >= 2 and len(clean) <= 15:
            if re.search(r'\d', clean): # Doit contenir au moins un chiffre
                return val
        return None
        
    elif field_type == "date":
        # DD/MM/YYYY ou 26 mai 2009
        m = re.search(r'(\d{1,2}\s*[/\-\.]\s*\d{1,2}\s*[/\-\.]\s*\d{2,4}|\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+\d{4})', val_norm)
        if m:
            return val
        return None
        
    # Pour commune, geometre, signataires, etc., on fait juste un filtre sur la taille
    if len(val) > 100:
        return None
        
    return val



# ── Détection du type depuis le nom de fichier ────────────────────
def classify_plan(filepath: str) -> str:
    name = os.path.basename(filepath).lower()
    if re.search(r'pva|1pva|pvb|pvc|proc[eè]s.?verbal', name):
        return "PVa"
    if re.search(r'2pla|1pla|pla|lotissement|division', name):
        return "PLa"
    if re.search(r'dmpc|geofoncier_dmpc', name):
        return "DMPC"
    # Lecture rapide du texte PDF pour confirmer
    try:
        doc = fitz.open(filepath)
        txt = doc[0].get_text("text").upper()
        doc.close()
        if "PROCÈS-VERBAL" in txt or "PROCES-VERBAL" in txt or "D'ARPENTAGE" in txt:
            return "PVa"
        if "DMPC" in txt or "DOCUMENT MODIFICATIF" in txt:
            return "DMPC"
        if "LOTISSEMENT" in txt or "DIVISION" in txt:
            return "PLa"
    except Exception:
        pass
    return "GENERIC"


def is_plan_document(filepath: str) -> bool:
    return True   # Tous les fichiers dans inputs/ sont des plans


# ── Extraction OCR dans une zone fractionnelle ────────────────────
def _ocr_in_zone(reader, img: np.ndarray, zone: List[float]) -> List[Tuple[Any, str, float]]:
    """OCR EasyOCR sur une sous-région de l'image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(zone[0]*w), int(zone[1]*h), int(zone[2]*w), int(zone[3]*h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return []
    crop = img[y0:y1, x0:x1]
    results = reader.readtext(crop)
    # Remettre les bbox en coordonnées absolues
    shifted = []
    for (bbox, text, prob) in results:
        abs_bbox = [[pt[0]+x0, pt[1]+y0] for pt in bbox]
        shifted.append((abs_bbox, text, prob))
    return shifted

# ── Extraction Sémantique (Intelligence Artificielle) ────────────
GLINER_LABELS_MAP = {
    "commune": "nom de commune",
    "section": "section cadastrale",
    "n_ordre": "numéro de document d'arpentage DA",
    "n_dossier": "numéro de référence dossier",
    "date": "date",
    "echelle": "échelle du plan",
    "geometre": "nom du géomètre expert",
    "proprietaires_anciens": "ancien propriétaire cédant",
    "proprietaires_nouveaux": "nouveau propriétaire acquéreur",
    "indication": "objet du document bornage"
}

def _semantic_extract(field_type: str, ocr_results: List[Tuple], img_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """Utilise un VLM/NLP pour lire le texte et comprendre où est l'information."""
    model = get_gliner_model()
    if not model: return None
    
    items = []
    for (bbox, text, prob) in ocr_results:
        if prob < 0.15: continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({"text": text, "box": [min(xs), min(ys), max(xs), max(ys)]})
        
    if not items: return None
    
    full_text = " | ".join([it["text"] for it in items])
    label_to_find = GLINER_LABELS_MAP.get(field_type)
    if not label_to_find: return None
    
    # On demande à l'IA de trouver l'entité
    entities = model.predict_entities(full_text, [label_to_find], threshold=0.15)
    if not entities: return None
    
    # On prend la prédiction la plus confiante
    best_ent = max(entities, key=lambda x: x["score"])
    val = best_ent["text"].strip()
    
    # On passe quand même par la validation métier pour être sûr
    validated = _validate_field(field_type, val)
    if not validated: return None
    
    # Retrouver la zone d'origine pour le zoom Streamlit
    h, w = img_shape
    best_box = [0, 0, w, h] # Fallback : pleine image
    raw_source = validated
    for it in items:
        if validated.lower() in it["text"].lower() or it["text"].lower() in validated.lower():
            best_box = it["box"]
            raw_source = it["text"]
            break
            
    zone_f = [best_box[0]/w, best_box[1]/h, best_box[2]/w, best_box[3]/h]
    return {"valeur": validated, "zone": zone_f, "brut": raw_source}


# ── Recherche d'un champ par mots-clés dans l'OCR d'une zone ─────
def _find_field(
    field_type: str,
    ocr_results: List[Tuple],
    keywords: List[str],
    img_shape: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    """
    Parcourt les résultats OCR pour trouver le label puis capture la valeur.
    Gère les cas avec séparateurs (:) et sans séparateurs (Section C).
    """
    h, w = img_shape

    items = []
    for (bbox, text, prob) in ocr_results:
        if prob < 0.15:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({"text": text, "norm": _norm(text),
                      "bx0": min(xs), "by0": min(ys), "bx1": max(xs), "by1": max(ys)})

    for kw in sorted(keywords, key=len, reverse=True):
        kw_norm = _norm(kw)
        
        for i, item in enumerate(items):
            if kw_norm not in item["norm"]:
                continue
            
            # 1. Tentative: Valeur sur la même ligne
            raw = item["text"]
            val_candidate = ""
            
            # Cas A: Séparateur présent (ex: "Section : C")
            parts = re.split(r'[:–\-]', raw, maxsplit=1)
            if len(parts) > 1:
                # On vérifie si le mot clé est avant le séparateur
                if kw_norm in _norm(parts[0]):
                    val_candidate = parts[1].strip()
            
            # Cas B: Pas de séparateur (ex: "Section C")
            if not val_candidate:
                # On retire le mot clé du texte pour voir ce qu'il reste
                # On cherche la position du kw dans le texte original (insensible à la casse)
                match = re.search(re.escape(kw), raw, re.IGNORECASE)
                if match:
                    val_candidate = raw[match.end():].strip()
                    # Si le reste commence par un séparateur qu'on aurait raté
                    val_candidate = re.sub(r'^[:–\-\s\.]+', '', val_candidate)

            if val_candidate:
                # Si la valeur contient un autre label (ex: "Section C - Lieudit"), on coupe au prochain label
                for other_ks in KEYWORDS.values():
                    for okw in other_ks:
                        if okw.lower() != kw.lower():
                            m_other = re.search(r'\b' + re.escape(okw) + r'\b', val_candidate, re.IGNORECASE)
                            if m_other:
                                val_candidate = val_candidate[:m_other.start()].strip()

                validated = _validate_field(field_type, val_candidate)
                if validated:
                    zone_f = [item["bx0"]/w, item["by0"]/h, item["bx1"]/w, item["by1"]/h]
                    return {"valeur": validated, "zone": zone_f, "brut": raw}
                    
            # 2. Tentative: Valeur dans les items suivants
            for j in range(i+1, min(i+5, len(items))):
                nxt = items[j]
                y_dist = nxt["by0"] - item["by1"]
                if y_dist < -15 or y_dist > 150: break
                
                if any(_norm(k2) in nxt["norm"] for k2 in [kk for ks in KEYWORDS.values() for kk in ks]):
                    break
                
                validated = _validate_field(field_type, nxt["text"].strip())
                if validated:
                    bx0f = min(item["bx0"], nxt["bx0"]) / w
                    by0f = item["by0"] / h
                    bx1f = max(item["bx1"], nxt["bx1"]) / w
                    by1f = nxt["by1"] / h
                    return {"valeur": validated, "zone": [bx0f, by0f, bx1f, by1f], "brut": f"{raw} | {nxt['text']}"}
            break
    return None



# ── Matching commune contre la base ──────────────────────────────
def _match_commune(text: str, commune_db: Optional[List[Dict]]) -> str:
    if not text or not commune_db:
        return text
    try:
        from rapidfuzz import process as rfp, fuzz
    except ImportError:
        return text

    def norm_c(t):
        nfkd = unicodedata.normalize("NFKD", str(t))
        s = "".join(c for c in nfkd if not unicodedata.combining(c))
        s = re.sub(r"[-''`]", " ", s)
        return re.sub(r"[^A-Z0-9 ]", " ", s.upper()).strip()

    text_n = norm_c(text)
    noms = [norm_c(e["officiel"]) for e in commune_db]
    result = rfp.extractOne(text_n, noms, scorer=fuzz.WRatio)
    if result and result[1] >= 55:
        return commune_db[result[2]]["officiel"]
    return text


# ── Pipeline principal ────────────────────────────────────────────
def process_plan(pdf_path: str, models=None, commune_db=None) -> dict:
    print(f"  [PlanClassifier] Traitement : {os.path.basename(pdf_path)}")
    reader = models[1] if models and len(models) > 1 else None
    if reader is None:
        print("  [PlanClassifier] Pas de reader OCR disponible.")
        return {"fichier": pdf_path, "type_plan": "GENERIC", "pages": [], "skipped": False}

    type_plan = classify_plan(pdf_path)
    zones_def = ZONES_PAR_TYPE.get(type_plan, ZONES_PAR_TYPE["GENERIC"])
    print(f"  [PlanClassifier] Type détecté : {type_plan}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [PlanClassifier] Erreur ouverture PDF : {e}")
        return {"fichier": pdf_path, "skipped": True, "raison": str(e)}

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    pages_out = []

    for pi, page in enumerate(doc):
        page_num = pi + 1
        print(f"  [PlanClassifier] Page {page_num}...")

        # Rendu haute résolution
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        champs: Dict[str, Any] = {}

        # ── 1. OCR par zone pour chaque champ ──────────────────────
        for field, zone in zones_def.items():
            if field == "parcelles":
                continue   # Traité séparément (détection rouge)
            kws = KEYWORDS.get(field, [])
            if not kws:
                continue
            # OCR dans la zone du champ
            ocr_zone = _ocr_in_zone(reader, img_bgr, zone)
            
            result = None
            
            # 1. Tentative avec Intelligence Artificielle Sémantique
            if _GLINER_AVAILABLE:
                result = _semantic_extract(field, ocr_zone, (h, w))
                if result:
                    print(f"    [{field}] → '{result['valeur']}' (Via IA GLiNER)")
                    
            # 2. Fallback avec Expressions Régulières si pas d'IA ou IA a échoué
            if not result:
                kws = KEYWORDS.get(field, [])
                if kws:
                    result = _find_field(field, ocr_zone, kws, (h, w))
                    if result:
                        print(f"    [{field}] → '{result['valeur']}' (Via Regex)")

            if result:
                val = result["valeur"]
                # Matching commune si besoin
                if field == "commune" and commune_db:
                    val_matched = _match_commune(val, commune_db)
                    result["valeur_brute"] = val
                    result["valeur"] = val_matched
                    result["confiance_match"] = 1.0 if val_matched != val else 0.7
                champs[field] = {
                    "valeur": result["valeur"],
                    "zone": result["zone"],
                    "brut": result.get("brut", val),
                }
                print(f"    [{field}] → '{result['valeur']}'")

        # ── 2. Détection parcelles (texte rouge dans l'image) ──────
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 80, 80), (12, 255, 255)),
            cv2.inRange(hsv, (158, 80, 80), (180, 255, 255)),
        )
        # Dilater légèrement pour agglomérer
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_red = cv2.dilate(mask_red, kernel, iterations=1)

        red_ocr = reader.readtext(mask_red)
        parcelles = [r[1] for r in red_ocr if len(r[1].strip()) >= 1 and r[2] >= 0.20]

        # Zone englobante des parcelles trouvées
        if red_ocr:
            all_xs = [p[0] for r in red_ocr for p in r[0]]
            all_ys = [p[1] for r in red_ocr for p in r[0]]
            parc_zone = [min(all_xs)/w, min(all_ys)/h, max(all_xs)/w, max(all_ys)/h]
        else:
            parc_zone = [0.0, 0.0, 1.0, 1.0]

        champs["parcelles"] = {"valeur": parcelles, "zone": parc_zone}

        # ── 3. Si commune non trouvée, fallback zone large ─────────
        if "commune" not in champs:
            zone_large = zones_def.get("commune", [0.0, 0.0, 0.6, 0.3])
            ocr_large = _ocr_in_zone(reader, img_bgr, zone_large)
            
            result_c = None
            if _GLINER_AVAILABLE:
                result_c = _semantic_extract("commune", ocr_large, (h, w))
            if not result_c:
                result_c = _find_field("commune", ocr_large, KEYWORDS["commune"], (h, w))
                
            if result_c and commune_db:
                val_m = _match_commune(result_c["valeur"], commune_db)
                champs["commune"] = {
                    "valeur": val_m,
                    "zone": result_c["zone"],
                    "brut": result_c.get("brut", ""),
                }
            elif result_c:
                champs["commune"] = {
                    "valeur": result_c["valeur"],
                    "zone": result_c["zone"],
                    "brut": result_c.get("brut", ""),
                }

        # ── 4. Image annotée ────────────────────────────────────────
        ann = img_bgr.copy()
        # Cadres bleus pour tout le texte OCR (debug)
        all_ocr = reader.readtext(img_bgr)
        for (bbox, text, prob) in all_ocr:
            pts = np.array(bbox, np.int32)
            cv2.polylines(ann, [pts], True, (200, 200, 200), 1)
            xt, yt = int(bbox[0][0]), max(int(bbox[0][1]) - 3, 10)
            cv2.putText(ann, text[:25], (xt, yt), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (120, 120, 120), 1, cv2.LINE_AA)

        # Cadres colorés pour les champs détectés
        for field, info in champs.items():
            if not isinstance(info, dict) or "zone" not in info:
                continue
            z = info["zone"]
            if len(z) != 4:
                continue
            x0p = int(z[0]*w); y0p = int(z[1]*h)
            x1p = int(z[2]*w); y1p = int(z[3]*h)
            color = COLORS.get(field, (128, 128, 128))
            cv2.rectangle(ann, (x0p, y0p), (x1p, y1p), color, 2)
            val_str = info.get("valeur", "")
            if isinstance(val_str, list):
                val_str = ", ".join(str(v) for v in val_str[:3])
            tag = f"{field}: {str(val_str)[:28]}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ty = max(y0p - 5, th + 4)
            overlay = ann.copy()
            cv2.rectangle(overlay, (x0p, ty-th-3), (x0p+tw+6, ty+3), color, -1)
            cv2.addWeighted(overlay, 0.55, ann, 0.45, 0, ann)
            cv2.putText(ann, tag, (x0p+3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(output_dir, f"{base_name}_p{page_num}_annote.jpg")
        cv2.imwrite(out_path, ann)
        print(f"  [PlanClassifier] Image annotée : {out_path}")

        pages_out.append({"page": page_num, "champs": champs})

    doc.close()
    return {
        "fichier": pdf_path,
        "type_plan": type_plan,
        "pages": pages_out,
    }


def export_plan_to_csv(res: dict, output_dir: str = "outputs") -> str:
    base = os.path.splitext(os.path.basename(res["fichier"]))[0]
    csv_path = os.path.join(output_dir, f"{base}_plan_resultats.csv")
    rows = []
    for pg in res.get("pages", []):
        champs = pg.get("champs", {})
        def g(f):
            v = champs.get(f, {})
            if isinstance(v, dict):
                val = v.get("valeur", "")
                return ", ".join(val) if isinstance(val, list) else str(val)
            return ""
        rows.append({
            "ID": f"p{pg['page']}",
            "Page": pg["page"],
            "Type_Plan": res.get("type_plan", ""),
            "Commune": g("commune"),
            "N_Ordre": g("n_ordre"),
            "N_Dossier": g("n_dossier"),
            "Section": g("section"),
            "Feuille": g("feuille"),
            "Date": g("date"),
            "Echelle": g("echelle"),
            "Geometre": g("geometre"),
            "Signataires": g("signataires"),
            "Proprietaires_Anciens": g("proprietaires_anciens"),
            "Proprietaires_Nouveaux": g("proprietaires_nouveaux"),
            "Parcelles": g("parcelles"),
            "Indication": g("indication"),
            "Confirmation_Status": "À valider",
        })
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"  [PlanClassifier] CSV exporté : {csv_path}")
    return csv_path
