"""
modern_plan_extractor.py
========================
Pipeline dédié aux plans cadastraux modernes (1980–2010+).

Ces documents sont principalement dactylographiés / imprimés (typographie
Leroy ou similaire pour les anciens, puis Word/LibreOffice pour les récents).
Leur layout est structuré sous forme de paires LABEL : VALEUR disposées
sur la même ligne ou sur deux lignes consécutives.

Champs extraits (conformément à la demande) :
  - commune
  - section              (ex : B, ZD)
  - n_ordre              (numéro d'ordre du DA/DMPC)
  - proprietaires_anciens
  - proprietaires_nouveaux
  - parcelles            (liste)
  - signataires
  - indication           (piquetage contradictoire, bureau, etc.)
  - geometre             (nom du géomètre-expert)
  - echelle
  - date

Architecture :
  1. Tesseract PSM 6 (colonne de texte uniforme) → TSV → paires label/valeur
  2. SemanticFieldCorrector → validation et correction de chaque champ
  3. CorrectionLearner → injection des corrections apprises
  4. Export vers CSV/JSON/Excel compatibles avec app_validation.py

Usage depuis main.py :
    from modern_plan_extractor import process_modern_plan
    results = process_modern_plan(file_path, models, commune_db, geometre_id)
"""

import os
import re
import json
import subprocess
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import fitz  # PyMuPDF  # type: ignore
from PIL import Image  # type: ignore


# ---------------------------------------------------------------------------
# 0. CONSTANTES DOCUMENTS DGFIP RECENTS
# ---------------------------------------------------------------------------

# Signature texte identifiant les documents DGFIP recents (extrait cadastral modifié)
DGFIP_SIGNATURE = "DIRECTION GENERALE DES FINANCES PUBLIQUES"

# Blacklist : phrases boilerplate du formulaire DGFIP à NE PAS extraire
# comme propriétaires/signataires/géomètre
_DGFIP_BOILERPLATE = {
    "déclarent avoir pris connaissance des informations portées",
    "au dos de la chemise 6463",
    "s'il est différent du propriétaire",
    "mandataire",
    "avoué",
    "représentant qualifié de l'autorité expropriant",
    "le présent document d'arpentage",
    "certifié par les propriétaires soussignés",
    "a été établi",
    "d'après les indications",
    "en conformité d'un piquetage",
    "effectué sur le terrain",
    "d'après un plan d'arpentage",
    "modification selon les énonciations",
    "art. 25 du décret",
    "inspection des finances",
    "inspecteur des finances",
    "certification",
    "sdif",
    "ptgc",
    "a ptgc",
    "signé",
}

# ---------------------------------------------------------------------------
# 1. LABELS CONNUS DANS LES PLANS MODERNES (multi-variantes)
# ---------------------------------------------------------------------------

PLAN_LABELS: Dict[str, List[str]] = {
    "commune": [
        "commune", "commune de", "c.n.e.", "territoire de", "commune :",
        "sur la commune", "en la commune"
    ],
    "section": [
        "section", "section cadastrale", "feuille cadastrale", "sect.",
        "section :", "section n°"
    ],
    "n_ordre": [
        "n° d'ordre", "numéro d'ordre", "n° ordre", "numéro ordre",
        "référence", "n° dossier", "dossier n°", "affaire n°",
        "n°", "ordre", "document d'arpentage", "d'arpentage"
    ],
    "proprietaires_anciens": [
        "ancien propriétaire", "anciens propriétaires", "propriétaire sortant",
        "vendeur", "cédant", "ci-devant", "anciens prop", "anc. prop"
    ],
    "proprietaires_nouveaux": [
        "nouveau propriétaire", "nouveaux propriétaires", "propriétaire entrant",
        "acquéreur", "acheteur", "bénéficiaire", "nouv. prop", "propriétaires"
    ],
    "parcelles": [
        "parcelle", "parcelles", "n° parcelle", "numéro de parcelle",
        "lot", "lots", "parcelle(s)", "les parcelles", "n° part"
    ],
    "signataires": [
        "signé", "signataire", "signatures", "le géomètre", "certifié",
        "approuvé", "vu et approuvé", "soussigné", "soussignés"
    ],
    "indication": [
        "piquetage", "piquetage contradictoire", "contradictoire",
        "bornage", "reconnaissance", "bureau", "issu de", "suite à",
        "opérations effectuées", "objet", "nature des opérations",
        "type d'opération", "présent document"
    ],
    "geometre": [
        "géomètre-expert", "géomètre expert", "geometre expert",
        "cabinet de géomètre", "le soussigné géomètre", "géomètre :",
        "cabinet", "bureau d'études", "m.", "mr.", "mme"
    ],
    "echelle": [
        "échelle", "echelle", "ech.", "ech :", "à l'échelle de", "1/"
    ],
    "date": [
        "fait à", "dressé le", "établi le", "signé le", "le ", "date",
        "en date du", "à", "fait en"
    ],
}

# ---------------------------------------------------------------------------
# 2. EXTRACTION — Stratégie double : PyMuPDF (vectoriel) + Tesseract (scan)
# ---------------------------------------------------------------------------

def _extract_via_pymupdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extraction directe via PyMuPDF — fiable pour PDFs vectoriels (texte natif).
    Retourne une liste de spans : {page, text, bbox=[x1,y1,x2,y2], conf}
    où bbox est en pixels à 150 DPI (scale factor 150/72).
    """
    SCALE = 150 / 72  # DPI de travail → cohérent avec _pdf_to_images(dpi=150)
    spans_out: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(file_path)
        for page_idx, page in enumerate(doc):
            blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = " ".join(
                        sp.get("text", "") for sp in line.get("spans", [])
                    ).strip()
                    if not line_text:
                        continue
                    # bbox de la ligne entière
                    x0, y0, x1, y1 = line["bbox"]
                    spans_out.append({
                        "page":  page_idx + 1,
                        "text":  line_text,
                        "bbox":  [int(x0*SCALE), int(y0*SCALE),
                                  int(x1*SCALE), int(y1*SCALE)],
                        "conf":  92,  # texte vectoriel = très fiable
                    })
        doc.close()
    except Exception as e:
        print(f"  [ModernPlan] PyMuPDF extraction erreur: {e}")
    return spans_out


def _run_ocr_fallback(img_gray: np.ndarray) -> pd.DataFrame:
    """Fallback OCR using EasyOCR."""
    try:
        import easyocr # type: ignore
        # On instancie le reader (peut être lent au premier appel, mais mis en cache par EasyOCR)
        reader = easyocr.Reader(['fr'], gpu=False)
        results = reader.readtext(img_gray)
        rows = []
        for i, (bbox, text, prob) in enumerate(results):
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x1, y1 = int(min(x_coords)), int(min(y_coords))
            x2, y2 = int(max(x_coords)), int(max(y_coords))
            rows.append({
                "block_num": i, "par_num": 0, "line_num": i, "word_num": 0,
                "left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1,
                "text": text, "conf": int(prob * 100),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  [ModernPlan] EasyOCR erreur: {e}")
        return pd.DataFrame()


def _pdf_to_images(file_path: str, dpi: int = 150) -> List[np.ndarray]:
    """Convertit un PDF en liste d'images OpenCV (BGR)."""
    images = []
    try:
        doc = fitz.open(file_path)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8)
            img_np = img_np.reshape(pix.height, pix.width, 3)
            images.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        doc.close()
    except Exception as e:
        print(f"  [ModernPlan] Erreur lecture PDF: {e}")
    return images


def _spans_to_tess_df(spans: List[Dict]) -> pd.DataFrame:
    """Convertit les spans PyMuPDF en DataFrame compatible parse_label_value_pairs."""
    if not spans:
        return pd.DataFrame()
    rows = []
    for i, sp in enumerate(spans):
        x1, y1, x2, y2 = sp["bbox"]
        rows.append({
            "block_num": i, "par_num": 0, "line_num": i, "word_num": 0,
            "left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1,
            "text": sp["text"], "conf": sp["conf"],
        })
    return pd.DataFrame(rows)


def _generate_annotated_image(
    img_bgr: np.ndarray,
    champs: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Génère et sauvegarde une image annotée du plan avec les champs détectés
    encadrés par des rectangles colorés et légendés.
    """
    COLORS = {
        "commune":                (0,   180,  90),   # vert
        "section":                (255, 140,   0),   # orange
        "n_ordre":                (30,  120, 200),   # bleu
        "geometre":               (150,   0, 200),   # violet
        "echelle":                (0,   180, 200),   # cyan
        "date":                   (180,  60,  60),   # rouge foncé
        "parcelles":              (180, 180,   0),   # jaune
        "proprietaires_anciens":  (0,   100, 160),   # bleu acier
        "proprietaires_nouveaux": (0,   160, 100),   # vert teal
        "signataires":            (200,  80,   0),   # orange foncé
        "indication":             (100, 100, 100),   # gris
    }
    LABELS_FR = {
        "commune": "Commune", "section": "Section", "n_ordre": "N° ordre",
        "geometre": "Géomètre", "echelle": "Échelle", "date": "Date",
        "parcelles": "Parcelles", "proprietaires_anciens": "Prop. Anciens",
        "proprietaires_nouveaux": "Prop. Nouveaux", "signataires": "Signataires",
        "indication": "Indication",
    }

    out = img_bgr.copy()
    
    # 1. Dessiner d'abord TOUTES les détections brutes (en fond, gris clair)
    raw_detections = champs.pop('_raw_detections', [])
    for det in raw_detections:
        bbox = det.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        rx1, ry1, rx2, ry2 = [int(v) for v in bbox]
        r_text = det.get('texte', '')[:30]
        
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (200, 200, 200), 1)
        # On affiche le texte brut très discrètement au-dessus
        (tw, th), _ = cv2.getTextSize(r_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        r_ty = max(ry1 - 2, th + 2)
        cv2.putText(out, r_text, (rx1, r_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    # 2. Dessiner ensuite les champs identifiés (en couleur, bien visibles)
    for field, info in champs.items():
        if not isinstance(info, dict):
            continue
        bbox = info.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS.get(field, (128, 128, 128))

        # Rectangle de détection
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Étiquette
        label = LABELS_FR.get(field, field)
        val_str = info.get("valeur", "")
        tag = f"{label}: {str(val_str)[:30]}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 5, th + 4)
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), color, -1)
        cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
        cv2.putText(out, tag, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, out)
    print(f"  [ModernPlan] Image annotée : {output_path}")



# ---------------------------------------------------------------------------
# 3. PARSEUR DE PAIRES LABEL/VALEUR
# ---------------------------------------------------------------------------

def _normalize_label(text: str) -> str:
    """Normalise un texte pour comparaison avec les labels."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()
    

def _identify_label(text: str) -> Optional[str]:
    """Identifie le type de champ depuis un texte de label."""
    norm = _normalize_label(text)
    for field_type, keywords in PLAN_LABELS.items():
        for kw in sorted(keywords, key=len, reverse=True):
            kw_norm = _normalize_label(kw)
            if not kw_norm: continue
            
            # Si le mot-clé normalisé est très court (ex: 'n' pour 'n°', '1' pour '1/'), 
            # il doit apparaître comme un mot isolé pour éviter les faux positifs massifs.
            if len(kw_norm) <= 2:
                # Vérifie les frontières de mots
                if re.search(r'\b' + re.escape(kw_norm) + r'\b', norm):
                    return field_type
            else:
                if kw_norm in norm:
                    return field_type
    return None


def _extract_inline_value(text: str, field_type: str) -> Optional[str]:
    """
    Extrait la valeur sur la même ligne en cherchant le texte APRÈS le label.
    
    Stratégie :
    1. Localiser la position du label dans la ligne
    2. Prendre le texte qui se trouve APRÈS ce label (+ séparateur optionnel)
    3. Nettoyer les séparateurs résiduels
    
    Ex: "Section C – commune: Ucel – Lieudit" avec label="commune"
        → cherche "commune" dans le texte → prend ": Ucel – Lieudit"
        → split sur "–" → prend "Ucel" ✅
    """
    import re
    
    text_lower = text.lower()
    
    # Trouver la position du label le plus long qui matche (le plus spécifique en premier)
    best_kw_end = -1
    for kw in sorted(PLAN_LABELS.get(field_type, []), key=len, reverse=True):
        kw_lower = kw.lower()
        idx = text_lower.find(kw_lower)
        if idx != -1:
            kw_end = idx + len(kw)
            if kw_end > best_kw_end:
                best_kw_end = kw_end
                break  # prendre le plus long match trouvé
    
    if best_kw_end == -1:
        return None  # label non trouvé dans la ligne
    
    # Texte après le label
    after_label = text[best_kw_end:].strip()
    
    # Retirer les séparateurs de début (:, –, -, ., espaces)
    after_label = re.sub(r'^[\s:;\-–\|\.]+', '', after_label).strip()
    
    if not after_label or len(after_label) < 2:
        return None
    
    # Si la valeur contient encore un séparateur fort (–, |) → prendre jusqu'au 1er
    # (ex: "Ucel – Lieudit Montagne" → prendre "Ucel" seulement)
    first_sep = re.search(r'[–|]|\s{3,}', after_label)
    if first_sep and first_sep.start() > 1:
        candidate = after_label[:first_sep.start()].strip()
        if len(candidate) > 1:
            return candidate
    
    return after_label


def parse_label_value_pairs(
    tess_df: pd.DataFrame,
    img_h: int,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse les paires label/valeur depuis le TSV Tesseract.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    if tess_df is None or (hasattr(tess_df, 'empty') and tess_df.empty):
        return results

    # Filtrer les lignes valides
    df = tess_df.copy()
    df = df[df['conf'].apply(lambda x: str(x).lstrip('-').isdigit())]
    df['conf'] = df['conf'].astype(int)
    df = df[df['conf'] >= 20]
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']

    if df.empty:
        return results

    # Grouper par (block_num, par_num, line_num)
    group_cols = ['block_num', 'par_num', 'line_num']
    for col in group_cols:
        if col not in df.columns:
            df[col] = 0

    lines_grouped = df.groupby(group_cols, sort=False)
    lines_list = []
    for _, grp in lines_grouped:
        words = grp.sort_values('word_num')
        line_text = ' '.join(words['text'].astype(str).tolist()).strip()
        if not line_text:
            continue
        x1 = int(words['left'].min())
        y1 = int(words['top'].min())
        x2 = int((words['left'] + words['width']).max())
        y2 = int((words['top'] + words['height']).max())
        conf = int(words['conf'].mean())
        lines_list.append({
            'text': line_text,
            'bbox': [x1, y1, x2, y2],
            'conf': conf,
        })

    i = 0
    while i < len(lines_list):
        line = lines_list[i]
        text = line['text']
        field_type = _identify_label(text)

        if field_type:
            # Chercher la valeur sur la même ligne (même sans séparateur)
            val_inline = _extract_inline_value(text, field_type)
            
            # Validation pour n_ordre
            if val_inline and field_type == 'n_ordre':
                if not any(c.isdigit() for c in val_inline):
                    val_inline = None
                    
            if val_inline:
                _add_result(results, field_type, val_inline, line['bbox'], line['conf'])
            else:
                # Recherche spatiale de la ligne suivante (directement en dessous)
                # Critères : Y > Y_label, distance Y < 100px, alignement X (overlap ou proximité)
                l_x1, l_y1, l_x2, l_y2 = line['bbox']
                l_cx = (l_x1 + l_x2) / 2
                
                best_next = None
                best_score = float('inf')  # Score combinant distance Y et X
                
                for j, cand in enumerate(lines_list):
                    if i == j: continue
                    c_x1, c_y1, c_x2, c_y2 = cand['bbox']
                    c_cx = (c_x1 + c_x2) / 2
                    
                    y_dist = c_y1 - l_y1
                    x_dist = c_x1 - l_x2
                    
                    # Cas 1 : À droite sur la même ligne (tolérance Y ±15px)
                    if -15 < y_dist < 20 and 0 < x_dist < 150:
                        score = x_dist  # Privilégier la proximité en X
                        if score < best_score:
                            best_score = score
                            best_next = cand
                            
                    # Cas 2 : En dessous, aligné à gauche ou centré (si aucun à droite)
                    elif best_next is None and 0 < y_dist < 150:
                        x_overlap = min(l_x2, c_x2) - max(l_x1, c_x1)
                        is_aligned = x_overlap > -50 or abs(l_x1 - c_x1) < 100 or abs(l_cx - c_cx) < 100
                        if is_aligned:
                            score = y_dist * 2  # Pénaliser un peu la distance Y par rapport à X
                            if score < best_score:
                                best_score = score
                                best_next = cand
                
                if best_next:
                    next_field = _identify_label(best_next['text'])
                    if not next_field:
                        # Nettoyer d'éventuels séparateurs parasites (ex: ": Aubenas" -> "Aubenas")
                        val_txt = best_next['text']
                        if val_txt.startswith(':') or val_txt.startswith('-'):
                            val_txt = val_txt[1:].strip()
                        
                        # ── Filtres de pertinence par type de champ ──────────────────
                        # Pour "commune" : rejeter un texte trop court ou purement numérique
                        if field_type == 'commune':
                            alpha_count = sum(c.isalpha() for c in val_txt)
                            if alpha_count < 3 or len(val_txt.strip()) < 3:
                                i += 1
                                continue
                        
                        # Pour "n_ordre" : privilégier un texte alphanumérique mixte
                        # Si le texte ne contient pas au moins 1 lettre ET 1 chiffre,
                        # on continue à chercher (mais on garde quand même en fallback)
                        if field_type == 'n_ordre':
                            has_digit  = any(c.isdigit() for c in val_txt)
                            # Rejeter les longues phrases parasites (ex: "'ONO avril 1955)")
                            if len(val_txt) > 20 or "avril" in val_txt.lower() or "mars" in val_txt.lower():
                                i += 1
                                continue
                            if not has_digit:
                                # un numéro de DA doit absolument contenir un chiffre, sinon c'est du parasite
                                i += 1
                                continue
                        
                        _add_result(results, field_type, val_txt,
                                    best_next['bbox'], best_next['conf'])
        i += 1

    return results


def _add_result(
    results: Dict,
    field_type: str,
    value: str,
    bbox: List[int],
    conf: int,
) -> None:
    """Ajoute un résultat en évitant les doublons exacts."""
    val = value.strip()
    if not val or val.lower() in ('n/a', 'nan', ''):
        return
    existing = results.get(field_type, [])
    # Éviter les doublons exacts
    if any(e['valeur'] == val for e in existing):
        return
    existing.append({'valeur': val, 'bbox': bbox, 'conf': conf})
    results[field_type] = existing


# ---------------------------------------------------------------------------
# 3b. EXTRACTION SPATIALE DGFIP (documents modernes DGFIP)
# ---------------------------------------------------------------------------

def _is_dgfip_document(spans: List[Dict]) -> bool:
    """
    Détecte si le document est un extrait cadastral DGFIP récent.
    Signature : 'DIRECTION GENERALE DES FINANCES PUBLIQUES' en haut de page.
    """
    for sp in spans:
        if DGFIP_SIGNATURE in sp.get("text", "").upper():
            return True
    return False


def _extract_dgfip_fields(spans: List[Dict], page_w: int) -> Dict[str, Any]:
    """
    Extraction dédiée aux documents DGFIP récents.

    Structure du document (page paysage ou portrait) :
    ┌─────────────────────────────────┬──────────────────────────────────┐
    │ CARTOUCHE GAUCHE (x < 45% larg) │ CARTOUCHE DROIT (x > 45% larg)  │
    │  Commune :                       │  Section      : AH               │
    │  SAINT-PRIVAT (289)              │  Feuille(s)   : 000 AH 01        │
    │                                  │  Qualité...                      │
    │  Numéro d'ordre du document      │  Echelle d'origine  : 1/1000     │
    │  d'arpentage : 891-A             │                                  │
    │                                  │                                  │
    │  Document vérifié le 24/11/2025  │                                  │
    │  Par M.MECHIN Eric               │                                  │
    └─────────────────────────────────┴──────────────────────────────────┘

    Retourne un dict de champs avec source='dgfip_spatial'.
    """
    champs: Dict[str, Any] = {}
    sep_x = page_w * 0.45  # frontière gauche/droite

    # Trier les spans par position Y (haut → bas)
    sorted_spans = sorted(spans, key=lambda s: s["bbox"][1])

    # Séparer les spans gauche et droit
    left_spans  = [s for s in sorted_spans if s["bbox"][0] < sep_x]
    right_spans = [s for s in sorted_spans if s["bbox"][0] >= sep_x]

    # ── CARTOUCHE GAUCHE ────────────────────────────────────────────────────
    # Parcourir les spans gauche à la recherche des labels clés
    i = 0
    while i < len(left_spans):
        sp = left_spans[i]
        txt = sp["text"].strip()
        txt_lower = txt.lower()

        # 1. Label COMMUNE :
        if re.match(r'(?i)^commune\s*:', txt_lower):
            # La commune peut être sur la même ligne après ":" ou sur la ligne suivante
            val_inline = re.sub(r'(?i)^commune\s*:\s*', '', txt).strip()
            if not val_inline and i + 1 < len(left_spans):
                val_inline = left_spans[i + 1]["text"].strip()
                i += 1
            if val_inline:
                champs["commune_brut"] = val_inline
                champs["commune"] = {
                    "valeur": val_inline,  # sera normalisé après
                    "confiance": 0.98,
                    "methode": "dgfip_spatial",
                    "brut": val_inline,
                    "bbox": sp["bbox"],
                }
                # Extraire code INSEE si présent : SAINT-PRIVAT (289)
                m_insee = re.search(r'\((\d{3,5})\)', val_inline)
                if m_insee:
                    champs["code_insee"] = m_insee.group(1)
                    print(f"    [DGFIP] Code INSEE extrait : {m_insee.group(1)}")

        # 2. Label N° d'ordre / d'arpentage :
        elif re.search(r"(?i)d'arpentage\s*:", txt):
            # Format : "d'arpentage : 891-A" ou sur la même ligne
            m_da = re.search(r"(?i)d'arpentage\s*:\s*([\w\-\.]+)", txt)
            if m_da:
                champs["n_ordre"] = {
                    "valeur": m_da.group(1).strip(),
                    "confiance": 0.98,
                    "methode": "dgfip_spatial",
                    "brut": txt,
                    "bbox": sp["bbox"],
                }
                print(f"    [DGFIP] N° d'ordre : {m_da.group(1).strip()}")

        # 3. Label Document vérifié + date :
        elif re.search(r'(?i)document.{0,10}vérifié|document.{0,10}verifie', txt):
            m_date = re.search(r'(\d{1,2}/\d{1,2}/\d{4}|\d{4})', txt)
            if m_date:
                champs["date"] = {
                    "valeur": m_date.group(1),
                    "confiance": 0.95,
                    "methode": "dgfip_spatial",
                    "brut": txt,
                    "bbox": sp["bbox"],
                }
                print(f"    [DGFIP] Date : {m_date.group(1)}")

        # 4. Géomètre-expert : "Dressé par NOM" ou "dressé par le Géomètre-Expert NOM"
        #    UNIQUEMENT les noms de la whitelist → rejet sinon
        elif re.search(r'(?i)dress[eé]\s+par', txt):
            # Extraire ce qui suit "dressé par" (en ignorant les titres parasites)
            m_geo = re.sub(r'(?i)dress[eé]\s+par\s+(?:le\s+)?(?:g[eé]om[eè]tre[\s\-]+expert\s+)?', '', txt).strip()
            m_geo = re.sub(r'(?i)\s*g[eé]om[eè]tre[\s\-]+expert\s*', ' ', m_geo).strip()
            if m_geo and len(m_geo) >= 2:
                # Valider contre la whitelist stricte
                try:
                    from plan_classifier import GEOMETRES_CONNUS
                except ImportError:
                    GEOMETRES_CONNUS = ["DUPUY", "HARROIS", "RACAT", "SERRET", "CEYTE", "BARRIAL", "ROBERT"]
                try:
                    from rapidfuzz import process as rfp, fuzz
                    result = rfp.extractOne(m_geo.upper(), GEOMETRES_CONNUS, scorer=fuzz.WRatio)
                    if result and result[1] >= 70:
                        champs["geometre"] = {
                            "valeur": result[0],
                            "confiance": 0.96,
                            "methode": "dgfip_dresse_par",
                            "brut": txt,
                            "bbox": sp["bbox"],
                        }
                        print(f"    [DGFIP] Géomètre (dressé par) : {result[0]}")
                except ImportError:
                    # Sans rapidfuzz, correspondance exacte
                    for geo in GEOMETRES_CONNUS:
                        if geo.lower() in m_geo.lower():
                            champs["geometre"] = {
                                "valeur": geo,
                                "confiance": 0.90,
                                "methode": "dgfip_dresse_par",
                                "brut": txt,
                                "bbox": sp["bbox"],
                            }
                            print(f"    [DGFIP] Géomètre (dressé par) : {geo}")
                            break
        i += 1


    # ── CARTOUCHE DROIT ─────────────────────────────────────────────────────
    for sp in right_spans:
        txt = sp["text"].strip()

        # Section : "Section      : AH"
        m_sec = re.search(r'(?i)section\s*:\s*([A-Z]{1,2})', txt)
        if m_sec and "section" not in champs:
            champs["section"] = {
                "valeur": m_sec.group(1),
                "confiance": 0.98,
                "methode": "dgfip_spatial",
                "brut": txt,
                "bbox": sp["bbox"],
            }
            print(f"    [DGFIP] Section : {m_sec.group(1)}")

        # Feuille(s) : "Feuille(s)   :  000 AH 01"
        m_feuille = re.search(r'(?i)feuille(?:s)?\s*(?:\(s\))?\s*:\s*(.+)', txt)
        if m_feuille and "feuille" not in champs:
            fval = m_feuille.group(1).strip()
            if fval:
                champs["feuille"] = {
                    "valeur": fval,
                    "confiance": 0.95,
                    "methode": "dgfip_spatial",
                    "brut": txt,
                    "bbox": sp["bbox"],
                }
                print(f"    [DGFIP] Feuille(s) : {fval}")

        # Echelle d'origine : "Echelle d'origine  : 1/1000"
        m_ech = re.search(r"(?i)(?:echelle|échelle)(?:.{0,20})?:\s*(1[/\\]\d+)", txt)
        if m_ech and "echelle" not in champs:
            champs["echelle"] = {
                "valeur": m_ech.group(1).replace('\\', '/'),
                "confiance": 0.98,
                "methode": "dgfip_spatial",
                "brut": txt,
                "bbox": sp["bbox"],
            }
            print(f"    [DGFIP] Echelle : {m_ech.group(1)}")

    return champs


def _is_dgfip_boilerplate(text: str) -> bool:
    """
    Retourne True si le texte est un texte de formulaire DGFiP à ignorer
    (ne pas extraire comme propriétaire, signataire, etc.).
    """
    text_lower = text.lower()
    return any(bp in text_lower for bp in _DGFIP_BOILERPLATE)


# ---------------------------------------------------------------------------
# 4. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def process_modern_plan(
    file_path: str,
    commune_db: List[Dict[str, Any]],
    geometre_id: str = "inconnu",
    writer_styles_dir: str = "writer_styles",
) -> Dict[str, Any]:
    """
    Traite un plan cadastral moderne et extrait tous les champs KPI.

    Retourne un dict :
    {
        "fichier": str,
        "type": "Plan Moderne",
        "geometre_id": str,
        "pages": [
            {
                "page": int,
                "champs": {
                    "commune": {"valeur": str, "confiance": float},
                    "section": {...},
                    "n_ordre": {...},
                    "proprietaires_anciens": {"valeurs": [str], ...},
                    "proprietaires_nouveaux": {"valeurs": [str], ...},
                    "parcelles": {"valeurs": [str], ...},
                    "signataires": {"valeurs": [str], ...},
                    "indication": {"valeur": str, ...},
                    "geometre": {"valeur": str, ...},
                    "echelle": {"valeur": str, ...},
                    "date": {"valeur": str, ...},
                },
                "raw_detections": [...],
            }
        ]
    }
    """
    print(f"\n  [ModernPlan] Traitement : {os.path.basename(file_path)}")

    # Correcteur sémantique (optionnel)
    try:
        from semantic_ocr_engine import SemanticFieldCorrector, CorrectionLearner
        corrector = SemanticFieldCorrector(commune_db)
        learner   = CorrectionLearner(geometre_id, base_dir=writer_styles_dir)
    except ImportError:
        corrector = None
        learner   = None

    # ── Stratégie 1 : PyMuPDF (texte vectoriel natif) ──────────────────────
    pymupdf_spans = _extract_via_pymupdf(file_path)
    has_vector_text = len(pymupdf_spans) > 10  # au moins 10 tokens → PDF vectoriel
    print(f"  [ModernPlan] PyMuPDF → {len(pymupdf_spans)} spans détectés "
          f"({'texte natif' if has_vector_text else 'scan — fallback Tesseract'})")

    # ── Images raster du PDF ────────────────────────────────────────────────
    images = _pdf_to_images(file_path, dpi=150)
    if not images:
        print(f"  [ModernPlan] Impossible de lire le PDF.")
        return {"fichier": file_path, "type": "Plan Moderne", "pages": []}

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.dirname(file_path).replace("inputs", "outputs") \
                 if "inputs" in file_path else "outputs"

    all_pages = []

    for page_idx, img_bgr in enumerate(images):
        page_num = page_idx + 1
        print(f"  [ModernPlan] Page {page_num}...")

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_gray)

        # ── Sélection de la source d'extraction ─────────────────────────────
        # ── Sélection de la source d'extraction ─────────────────────────────
        page_spans = [s for s in pymupdf_spans if s["page"] == page_num]

        # Vérifier si les spans vectoriels contiennent les infos clés
        text_vec = " ".join([s["text"] for s in page_spans]).lower()
        is_valid_vector = has_vector_text and len(page_spans) > 3 and any(k in text_vec for k in ["commune", "d'ordre", "section", "dossier", "parcelle"])

        raw_detections = []
        # ── Détection DGFIP : extraction spécifique si document DGFIP détecté ──
        # Fonctionne AUSSI sur les scans via les résultats OCR (EasyOCR à venir)
        is_dgfip = _is_dgfip_document(page_spans if is_valid_vector else [])
        if is_dgfip:
            print(f"    [ModernPlan] ✅ Document DGFIP détecté (vectoriel) — extraction cartouche spatiale")

        if is_valid_vector:
            # Convertir les spans en DataFrame compatible avec parse_label_value_pairs
            tess_df = _spans_to_tess_df(page_spans)
            raw_detections = [
                {"texte": s["text"], "confiance": f"{s['conf']}%",
                 "bbox": s["bbox"], "type_ocr": "PyMuPDF (vectoriel)"}
                for s in page_spans
            ]
            print(f"    [ModernPlan] Extraction PyMuPDF : {len(page_spans)} lignes")
        else:
            if has_vector_text and len(page_spans) > 3:
                print("    [ModernPlan] PDF partiellement vectoriel (mots-clés absents), passage à l'OCR...")
            else:
                print("    [ModernPlan] Pas de texte vectoriel suffisant, passage à l'OCR...")
            
            # Fallback EasyOCR
            best_df = _run_ocr_fallback(img_gray)
            if best_df is not None and not best_df.empty:
                tess_df = best_df
                print(f"    [ModernPlan] Fallback EasyOCR OK : {len(tess_df)} éléments")
            else:
                tess_df = pd.DataFrame()
                print("    [ModernPlan] Échec extraction vectorielle ET OCR")

            if not tess_df.empty:
                try:
                    df_v = tess_df.copy()
                    df_v = df_v[df_v['conf'].apply(lambda x: str(x).lstrip('-').isdigit())]
                    df_v['conf'] = df_v['conf'].astype(int)
                    df_v = df_v[df_v['conf'] >= 30].dropna(subset=['text'])
                    for _, row in df_v.iterrows():
                        txt = str(row.get('text', '')).strip()
                        if txt:
                            raw_detections.append({
                                "texte": txt, "confiance": f"{row['conf']}%",
                                "bbox": [int(row.get('left', 0)), int(row.get('top', 0)),
                                         int(row.get('left', 0) + row.get('width', 0)),
                                         int(row.get('top', 0) + row.get('height', 0))],
                                "type_ocr": "Tesseract (Plan Moderne)",
                            })
                except Exception:
                    pass
                # ── Détection DGFIP via OCR pour les scans ────────────────────────
                if not is_dgfip:
                    ocr_texts = [str(row.get('text', '')) for _, row in df_v.iterrows()]
                    combined = " ".join(ocr_texts).upper()
                    if DGFIP_SIGNATURE.replace(' ', '') in combined.replace(' ', '') or \
                       "FINANCES PUBLIQUES" in combined or "DIRECTION GENERALE" in combined:
                        is_dgfip = True
                        print(f"    [ModernPlan] ✅ Document DGFIP détecté via OCR (scan) — extraction DGFIP activée")

        text_lines = []
        if tess_df is not None and not (hasattr(tess_df, 'empty') and tess_df.empty):
            df_t = tess_df.sort_values(by=['top', 'left'])
            text_lines = df_t['text'].dropna().astype(str).tolist()
            
        full_text = "\n".join(text_lines)
        champs: Dict[str, Any] = {}

        # ── PRIORITÉ 0 : Extraction DGFIP (s'applique vectoriel ET scan) ───────────────
        if is_dgfip:
            # Choisir la source de spans selon disponibilité
            if is_valid_vector:
                spans_dgfip = page_spans
            else:
                # Convertir les raw_detections OCR en spans factices pour réutiliser _extract_dgfip_fields
                spans_dgfip = [
                    {"text": d["texte"], "bbox": d["bbox"], "conf": 90, "page": page_num}
                    for d in raw_detections if d.get("texte")
                ]
            dgfip_champs = _extract_dgfip_fields(spans_dgfip, img_bgr.shape[1])
            for field, val in dgfip_champs.items():
                if field.startswith('_') or field == 'commune_brut':
                    continue
                champs[field] = val

            # Normaliser la commune DGFiP : retirer le code INSEE, puis fuzzy-matcher
            if "commune" in champs:
                raw_commune = champs["commune"]["valeur"]
                # Retirer le code INSEE : "SAINT-PRIVAT (289)" → "SAINT-PRIVAT"
                commune_clean = re.sub(r'\s*\(\d{3,5}\)\s*$', '', raw_commune).strip()
                # Normalisation via corrector si disponible
                if corrector:
                    c_res = corrector.correct(commune_clean, "commune")
                    commune_normalisee = c_res.get("valeur", commune_clean)
                    if commune_normalisee and commune_normalisee != "Non identifiée":
                        champs["commune"]["valeur"] = commune_normalisee
                        champs["commune"]["confiance"] = c_res.get("confiance", 0.95)
                        print(f"    [DGFIP] Commune normalisée : {raw_commune!r} → {commune_normalisee!r}")
                    else:
                        champs["commune"]["valeur"] = commune_clean
                else:
                    champs["commune"]["valeur"] = commune_clean

        regex_results = parse_label_value_pairs(tess_df, img_gray.shape[0] if img_gray is not None else 2000)
        for field, vals in regex_results.items():
            # Si champ déjà trouvé par DGFIP spatial → ne pas écraser (sauf si confiance plus haute)
            if field in champs and champs[field].get("confiance", 0) >= 0.95:
                continue

            valid_vals = [str(v.get('valeur', '')).strip() for v in vals if str(v.get('valeur', '')).strip()]
            if not valid_vals: continue

            if field in ['n_ordre', 'date']:
                with_digits = [v for v in valid_vals if re.search(r'\d', v)]
                if with_digits:
                    valid_vals = with_digits

            # ── FILTRE ECHELLE : accepter uniquement le format 1/XXXX ────────
            if field == 'echelle':
                echelle_valides = [v for v in valid_vals if re.search(r'^1\s*/\s*\d{3,6}$', v.strip())]
                if not echelle_valides:
                    # Chercher dans le texte brut si le parseur a pris quelque chose d'incorrect
                    echelle_valides = []
                    for v_raw in valid_vals:
                        m_ech = re.search(r'1\s*/\s*(\d{3,6})', v_raw)
                        if m_ech:
                            echelle_valides.append(f"1/{m_ech.group(1)}")
                if not echelle_valides:
                    continue  # Ignorer si pas de format 1/XXXX valide
                valid_vals = echelle_valides

            # ── FILTRE SECTION : doit être 1-2 lettres uniquement (ex: A, AB, AH) ─────
            if field == 'section':
                valid_sections = [v for v in valid_vals
                                  if re.match(r'^[A-Z]{1,2}$', v.strip().upper())]
                if not valid_sections:
                    continue  # Rejeter '0', 'Feuille(s)', etc.
                valid_vals = valid_sections

            # ── FILTRE BOILERPLATE DGFIP ─────────────────────────────────────
            if is_dgfip and field in ('proprietaires_nouveaux', 'proprietaires_anciens', 'signataires'):
                valid_vals = [v for v in valid_vals if not _is_dgfip_boilerplate(v)]
                if not valid_vals:
                    continue

            if field in ['parcelles', 'proprietaires_anciens', 'proprietaires_nouveaux', 'signataires']:
                val_str = " / ".join(valid_vals)
            else:
                val_str = valid_vals[0]

            champs[field] = {
                "valeur": val_str,
                "confiance": 1.0,
                "methode": "regex_spatiale",
                "brut": val_str,
                "bbox": vals[0].get('bbox', []) if vals else []
            }
            print(f"    [ModernPlan] {field} récupéré via structure/regex : {val_str}")


        # ── Palier 1 : Recherche globale du Géomètre dans tout le texte (si non trouvé) ──
        if "geometre" not in champs and is_dgfip:
            try:
                from plan_classifier import GEOMETRES_CONNUS
            except ImportError:
                GEOMETRES_CONNUS = ["DUPUY", "HARROIS", "RACAT", "SERRET", "CEYTE", "BARRIAL", "ROBERT"]
            
            # Recherche exacte
            found_geo = None
            for geo in GEOMETRES_CONNUS:
                if geo.lower() in full_text.lower():
                    found_geo = geo
                    break
            
            # Recherche floue si pas de correspondance exacte
            if not found_geo:
                try:
                    from rapidfuzz import process as rfp, fuzz
                    # On découpe le texte en blocs de 2-3 mots pour chercher le nom
                    words = full_text.replace('\n', ' ').split()
                    chunks = [" ".join(words[i:i+2]) for i in range(len(words)-1)] + [" ".join(words[i:i+3]) for i in range(len(words)-2)]
                    
                    best_match = None
                    best_score = 0
                    for chunk in chunks:
                        res = rfp.extractOne(chunk.upper(), GEOMETRES_CONNUS, scorer=fuzz.token_set_ratio)
                        if res and res[1] > best_score:
                            best_score = res[1]
                            best_match = res[0]
                    
                    if best_match and best_score >= 85:
                        found_geo = best_match
                except ImportError:
                    pass

            if found_geo:
                champs["geometre"] = {
                    "valeur": found_geo,
                    "confiance": 0.85,
                    "methode": "dgfip_global_search",
                    "brut": found_geo,
                    "bbox": []
                }
                print(f"    [DGFIP] Géomètre trouvé par recherche globale : {found_geo}")

        # ── Palier 2 : Recherche explicite du label 'commune' par regex (si non trouvé) ──
        if "commune" not in champs:
            commune_from_ocr = None
            from rapidfuzz import process as rfp, fuzz
            m_comm = re.search(r'(?i)commune\s*(?:de\s*)?(?::|\-)?\s*([A-Za-zÀ-ÿ0-9\-\'\s\(\)]{3,45}?)(?:\n|section|feuille|date|echelle|dossier|num[eé]ro|n[o°]|$)', full_text)
            if m_comm:
                val_brute = m_comm.group(1).strip()
                val_brute = re.split(r'(?i)lieu[ \-]dit|proprietaire|echelle', val_brute)[0].strip()
                if len(val_brute) >= 3:
                    noms_officiels = [e['officiel'] for e in commune_db]
                    best = rfp.extractOne(val_brute, noms_officiels, scorer=fuzz.token_set_ratio)
                    if best and best[1] >= 80:
                        commune_from_ocr = best[0]
                    else:
                        if len(val_brute) >= 5:
                            commune_from_ocr = val_brute
            if commune_from_ocr:
                print(f"  [ModernPlan] Palier 1 OCR : Commune trouvée dans le texte : '{commune_from_ocr}'")
                champs["commune"] = {
                    "valeur": commune_from_ocr,
                    "confiance": 0.9,
                    "methode": "ocr_brut_exact",
                    "brut": commune_from_ocr,
                    "bbox": [0,0,0,0]
                }

        # ── Extraction des parcelles par Colorimétrie (Rouge/Vert) ────────────────
        try:
            from color_ocr_engine import extract_color_parcels
            print("  [ModernPlan] Extraction des parcelles par couleur (rouge/vert)...")
            color_res = extract_color_parcels(img_bgr)
            
            if color_res["nouvelles_parcelles"]:
                champs["nouvelles_parcelles"] = {
                    "valeur": ", ".join([p["valeur"] for p in color_res["nouvelles_parcelles"]]),
                    "confiance": 0.95,
                    "methode": "couleur_rouge",
                    "brut": str(color_res["nouvelles_parcelles"]),
                    "bbox": color_res["nouvelles_parcelles"][0]["bbox"] if color_res["nouvelles_parcelles"] else [0,0,0,0]
                }
            if color_res["anciennes_parcelles"]:
                champs["anciennes_parcelles"] = {
                    "valeur": ", ".join([p["valeur"] for p in color_res["anciennes_parcelles"]]),
                    "confiance": 0.95,
                    "methode": "couleur_vert",
                    "brut": str(color_res["anciennes_parcelles"]),
                    "bbox": color_res["anciennes_parcelles"][0]["bbox"] if color_res["anciennes_parcelles"] else [0,0,0,0]
                }
        except Exception as e:
            print(f"  [ModernPlan] Erreur lors de l'extraction couleur : {e}")

        # ── Extraction par LLM Textuel (Fallback uniquement) ────────────────────────
        expected_fields = ["commune", "section", "n_ordre", "n_dossier", "geometre", "date", "proprietaires_anciens", "proprietaires_nouveaux", "parcelles", "echelle"]
        missing_fields = [f for f in expected_fields if f not in champs]
        
        if missing_fields:
            print(f"  [ModernPlan] Envoi au LLM pour champs manquants: {missing_fields}")
            prompt = f"""Tu es un expert en extraction de données cadastrales. 
Voici le texte brut OCR d'un plan cadastral moderne.
Extrais UNIQUEMENT les informations suivantes sous forme d'objet JSON strict. Si non trouvé, met une chaine vide "".
Champs attendus : {', '.join([f'"{f}"' for f in missing_fields])}.
Texte OCR :
---
{full_text}
---
Réponds UNIQUEMENT par le JSON."""
            
            payload = {"model": "llava", "prompt": prompt, "format": "json", "stream": False, "options": {"temperature": 0.0, "num_predict": 512, "seed": 42}}
            os.makedirs("outputs", exist_ok=True)
            payload_path = os.path.join(os.getcwd(), "outputs", "llm_text_payload.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            win_payload_path = payload_path.replace("/mnt/c/", "C:\\\\").replace("/", "\\\\")
            cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate", "-H", "Content-Type: application/json", "-d", f"@{win_payload_path}"]
            
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if res.returncode == 0 and res.stdout.strip():
                    try:
                        ollama_json = json.loads(res.stdout)
                        raw = ollama_json.get("response", "").strip()
                        
                        # Nettoyage des éventuelles balises markdown générées par l'IA
                        raw = re.sub(r'^```json\s*', '', raw, flags=re.IGNORECASE)
                        raw = re.sub(r'^```\s*', '', raw)
                        raw = re.sub(r'\s*```$', '', raw)
                        
                        m_obj = re.search(r'\{.*\}', raw, re.DOTALL)
                        if m_obj:
                            llm_data = json.loads(m_obj.group(0))
                            for field_type, v in llm_data.items():
                                if field_type not in missing_fields: continue
                                if not v or v == "None" or v == "null": continue
                                
                                val_finale = " / ".join(map(str, v)) if isinstance(v, list) else str(v)
                                champs[field_type] = {
                                    "valeur": val_finale,
                                    "confiance": 0.50,
                                    "methode": "llm_textuel",
                                    "brut": str(v),
                                    "bbox": [0,0,0,0]
                                }
                                print(f"    [ModernPlan] {field_type} récupéré via LLM : {val_finale}")
                    except json.JSONDecodeError:
                        print("    [ModernPlan] Échec de parsing du JSON LLM.")
            except Exception as e:
                print(f"  [ModernPlan] Erreur LLM Textuel: {e}")
            finally:
                if os.path.exists(payload_path): os.remove(payload_path)

        # ── Nettoyage final ───────────────────────────────
        for field_type, data in champs.items():
            if corrector:
                if "valeurs" in data or " / " in data["valeur"]:
                    vals = data["valeur"].split(" / ")
                    val_corr = [corrector.correct(item.strip(), field_type).get('valeur', item.strip()) for item in vals]
                    data["valeur"] = " / ".join(filter(None, val_corr))
                else:
                    c_res = corrector.correct(data["valeur"], field_type)
                    data["valeur"] = c_res.get("valeur", data["valeur"])

        if not champs:
            print(f"  [ModernPlan] Page {page_num} : aucun champ extrait.")

        # Passer les détections brutes pour l'affichage
        champs['_raw_detections'] = raw_detections

        # ── Génération de l'image annotée ───────────────────────────────────
        annot_path = os.path.join(output_dir, f"{base_name}_page_{page_num}_annote.jpg")
        try:
            _generate_annotated_image(img_bgr, champs, annot_path)
        except Exception as e_ann:
            print(f"  [ModernPlan] Avertissement image annotée: {e_ann}")
            
        # Retirer _raw_detections pour ne pas polluer les JSON
        champs.pop('_raw_detections', None)

        # ── Palier 3 : Fallback final depuis le nom de fichier ────────────────────
        fichier_base = os.path.basename(file_path)
        m_da = re.search(r'(?:^\d{3}[_\-])(\d{3,5}[_\-][A-Za-z])', fichier_base)
        if m_da and "n_ordre" not in champs:
            da_val = m_da.group(1).replace('_', '-')
            champs["n_ordre"] = {
                "valeur": da_val,
                "confiance": 0.95,
                "methode": "filename_fallback",
                "brut": f"[filename:{fichier_base}]",
                "bbox": [0,0,0,0]
            }
            print(f"  [ModernPlan] Fallback nom de fichier: DA récupéré '{da_val}'")
            
        m_geof = re.search(r'geofoncier_dmpc_\d{5}_[0-9a-z]{1,3}_(\d{1,5})', fichier_base.lower())
        if m_geof and "n_ordre" not in champs:
            champs["n_ordre"] = {
                "valeur": m_geof.group(1),
                "confiance": 0.95,
                "methode": "filename_fallback",
                "brut": f"[filename:{fichier_base}]",
                "bbox": [0,0,0,0]
            }
            print(f"  [ModernPlan] Fallback nom de fichier: DA récupéré '{m_geof.group(1)}'")

        all_pages.append({
            'page': page_num,
            'champs': champs,
            'raw_detections': raw_detections,
        })

    return {
        'fichier': file_path,
        'type': 'Plan Moderne (DA/DMPC)',
        'geometre_id': geometre_id,
        'pages': all_pages,
    }


# ---------------------------------------------------------------------------
# 5. EXPORT CSV POUR app_validation.py
# ---------------------------------------------------------------------------

def export_modern_plan_to_csv(
    result: Dict[str, Any],
    output_dir: str = "outputs",
) -> str:
    """
    Exporte les résultats du plan moderne vers un CSV compatible
    avec app_validation.py.

    Colonnes identiques aux livrets + colonnes supplémentaires spécifiques
    aux plans modernes.
    """
    base_name = os.path.splitext(os.path.basename(result['fichier']))[0]
    csv_path = os.path.join(output_dir, f"{base_name}_resultats.csv")
    csv_raw_path = os.path.join(output_dir, f"{base_name}_texte_brut.csv")

    rows = []
    raw_rows = []
    for page_data in result.get('pages', []):
        page_num = page_data['page']
        champs = page_data.get('champs', {})
        
        # Exporter l'intégralité du texte brut détecté (comme l'ancienne méthode)
        for idx, raw_det in enumerate(page_data.get('raw_detections', [])):
            raw_rows.append({
                'ID_Ligne': f"p{page_num}_l{idx}",
                'Page': page_num,
                'Texte_Extrait': raw_det.get('texte', ''),
                'Confiance': raw_det.get('confiance', ''),
                'Outil_Utilise': raw_det.get('type_ocr', ''),
                'Bbox': str(raw_det.get('bbox', []))
            })

        def _get(field, key='valeur', default='Inconnu'):
            f = champs.get(field, {})
            if isinstance(f, dict):
                return f.get(key, default)
            return default

        def _get_list(field, default=''):
            f = champs.get(field, {})
            if isinstance(f, dict):
                vals = f.get('valeurs', [f.get('valeur', '')])
                return ' / '.join(str(v) for v in vals if v)
            return default

        row = {
            'ID_Ligne': f"plan_p{page_num}",
            'Type_Document': result.get('type', 'Plan Moderne'),
            'Commune_Doc': _get('commune'),
            'Commune_Ligne': _get('commune'),
            'Confiance_Commune_%': int(_get('commune', 'confiance', 0) * 100),
            'Confirmation_Status': '1ère passe',
            'Methode_Commune': _get('commune', 'methode', ''),
            'OCR_brut_commune': _get('commune', 'brut', ''),
            'Hypotheses_OCR': '',
            'Geometre': _get('geometre'),
            'N_Dossier': _get('n_ordre'),
            'Ordre': _get('n_ordre'),
            'Echelle': _get('echelle'),
            'Page': page_num,
            'Texte_Extrait': '',
            'Confiance': f"{int(_get('commune', 'confiance', 0) * 100)}%",
            'Outil_Utilise': 'Tesseract (Plan Moderne)',
            'Coordonnees_Bbox_xyxy': str(_get('commune', 'bbox', [])),

            # Dictionnaire de toutes les bboxes pour l'UI (zoom par champ)
            'Bbox_Champs': json.dumps({
                f: champs[f].get('bbox', [])
                for f in champs if isinstance(champs[f], dict) and champs[f].get('bbox')
            }),

            # Champs spécifiques plans modernes
            'Section_Cadastrale': _get('section'),
            'Parcelles': _get_list('parcelles'),
            'Nouvelles_Parcelles': _get_list('nouvelles_parcelles'),
            'Anciennes_Parcelles': _get_list('anciennes_parcelles'),
            'Proprietaires_Anciens': _get_list('proprietaires_anciens'),
            'Proprietaires_Nouveaux': _get_list('proprietaires_nouveaux'),
            'Signataires': _get_list('signataires'),
            'Indication_Document': _get('indication'),
            'Date_Document': _get('date'),
            'Confiance_Section_%': int(_get('section', 'confiance', 0) * 100),
            'Confiance_Geometre_%': int(_get('geometre', 'confiance', 0) * 100),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(csv_path, sep=';', index=False, encoding='utf-8-sig')

        # Export Excel également
        excel_path = csv_path.replace('.csv', '.xlsx')
        try:
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer_xl:
                df.to_excel(writer_xl, index=False, sheet_name='Plan_Moderne')
        except Exception:
            pass

        print(f"  [ModernPlan] CSV exporté : {csv_path}")
        print(f"  [ModernPlan] Excel exporté : {excel_path}")
    else:
        print(f"  [ModernPlan] Aucune donnée à exporter.")

    if raw_rows:
        df_raw = pd.DataFrame(raw_rows)
        df_raw.to_csv(csv_raw_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"  [ModernPlan] CSV Brut (tout le texte) exporté : {csv_raw_path}")

    return csv_path


# ---------------------------------------------------------------------------
# 6. DÉTECTION DU TYPE DE DOCUMENT
# ---------------------------------------------------------------------------

def is_modern_plan(file_path: str) -> bool:
    """
    Détecte si un fichier est un plan cadastral moderne (vs livret manuscrit).

    Heuristiques :
    - Nom de fichier contient des patterns DMPC/DA (ex: 1992C100001...)
    - PDF peu volumineux (< 5 Mo) → plan A4/A3 imprimé
    - Première page : ratio texte imprimé / texte manuscrit élevé
    """
    filename = os.path.basename(file_path).lower()

    # Patterns de noms typiques des plans cadastraux numérisés
    plan_patterns = [
        r'\d{4}[a-z]\d+',          # 1992C100001...
        r'_2pl[a-z]?$',             # ..._2PLa
        r'_da\d*$',                  # ..._DA1
        r'_dmpc\d*$',               # ..._DMPC
        r'bornage', r'arpentage',
        r'division', r'lotissement',
        r'document.*travail',
        r'pv.*bornage', r'procès.*verbal', r'pva'
    ]
    for pat in plan_patterns:
        if re.search(pat, filename):
            return True

    # Heuristique taille : les plans modernes vectoriels sont < 0.5 Mo/page
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    try:
        doc = fitz.open(file_path)
        n_pages = doc.page_count
        
        # La taille seule ne suffit pas, car les vieux plans très compressés font < 0.5 Mo/page.
        # On va vérifier le contenu vectoriel ou la présence de mots-clés typiques.
            
        # Heuristique 2 : Contenu texte rapide (vectoriel)
        page1_text = doc[0].get_text("text").upper()
        doc.close()
        
        # Mots-clés discriminants pour les PV / Plans de bornage
        keywords = ['BORNAGE', 'PROCÈS-VERBAL', 'PROCES-VERBAL', 'GEOMETRE-EXPERT', 'GÉOMÈTRE-EXPERT', 'DMPC', 'PROCÉS-VERBAL']
        if any(kw in page1_text for kw in keywords):
            return True
            
        # Si texte vectoriel absent (scan lourd ou image raster), on tente un petit OCR rapide
        if len(page1_text.strip()) < 50:
            import easyocr
            import cv2
            import numpy as np
            
            # Recharger avec une faible résolution juste pour trouver les mots-clés
            doc = fitz.open(file_path)
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            doc.close()
            
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            reader = easyocr.Reader(['fr'], gpu=False, verbose=False)
            results = reader.readtext(img_gray, detail=0)
            ocr_text = " ".join(results).upper()
            
            if any(kw in ocr_text for kw in keywords):
                return True

    except Exception as e:
        print(f"  [is_modern_plan] Erreur: {e}")

    return False
