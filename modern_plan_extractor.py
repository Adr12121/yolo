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
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import fitz  # PyMuPDF  # type: ignore
from PIL import Image  # type: ignore


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
        "n°", "ordre"
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
            if kw_norm and kw_norm in norm:
                return field_type
    return None


def _extract_inline_value(text: str, field_type: str) -> Optional[str]:
    """Extrait la valeur sur la même ligne, même sans deux-points, en supprimant le label."""
    import re
    # 1. Cherche deux points ou une longue suite de points/tirets
    m = re.split(r'[:–]|\.{2,}', text, maxsplit=1)
    if len(m) > 1:
        val = m[1].strip()
        if len(val) > 1:
            return val
            
    # 2. Si pas de séparateur, on retire le label détecté du texte
    norm_text = text.lower()
    for kw in sorted(PLAN_LABELS.get(field_type, []), key=len, reverse=True):
        if kw.lower() in norm_text:
            idx = norm_text.find(kw.lower())
            if idx != -1:
                val = text[idx + len(kw):].strip()
                # Enlever les éventuels tirets ou points restants au début
                val = re.sub(r'^[\s\-\.:]+', '', val)
                if len(val) > 1:
                    return val
    return None


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
                    if -15 < y_dist < 20 and 0 < x_dist < 400:
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
        page_spans = [s for s in pymupdf_spans if s["page"] == page_num]

        if has_vector_text and len(page_spans) > 3:
            # Convertir les spans en DataFrame compatible avec parse_label_value_pairs
            tess_df = _spans_to_tess_df(page_spans)
            raw_detections = [
                {"texte": s["text"], "confiance": f"{s['conf']}%",
                 "bbox": s["bbox"], "type_ocr": "PyMuPDF (vectoriel)"}
                for s in page_spans
            ]
            print(f"    [ModernPlan] Extraction PyMuPDF : {len(page_spans)} lignes")
        else:
            # Fallback EasyOCR
            best_df = _run_ocr_fallback(img_gray)
            if best_df is not None and not best_df.empty:
                tess_df = best_df
                print(f"    [ModernPlan] Fallback EasyOCR OK : {len(tess_df)} éléments")
            else:
                tess_df = pd.DataFrame()
                print("    [ModernPlan] Échec extraction vectorielle ET OCR (image illisible ou OCR manquant)")

            raw_detections = []
            if tess_df is not None and not (hasattr(tess_df, 'empty') and tess_df.empty):
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

        # ── Extraction des paires label/valeur ──────────────────────────────
        raw_pairs = parse_label_value_pairs(tess_df, img_gray.shape[0])

        # ── Validation et construction des champs ────────────────────────────
        champs: Dict[str, Any] = {}
        for field_type, candidates in raw_pairs.items():
            if not candidates:
                continue
            best = max(candidates, key=lambda c: c['conf'])
            val_brut = best['valeur']

            val_corrigee = val_brut
            if learner:
                appris = learner.lookup(field_type, val_brut)
                if appris:
                    val_corrigee = appris
                    print(f"    [ModernPlan] Appris [{field_type}]: '{val_brut}' → '{appris}'")

            if corrector:
                correction = corrector.correct(val_corrigee, field_type)
                valeur_finale = correction.get('valeur', val_corrigee)
                confiance     = correction.get('confiance', best['conf'] / 100.0)
                methode       = correction.get('methode', 'pymupdf')
            else:
                valeur_finale = val_corrigee
                confiance     = best['conf'] / 100.0
                methode       = 'pymupdf' if has_vector_text else 'tesseract'

            if field_type in ('proprietaires_anciens', 'proprietaires_nouveaux',
                              'parcelles', 'signataires'):
                champs[field_type] = {
                    'valeurs': [c['valeur'] for c in candidates],
                    'valeur': valeur_finale, 'confiance': confiance,
                    'methode': methode, 'brut': val_brut, 'bbox': best['bbox'],
                }
            else:
                champs[field_type] = {
                    'valeur': valeur_finale, 'confiance': confiance,
                    'methode': methode, 'brut': val_brut, 'bbox': best['bbox'],
                }
            print(f"    [ModernPlan] {field_type}: '{valeur_finale}' "
                  f"(conf={confiance:.0%}, méth={methode})")

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
        
        # Heuristique 1 : Taille par page
        size_per_page = size_mb / max(n_pages, 1)
        if size_per_page < 0.5:
            doc.close()
            return True
            
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
