"""
spatial_extractor.py — Extraction par graphe de proximité spatial label→valeur.

Principe :
  1. Identifier les blocs OCR contenant un label connu (ex: "Dossier n°", "Fait le")
  2. Pour chaque label trouvé, chercher sa valeur dans le voisinage DIRECTIONNEL :
       - Priorité 1 : bloc à DROITE sur la même ligne
       - Priorité 2 : bloc EN DESSOUS dans la même colonne
  3. Si label et valeur sont dans le MÊME bloc OCR ("Commune : Vals-les-Bains"),
     extraire la partie après le séparateur.

Avantages vs regex sur zones fixes :
  - Agnostique au type de document (PVa, PLa, DMPC, autre)
  - Agnostique à la mise en page (cartouche en haut, à droite, au milieu)
  - Élimine naturellement les ambiguïtés ("Fait le" vs "née le")
  - Distingue propriétaire ancien et nouveau via leurs labels respectifs

Dépendances : numpy, unicodedata (standard library) — pas de modèle ML requis.
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Optional, Any


# ── Normalisation texte ──────────────────────────────────────────────────────
def _norm(text: str) -> str:
    # Normaliser d'abord les apostrophes/guillemets (Correction 1 universel)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u02bc", "'")
    text = text.replace("N'", "N°").replace("N\u00b0", "n ").replace("n'", "n ")
    nfkd = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).strip()


# ── Dictionnaire des labels connus par champ ─────────────────────────────────
# Format : {champ: [label_normalisé, ...]}
# Les labels sont triés par longueur décroissante (plus long = plus spécifique)
FIELD_LABELS: Dict[str, List[str]] = {
    "commune": [
        "sur la commune de",
        "en la commune de",
        "territoire de la commune de",
        "territoire de",
        "commune de",
        "commune :",
        "ville de",
        "commune",        # DMPC anciens : label seul, valeur en dessous
    ],
    "n_ordre": [
        "n d ordre du document d arpentage",  # Format exact DGFiP sur vieux plans
        "n d ordre du document",
        "numero d ordre du document d arpentage",
        "numero d ordre",
        "numero d arpentage",
        "n d arpentage",
        "n d ordre",
        "dossier n",
        "affaire n",
        "reference :",
        "da n",
        "n ordre",
        "ref :",
    ],
    "n_dossier": [
        "n de dossier",
        "dossier :",
        "reference :",
        "ref :",
    ],
    "section": [
        "section cadastrale n",
        "section cadastrale",
        "section n",
        "section :",
    ],
    "feuille": [
        "feuille cadastrale",
        "feuille n",
        "feuille :",
    ],
    "date": [
        "en date du",
        "etabli le",
        "dresse le",
        "signe le",
        "edite le",
        "fait a",
        "fait le",
        "date :",
        "date",
        "le :",
        "le",
    ],
    "echelle": [
        "echelle :",
        "ech :",
        "echelle",
    ],
    "geometre": [
        "geometre expert :",
        "geometre-expert :",
        "geometre expert",
        "dresse par m.",
        "dresse par m",
        "dresse par mr",
        "arpentage dresse par m",
        "arpentage dresse par",
        "etabli par :",
        "le geometre",
        "geometre :",
        "par le cabinet",
        "cabinet :",
        "cabinet",
        "dresse par",        # sans ":" pour DMPC anciens
        "le soussigne geometre",
        "geometre",
    ],
    "proprietaires_anciens": [
        "ancien proprietaire",
        "proprietaire actuel",
        "proprietaire sortant",
        "a la demande de",
        "a la requete de",
        "appartenant a",
        "propriete de",
        "cedant :",
        "cedant",
        "vendeur :",
        "vendeur",
        "requis par",
        "denomme",
        "ci dessus nomme",
    ],
    "proprietaires_nouveaux": [
        "nouveau proprietaire",
        "proprietaire entrant",
        "beneficiaire :",
        "beneficiaire",
        "acquereur :",
        "acquereur",
        "acheteur :",
        "acheteur",
        "au profit de",
        "en faveur de",
        "acquis par",
        "est attribue a",
        "cede a",
        "au benefice de",
        "transfere a",
        "desormais proprietaire",
        "mutation au profit de",
    ],
    "indication": [
        "nature des operations",
        "objet du document",
        "objet :",
    ],
}

# Pré-tri par longueur décroissante pour favoriser les labels les plus spécifiques
for _field in FIELD_LABELS:
    FIELD_LABELS[_field] = sorted(FIELD_LABELS[_field], key=len, reverse=True)


# ── Classe Block : représentation spatiale d'un bloc OCR ───────────────────
class Block:
    """Représente un bloc OCR avec ses coordonnées et son texte."""

    def __init__(self, bbox, text: str, prob: float):
        pts = bbox if hasattr(bbox[0], "__iter__") else bbox
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.x0 = min(xs)
        self.y0 = min(ys)
        self.x1 = max(xs)
        self.y1 = max(ys)
        self.cx = (self.x0 + self.x1) / 2
        self.cy = (self.y0 + self.y1) / 2
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        self.text = text.strip()
        self.norm = _norm(text)
        self.prob = prob

    def zone_frac(self, img_w: float, img_h: float) -> List[float]:
        return [self.x0 / img_w, self.y0 / img_h, self.x1 / img_w, self.y1 / img_h]

    def __repr__(self):
        return f"Block({self.text[:30]!r}, x={self.x0:.0f}-{self.x1:.0f}, y={self.y0:.0f}-{self.y1:.0f})"


# ── Détection de label ───────────────────────────────────────────────────────
def _find_label_match(block: Block) -> Optional[Tuple[str, str]]:
    """
    Retourne (field, label_matché) si ce bloc contient un label connu.
    Prioritise les labels les plus longs (plus spécifiques).
    """
    norm = block.norm
    import difflib
    
    # On cherche les labels dans le texte normalisé du bloc
    candidates = []
    
    # Séparer le texte du bloc en mots pour pouvoir faire du fuzzy match sur les fenêtres
    words = norm.split()
    
    for field, labels in FIELD_LABELS.items():
        for lbl in labels:
            # 1. Match exact (très rapide)
            if lbl in norm:
                candidates.append((field, lbl, len(lbl)))
                continue  # On passe au label suivant
                
            # 2. Fuzzy match pour contourner les erreurs OCR (ex: 'commundie', 'dr8sse')
            if len(lbl) >= 6:
                # Si le bloc entier est très proche du label
                if len(norm) <= len(lbl) + 5:
                    ratio = difflib.SequenceMatcher(None, lbl, norm).ratio()
                    if ratio >= 0.82:
                        candidates.append((field, lbl, len(lbl)))
                        continue
                
                # Ou si un mot du bloc est très proche du label (ex: mot unique)
                if len(lbl.split()) == 1:
                    for w in words:
                        if len(w) >= 5 and abs(len(w) - len(lbl)) <= 2:
                            if difflib.SequenceMatcher(None, lbl, w).ratio() >= 0.82:
                                candidates.append((field, lbl, len(lbl)))
                                break

    if not candidates:
        return None

    # Prendre le candidat au label le plus long (le plus spécifique)
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]


def _extract_inline_value(block: Block, label_str: str) -> Optional[str]:
    """
    Si label et valeur sont dans le MEME bloc OCR
    (ex: "Commune : VALS LES BAINS"), extrait la partie valeur.
    """
    norm = block.norm
    idx = norm.find(label_str)
    if idx == -1:
        return None

    # Le label doit se trouver dans la premiere moitie du bloc
    # Si idx > 50% du texte norm, le label est probablement la valeur (confusion)
    if idx > len(norm) * 0.7:
        return None

    # Calculer l'offset dans le texte original
    ratio = idx / max(len(norm), 1)
    orig_idx = int(ratio * len(block.text))

    # Trouver la vraie position dans le texte original
    lbl_words = label_str.split()
    for start_i in range(max(0, orig_idx - 10), min(len(block.text), orig_idx + 15)):
        chunk = _norm(block.text[start_i:start_i + len(label_str) + 5])
        if label_str in chunk:
            after = block.text[start_i + len(label_str):].strip()
            after = re.sub(r"^[\s:,\-\u2013\u2014]+", "", after).strip()
            # Rejeter si trop court ou si ca ressemble encore a un fragment du label
            if len(after) < 3:
                return None
            # Rejeter si tout en minuscules sans chiffres = probablement du texte label
            if after.islower() and not any(c.isdigit() for c in after):
                return None
            return after

    # Fallback : split sur le separateur
    for sep in [":", "-", "\u2013", "\u2014"]:
        parts = block.text.split(sep, 1)
        if len(parts) == 2:
            key_part = _norm(parts[0])
            if label_str in key_part or any(lw in key_part for lw in lbl_words if len(lw) > 3):
                after = parts[1].strip()
                if len(after) >= 3:
                    return after

    return None


def _clean_label_residue(text: str, label_str: str) -> str:
    """
    Supprime les fragments du label qui trainent en debut de valeur.
    Ex: label='cede a', value='ede a MMme MOULIN' -> 'MMme MOULIN'
    Cas arrive quand l'OCR coupe un mot en deux blocs.
    """
    cleaned = text.strip()
    norm_cleaned = _norm(cleaned)
    lbl_words = label_str.split()

    # Essayer de supprimer des prefixes de 1 a len(label)-1 mots
    for n_words in range(len(lbl_words), 0, -1):
        suffix_lbl = " ".join(lbl_words[-n_words:])
        if norm_cleaned.startswith(suffix_lbl):
            cleaned = cleaned[len(suffix_lbl):].strip()
            cleaned = re.sub(r"^[\s:,\-\u2013\u2014]+", "", cleaned).strip()
            break

    # Supprimer les caracteres parasites initiaux (lettres isolees, ponctuation)
    cleaned = re.sub(r"^[a-z\xc0-\xff]{1,3}\s+", "", cleaned).strip()
    cleaned = re.sub(r"^[\s:,\-\u2013\u2014']+", "", cleaned).strip()

    return cleaned


# ── Geometrie spatiale ──────────────────────────────────────────────────────
def _is_same_line(a: Block, b: Block, tolerance: float = 0.55) -> bool:
    """Vrai si deux blocs sont sur la même ligne (chevauchement vertical suffisant)."""
    overlap_y0 = max(a.y0, b.y0)
    overlap_y1 = min(a.y1, b.y1)
    min_h = min(a.h, b.h)
    if min_h == 0:
        return False
    return (overlap_y1 - overlap_y0) / min_h >= tolerance


def _find_value_block(
    label_block: Block,
    all_blocks: List[Block],
    img_w: float,
    img_h: float,
    max_gap_x_frac: float = 0.60,
    max_gap_y_frac: float = 0.15,  # C5: réduit de 30% à 15% — évite de capturer une section différente
) -> Optional[Block]:
    """
    Cherche le bloc-valeur le plus probable pour un label donné avec un score.
    
    Stratégie directionnelle (Correction 2 : priorité verticale améliorée) :
      1. À DROITE sur la même ligne → priorité maximale (formulaire horizontal)
      2. EN DESSOUS dans la même colonne → fallback (formulaire vertical, vieux DMPC)
         Sous-priorité : directement dessous dans la même colonne x0-x1
    """
    max_gap_x = max_gap_x_frac * img_w
    max_gap_y = max_gap_y_frac * img_h

    # ── Priorité 1 : voisin de droite sur la même ligne ──
    right_candidates = []
    for b in all_blocks:
        if b is label_block:
            continue
        if not _is_same_line(label_block, b, tolerance=0.4):
            continue
        gap_x = b.x0 - label_block.x1
        if 0 <= gap_x <= max_gap_x:
            right_candidates.append((gap_x, b))

    if right_candidates:
        right_candidates.sort(key=lambda x: x[0])
        return right_candidates[0][1]

    # ── Priorité 2a : voisin directement en-dessous (même colonne x) → DMPC anciens ──
    # Cas : label seul sur une ligne ("Commune"), valeur seule sur la ligne suivante
    direct_below = []
    for b in all_blocks:
        if b is label_block:
            continue
        gap_y = b.y0 - label_block.y1
        if gap_y < 0 or gap_y > max_gap_y:
            continue
        # Chevauchement horizontal fort : la valeur doit être dans la même colonne
        overlap_x = min(label_block.x1, b.x1) - max(label_block.x0, b.x0)
        label_w = max(label_block.w, 1)
        if overlap_x >= label_w * 0.3:  # au moins 30% de chevauchement
            # Score : favoriser les blocs proches verticalement et alignés horizontalement
            cx_dist = abs(label_block.cx - b.cx)
            score = gap_y * 1.0 + cx_dist * 0.5
            direct_below.append((score, b))

    if direct_below:
        direct_below.sort(key=lambda x: x[0])
        return direct_below[0][1]

    # ── Priorité 2b : voisin en dessous dans la même zone → fallback large ──
    below_candidates = []
    for b in all_blocks:
        if b is label_block:
            continue
        gap_y = b.y0 - label_block.y1
        if gap_y < 0 or gap_y > max_gap_y:
            continue
        cx_dist = abs(label_block.cx - b.cx)
        overlap = min(label_block.x1, b.x1) - max(label_block.x0, b.x0)
        if overlap > -label_block.w * 0.5 or cx_dist < img_w * 0.15:
            score = gap_y + (cx_dist * 2.0)
            below_candidates.append((score, b))

    if below_candidates:
        below_candidates.sort(key=lambda x: x[0])
        return below_candidates[0][1]

    return None



# ── Fonction principale ──────────────────────────────────────────────────────
def extract_fields_from_graph(
    ocr_results: List[Tuple],
    fields_to_extract: List[str],
    img_shape: Tuple[int, int],
    validate_fn=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Extraction principale par graphe de proximité spatial.

    Args:
        ocr_results       : sortie EasyOCR [(bbox, text, prob), ...]
        fields_to_extract : liste des champs à extraire sur cette page
        img_shape         : (hauteur, largeur) de l'image
        validate_fn       : fonction _validate_field(field, text) -> str|None

    Returns:
        dict {field: {"valeur", "zone", "brut", "methode", "label_trouve"}}
    """
    h, w = img_shape
    results: Dict[str, Dict[str, Any]] = {}

    # Construire les blocs filtrés (confiance minimale 10%)
    blocks = [
        Block(bbox, text, prob)
        for (bbox, text, prob) in ocr_results
        if prob >= 0.10 and text.strip()
    ]
    if not blocks:
        return results

    # ── Passe 1 : identifier tous les blocs contenant un label ──────────────
    label_blocks = []  # [(block, field, label_str)]
    for block in blocks:
        match = _find_label_match(block)
        if match is None:
            continue
        field, label_str = match
        if field not in fields_to_extract:
            continue
        label_blocks.append((block, field, label_str))

    # Trier par spécificité du label (plus long = plus prioritaire)
    label_blocks.sort(key=lambda x: len(x[2]), reverse=True)

    # ── Passe 2 : pour chaque label, trouver sa valeur ──────────────────────
    # Construire l'ensemble des blocs qui sont des labels (pour les exclure des valeurs)
    label_block_ids = {id(lb[0]) for lb in label_blocks}

    for label_block, field, label_str in label_blocks:
        if field in results:
            continue  # Deja trouve via un label plus specifique

        # Tentative A : valeur dans le meme bloc ("Dossier n° A09.4147")
        inline_val = _extract_inline_value(label_block, label_str)
        if inline_val:
            inline_val = _clean_label_residue(inline_val, label_str)
            validated = validate_fn(field, inline_val) if validate_fn else inline_val
            if validated:
                results[field] = {
                    "valeur": validated,
                    "zone": label_block.zone_frac(w, h),
                    "brut": label_block.text,
                    "methode": "graph_inline",
                    "label_trouve": label_str,
                    "confidence": 0.92,
                }
                continue

        # Tentative B : valeur dans le voisin directionnel le plus proche
        # On exclut les blocs qui sont eux-memes des labels
        value_block = _find_value_block(label_block, blocks, w, h)
        if value_block and id(value_block) not in label_block_ids:
            val_text = value_block.text.strip()
            # Nettoyer separateurs et fragments du label en debut de valeur
            val_text = re.sub(r"^[\s:,\-\u2013\u2014]+", "", val_text).strip()
            val_text = _clean_label_residue(val_text, label_str)
            if len(val_text) >= 3:
                validated = validate_fn(field, val_text) if validate_fn else val_text
                if validated:
                    results[field] = {
                        "valeur": validated,
                        "zone": value_block.zone_frac(w, h),
                        "brut": f"{label_block.text} -> {val_text}",
                        "methode": "graph_neighbor",
                        "label_trouve": label_str,
                        "confidence": 0.88,
                    }

    return results

def generate_anchor_crops(
    ocr_results: List[Tuple],
    fields_to_extract: List[str],
    img_shape: Tuple[int, int],
    zone_constraints: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Génère des bounding boxes (crops) autour des labels d'ancrage trouvés dans l'OCR.
    Au lieu de deviner où est la valeur par algorithme rigide, on crée une boîte relative
    autour du label (l'ancre) et on délègue la lecture de cette boîte au VLM (Ollama).

    Args:
        zone_constraints : dict optionnel {field: [x0_frac, y0_frac, x1_frac, y1_frac]}
            Si fourni, seuls les labels dont le CENTRE se trouve dans la zone autorisée
            sont considérés comme ancres valides. Cela évite qu'un mot "date" dans le
            corps graphique du plan soit utilisé à la place du label dans le cartouche.

    Retourne un dict: {field: {"zone": [...], "display_zone": [...], "label_trouve": "...", "brut": "..."}}
    """
    h, w = img_shape
    results: Dict[str, Dict[str, Any]] = {}

    blocks = [Block(bbox, text, prob) for (bbox, text, prob) in ocr_results if prob >= 0.10 and text.strip()]
    if not blocks:
        return results

    label_blocks = []
    for block in blocks:
        match = _find_label_match(block)
        if match:
            field, label_str = match
            if field not in fields_to_extract:
                continue
            # ── Filtre de zone structurelle (correction du mauvais ancrage) ──
            # Si zone_constraints est fourni pour ce champ, le label doit être
            # dans la zone attendue (évite les labels dans le corps graphique).
            if zone_constraints and field in zone_constraints:
                zc = zone_constraints[field]
                cx_frac = block.cx / w
                cy_frac = block.cy / h
                if not (zc[0] <= cx_frac <= zc[2] and zc[1] <= cy_frac <= zc[3]):
                    continue  # Ce label est hors de la zone structurelle → ignoré
            label_blocks.append((block, field, label_str))

    # Trier par spécificité du label (plus long = plus fiable)
    label_blocks.sort(key=lambda x: len(x[2]), reverse=True)

    # Nombre de caractères maximum attendus pour le champ (détermine la largeur du crop)
    expected_chars = {
        "commune": 30,
        "section": 5,
        "n_ordre": 15,
        "n_dossier": 12,
        "feuille": 5,
        "geometre": 35,
        "date": 12,
        "proprietaires_anciens": 60,
        "proprietaires_nouveaux": 60,
        "indication": 50,
        "signataires": 40,
    }

    # Nombre de lignes maximum attendues (détermine la hauteur du crop)
    expected_lines = {
        "commune": 2,
        "section": 1.5,
        "n_ordre": 1.5,
        "n_dossier": 1.5,
        "feuille": 1.5,
        "geometre": 2.5,
        "date": 1.5,
        "proprietaires_anciens": 3,
        "proprietaires_nouveaux": 3,
        "indication": 2.5,
        "signataires": 3,
    }

    # Marge gauche (en nombre de caractères) selon le champ.
    # Pour "commune", la valeur peut précéder le mot-clé (ex: "PRADES commune").
    # Pour les autres champs, la valeur est toujours à droite/dessous → marge gauche réduite.
    left_margin_chars = {
        "commune": 12,    # La commune peut être écrite avant "commune" sur certains docs
        "geometre": 5,
        "signataires": 3,
        "date": 2,        # La date est toujours à droite du mot "date"
        "section": 2,
        "feuille": 2,
        "n_ordre": 2,
        "n_dossier": 2,
        "echelle": 2,
        "indication": 3,
        "proprietaires_anciens": 3,
        "proprietaires_nouveaux": 3,
    }

    for label_block, field, label_str in label_blocks:
        if field in results:
            continue

        # Estimation de la taille de police du label (ancrage)
        char_w = label_block.w / max(1, len(label_block.text))
        line_h = label_block.h

        # Gardes-fous (si l'OCR a fait une boîte aberrante)
        char_w = min(max(char_w, w * 0.002), w * 0.04)
        line_h = min(max(line_h, h * 0.005), h * 0.04)

        # Facteur d'expansion : légèrement réduit pour éviter les zones trop larges.
        # 1.5 est suffisant pour l'écriture manuscrite, 1.8 causait des débordements.
        handwriting_multiplier = 1.5
        ext_w_pixels = expected_chars.get(field, 15) * char_w * handwriting_multiplier
        ext_h_pixels = expected_lines.get(field, 1.5) * line_h * handwriting_multiplier

        # Marge gauche : réduite selon le champ (évite de capturer des zones non pertinentes)
        left_chars = left_margin_chars.get(field, 3)
        x0_crop = max(0, label_block.x0 - (char_w * left_chars))
        y0_crop = max(0, label_block.y0 - (line_h * 0.5))

        # Extension droite et bas depuis la FIN de la boîte du label
        x1_crop = min(w, label_block.x1 + ext_w_pixels)
        y1_crop = min(h, label_block.y1 + ext_h_pixels)

        # Zone de crop (pour le VLM) — peut être assez large
        crop_zone = [x0_crop / w, y0_crop / h, x1_crop / w, y1_crop / h]

        # Zone d'affichage (plus petite, centrée sur le label + valeur attendue à droite)
        # On limite à la largeur du label + expected_chars à droite, et 1 ligne de hauteur
        disp_x0 = max(0, label_block.x0 - (char_w * 2)) / w
        disp_y0 = label_block.y0 / h
        disp_x1 = min(w, label_block.x1 + expected_chars.get(field, 15) * char_w) / w
        disp_y1 = min(h, label_block.y1 + line_h * expected_lines.get(field, 1.5)) / h
        display_zone = [disp_x0, disp_y0, disp_x1, disp_y1]

        results[field] = {
            "zone": crop_zone,
            "display_zone": display_zone,
            "label_trouve": label_str,
            "brut": label_block.text,
            "methode": "crop_ancrage",
        }

    return results

