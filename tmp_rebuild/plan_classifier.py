"""
plan_classifier.py Ã¢â‚¬â€ Extraction complÃƒÂ¨te des champs par type de plan cadastral.
Types : PVa (ProcÃƒÂ¨s-Verbal d'arpentage), PLa (Plan de Lotissement/arpentage), DMPC
Chaque champ est retournÃƒÂ© avec sa zone fractionnelle [x0,y0,x1,y1] pour le zoom dans app_validation.py
"""
import os, re, json, fitz, cv2
import numpy as np
import pandas as pd
import unicodedata
from typing import Dict, Any, List, Optional, Tuple


# ============================================================
# CORRECTION ENCODAGE MOJIBAKE
# Corrige les sequences corrompues (latin-1 relu comme UTF-8)
# produites par le passage WSL -> Windows curl -> Python.
# Exemples : "DressA(c)" -> "Dresse", "GeomA(R)tre" -> "Geometre"
# ============================================================
_MOJIBAKE_TABLE = [
    ("\u00c3\u00a9", "\u00e9"), ("\u00c3\u00a8", "\u00e8"), ("\u00c3\u00aa", "\u00ea"), ("\u00c3\u00ab", "\u00eb"),
    ("\u00c3\u00a0", "\u00e0"), ("\u00c3\u00a2", "\u00e2"), ("\u00c3\u00a4", "\u00e4"),
    ("\u00c3\u00ae", "\u00ee"), ("\u00c3\u00af", "\u00ef"),
    ("\u00c3\u00b4", "\u00f4"), ("\u00c3\u00b6", "\u00f6"), ("\u00c3\u00b9", "\u00f9"),
    ("\u00c3\u00bb", "\u00fb"), ("\u00c3\u00bc", "\u00fc"), ("\u00c3\u00a7", "\u00e7"),
    ("\u00c3\u0089", "\u00c9"), ("\u00c3\u0088", "\u00c8"), ("\u00c3\u008a", "\u00ca"),
    ("\u00c3\u0080", "\u00c0"), ("\u00c3\u0082", "\u00c2"),
    ("\u00c3\u008e", "\u00ce"), ("\u00c3\u008f", "\u00cf"),
    ("\u00c3\u0094", "\u00d4"), ("\u00c3\u0099", "\u00d9"), ("\u00c3\u009b", "\u00db"),
    ("\u00c3\u009c", "\u00dc"), ("\u00c3\u0087", "\u00c7"),
    ("\u00e2\u0080\u0099", "\u2019"), ("\u00e2\u0080\u0098", "\u2018"),
    ("\u00e2\u0080\u009c", "\u201c"), ("\u00e2\u0080\u009d", "\u201d"),
    ("\u00e2\u0080\u0093", "\u2013"), ("\u00e2\u0080\u0094", "\u2014"),
    ("\u00c2\u00b0", "\u00b0"), ("\u00c2\u00ab", "\u00ab"), ("\u00c2\u00bb", "\u00bb"),
]

def _reparer_encodage(texte: str) -> str:
    """
    Corrige les sequences Mojibake dans une chaine de caracteres.
    Methode 1 : re-encodage latin-1 puis decodage utf-8.
    Methode 2 : remplacement direct via table de caracteres frequents.
    Retourne le texte corrige, ou le texte original si rien a corriger.
    """
    if not texte or not isinstance(texte, str):
        return texte
    # Methode 1 : re-decodage (fonctionne si le texte est du latin-1 mal etiquete)
    try:
        corrige = texte.encode('latin-1').decode('utf-8')
        if corrige.count('\u00c3') < texte.count('\u00c3'):
            return corrige
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # Methode 2 : remplacement direct par table
    for mauvais, bon in _MOJIBAKE_TABLE:
        if mauvais in texte:
            texte = texte.replace(mauvais, bon)
    return texte


def _reparer_dict_recursif(obj):
    """
    Applique _reparer_encodage() recursivement sur tous les strings
    d'un dict/list imbrique (resultat JSON d'extraction).
    """
    if isinstance(obj, str):
        return _reparer_encodage(obj)
    elif isinstance(obj, dict):
        return {k: _reparer_dict_recursif(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_reparer_dict_recursif(item) for item in obj]
    return obj

# Import du module de vÃ©rification de cohÃ©rence (4 couches)
# Import du module d'extraction spatiale (graphe de proximite label->valeur)
try:
    from spatial_extractor import extract_fields_from_graph as _graph_extract, generate_anchor_crops as _generate_crops
    _SPATIAL_AVAILABLE = True
    _GENERATE_CROPS_AVAILABLE = True
    print("[PlanClassifier] Graphe spatial et Ancrage disponibles.")
except ImportError:
    _SPATIAL_AVAILABLE = False
    _GENERATE_CROPS_AVAILABLE = False
    _graph_extract = None
    _generate_crops = None
    print("[PlanClassifier] spatial_extractor.py introuvable - extraction par graphe desactivee.")


try:
    from coherence_checker import check_coherence as _check_coherence
    _COHERENCE_AVAILABLE = True
except ImportError:
    _COHERENCE_AVAILABLE = False
    print("[PlanClassifier] coherence_checker.py introuvable â€” vÃ©rification de cohÃ©rence dÃ©sactivÃ©e.")

try:
    from gliner import GLiNER
    import torch
    _GLINER_AVAILABLE = True
    print("[PlanClassifier] Ã¢Å“â€¦ GLiNER est installÃƒÂ© et sera utilisÃƒÂ© pour l'extraction sÃƒÂ©mantique.")
except Exception as e:
    _GLINER_AVAILABLE = False
    print(f"[PlanClassifier] Ã¢Å¡Â Ã¯Â¸ï¿½ Impossible de charger GLiNER. Erreur exacte : {e}")
    print("[PlanClassifier] Ã¢Å¡Â Ã¯Â¸ï¿½ Fallback sur les expressions rÃƒÂ©guliÃƒÂ¨res.")

_gliner_model = None
def get_gliner_model():
    global _gliner_model
    if _gliner_model is None and _GLINER_AVAILABLE:
        print("  [PlanClassifier] Ã°Å¸Â§Â  Chargement du modÃƒÂ¨le sÃƒÂ©mantique GLiNER...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # On utilise une version small, trÃƒÂ¨s rapide et lÃƒÂ©gÃƒÂ¨re
        try:
            _gliner_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1").to(device)
            print("  [PlanClassifier] Ã¢Å“Â¨ ModÃƒÂ¨le GLiNER chargÃƒÂ© avec succÃƒÂ¨s.")
        except Exception as e:
            print(f"  [PlanClassifier] Ã¢ï¿½Å’ Erreur lors du chargement de GLiNER : {e}")
            return None
    return _gliner_model

# Ã¢â€â‚¬Ã¢â€â‚¬ Couleurs annotation Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
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

# â”€â”€ Liste globale des gÃ©omÃ¨tres experts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
GEOMETRES_CONNUS = ["DUPUY", "HARROIS", "RACAT", "SERRET", "CEYTE", "BARRIAL", "ROBERT"]

# â”€â”€ ModÃ¨le VLM adaptatif selon le type de document â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# llama3.2-vision (11B) est plus prÃ©cis mais plus lent.
# On l'utilise pour les documents non-standards oÃ¹ llava (7B) hallucine.
_OLLAMA_MODEL_FOR_TYPE = {
    "DMPC":    "llava",             # Formulaire structurÃ© â†’ llava rapide suffit
    "PVa":     "llama3.2-vision",   # Texte libre tapÃ© â†’ meilleure comprÃ©hension sÃ©mantique
    "PLa":     "llava",             # Similaire au DMPC
    "CROQUIS": "llama3.2-vision",   # Documents anciens â†’ llama3.2-vision obligatoire
    "GENERIC": "llama3.2-vision",   # Inconnu â†’ prendre le meilleur
    "DEFAULT": "llava",
}

# â”€â”€ Prompts VLM spÃ©cialisÃ©s par type et par champ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_VLM_PROMPTS_BY_TYPE = {
    "DMPC": {
        "commune":  "Quel est le nom de la commune Ã©crit dans cette case ? RÃ©ponds uniquement par le nom de la commune, sans phrase.",
        "section":  "Quelle est la lettre de section cadastrale dans cette case ? RÃ©ponds uniquement par la lettre (ex: A, B, AB).",
        "feuille":  "Quel est le numÃ©ro de feuille dans cette case ? RÃ©ponds uniquement par le numÃ©ro.",
        "n_ordre":  "Quel est le numÃ©ro d'ordre ou numÃ©ro DA dans cette case ? RÃ©ponds uniquement par le numÃ©ro.",
        "geometre": "Quel est le nom du gÃ©omÃ¨tre ou cabinet Ã©crit dans cette case ? RÃ©ponds uniquement par le nom, sans phrase.",
        "date":     "Quelle est la date Ã©crite dans cette case ? RÃ©ponds uniquement par la date (format JJ/MM/AAAA).",
        "_default": "Extrais UNIQUEMENT la valeur Ã©crite dans cette case. NE FAIS AUCUNE PHRASE. RÃ©ponds juste par le texte lu. Si illisible, rÃ©ponds 'vide'.",
    },
    "PVa": {
        "commune":  "Ce document est un procÃ¨s-verbal de bornage. Quel est le nom de la commune ? Il peut apparaÃ®tre sous la forme 'commune de X', 'sur le territoire de X'. RÃ©ponds uniquement par le nom.",
        "date":     "Ce document est un acte de gÃ©omÃ¨tre. Cherche la date Ã  laquelle il a Ã©tÃ© dressÃ©. Elle apparaÃ®t souvent sous 'Fait Ã  [ville], le [date]'. RÃ©ponds uniquement par la date (JJ/MM/AAAA ou JJ mois AAAA).",
        "geometre": "Ce document est un procÃ¨s-verbal de gÃ©omÃ¨tre-expert. Quel est le nom du gÃ©omÃ¨tre qui a dressÃ© ce document ? Il apparaÃ®t aprÃ¨s 'le soussignÃ© gÃ©omÃ¨tre-expert' ou 'dressÃ© par'. RÃ©ponds uniquement par le nom.",
        "section":  "Dans ce procÃ¨s-verbal, quelle est la section cadastrale ? Elle apparaÃ®t sous la forme 'section [lettre]'. RÃ©ponds uniquement par la lettre.",
        "proprietaires_anciens": "Qui est le demandeur ou propriÃ©taire actuel ? Il apparaÃ®t aprÃ¨s 'Ã  la demande de' ou 'requis par'. RÃ©ponds uniquement par le nom.",
        "proprietaires_nouveaux": "Qui est le nouveau propriÃ©taire ou bÃ©nÃ©ficiaire ? Il apparaÃ®t aprÃ¨s 'au profit de' ou 'acquis par'. RÃ©ponds uniquement par le nom.",
        "_default": "Ce document est un acte gÃ©omÃ¨tre. Extrais UNIQUEMENT la valeur de '{field}'. NE FAIS AUCUNE PHRASE. RÃ©ponds juste par le texte lu. Si illisible, rÃ©ponds 'vide'.",
    },
    "CROQUIS": {
        "commune":  "Ce document est un ancien plan cadastral ou croquis. Le nom de la commune peut Ãªtre en tampon (avec des points entre les lettres, ex: S.A.I.N.T = SAINT). Quel est le nom de la commune ? RÃ©ponds uniquement par le nom.",
        "date":     "Ce document est un ancien plan. La date peut Ãªtre manuscrite ou tamponnÃ©e. Cherche une date de dressÃ© ou signÃ©. RÃ©ponds uniquement par la date.",
        "geometre": "Ce document est un ancien plan cadastral. Le gÃ©omÃ¨tre est identifiÃ© par son tampon ou signature. Son nom peut Ãªtre en tampon avec des points entre les lettres â€” ignore ces points. Quel est le nom ? RÃ©ponds uniquement par le nom.",
        "section":  "Dans ce plan cadastral, quelle est la section cadastrale ? Souvent une lettre manuscrite ou tamponnÃ©e. RÃ©ponds uniquement par la lettre.",
        "echelle":  "Dans ce plan, quelle est l'Ã©chelle ? Format '1/500', '1/1000', '1/2000'. RÃ©ponds uniquement par l'Ã©chelle.",
        "_default": "Ce document est un ancien plan cadastral. Le texte peut Ãªtre manuscrit ou en tampon. Si tu vois des points entre les lettres (ex: A.U.B.E.N.A.S), lis le mot sans les points. Extrais UNIQUEMENT la valeur de '{field}'. Si illisible, rÃ©ponds 'vide'.",
    },
    "GENERIC": {
        "_default": "Tu es un expert gÃ©omÃ¨tre. Ce document est un plan ou acte cadastral. Extrais UNIQUEMENT la valeur de '{field}'. NE FAIS AUCUNE PHRASE. RÃ©ponds juste par la valeur brute. Si illisible, rÃ©ponds 'vide'.",
    },
    "DEFAULT": {
        "_default": "Extrais UNIQUEMENT la valeur de '{field}' sur l'image. NE FAIS AUCUNE PHRASE. RÃ©ponds juste par le texte lu. Si illisible, rÃ©ponds 'vide'.",
    },
}

def _get_vlm_model(type_plan: str) -> str:
    """Retourne le modÃ¨le Ollama optimal pour un type de document donnÃ©."""
    return _OLLAMA_MODEL_FOR_TYPE.get(type_plan, _OLLAMA_MODEL_FOR_TYPE["DEFAULT"])

def _get_vlm_prompt(type_plan: str, field: str) -> str:
    """Retourne le prompt VLM spÃ©cialisÃ© pour un type de document et un champ donnÃ©s."""
    prompts_for_type = _VLM_PROMPTS_BY_TYPE.get(type_plan, _VLM_PROMPTS_BY_TYPE["DEFAULT"])
    prompt = prompts_for_type.get(field, prompts_for_type.get("_default", _VLM_PROMPTS_BY_TYPE["DEFAULT"]["_default"]))
    return prompt.replace("{field}", field)


# Ã¢â€â‚¬Ã¢â€â‚¬ Patterns CONTEXTUELS par champ Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# Chaque pattern capture la valeur dans la meme phrase que le label.
# Groupe 1 = la valeur attendue.
_AP = r"[A-Za-z\xc0-\xff\s\-]"   # lettre + espace + tiret (helper)

CONTEXTUAL_PATTERNS: Dict[str, List[str]] = {
    "commune": [
        r"(?:commune\s*(?:d[e']?\s*|d\b\s*)?|sur\s+la\s+commune\s*(?:d[e']?\s*|d\b\s*)?|en\s+la\s+commune\s*(?:d[e']?\s*|d\b\s*)?|territoire\s*(?:d[e']?\s*|d\b\s*)?|c\.n\.e\.)\s*[:\s]*([A-Za-z\xc0-\xff][A-Za-z\xc0-\xff\s\-]{2,50}?)(?:\s*[-,]|\s+(?:section|feuille|parcelle|lieudit|sect\.|ech)|$|\n)",
        r"commune\s*:\s*([A-Za-z\xc0-\xff][A-Za-z\xc0-\xff\s\-]{2,50}?)(?:\s*[-,]|\n|$)",
        r"\bcommune\s*[:\-\n]*\s*([A-Za-z\xc0-\xff][A-Za-z\xc0-\xff\s\-]{2,50}?)(?:\s*[-,]|\s+(?:section|feuille|parcelle|lieudit|sect\.|ech)|$|\n)",
    ],
    "n_ordre": [
        # Patterns STRICTS : exigent le label explicite devant la valeur
        r"(?:num[e\xe9]ro\s+d['â€™]ordre\s*(?:du\s*document\s*d['â€™]arpentage)?|n[o\xb0Â°]\s*d['â€™]ordre\s*(?:du\s*document\s*d['â€™]arpentage)?|da\s*n[o\xb0Â°]|n[o\xb0Â°]\s*d['â€™]arpentage|n[o\xb0Â°]\s*d['â€™]ordre)\s*[:\-]?\s*([A-Z0-9][A-Za-z0-9\.\-\/_ ]{1,20}(?:\s*\(\d+\))?)",
        r"(?:num[e\xe9]ro\s+d['â€™]ordre\s*(?:du\s*document\s*d['â€™]arpentage)?|n[o\xb0Â°]\s*d['â€™]ordre\s*(?:du\s*document\s*d['â€™]arpentage)?|num[e\xe9]ro\s*d['â€™](?:ordre|arpentage)|da\s*n[o\xb0Â°])\s*[:\-]?\s*(\d{1,6}[A-Za-z]?(?:\s*[_\-]\s*\d{1,6}[A-Za-z]?)*(?:\s*\(\d+\))?)",
    ],
    "n_dossier": [
        # n_dossier = rÃ©fÃ©rence du dossier (plus large, capte le numÃ©ro de dossier)
        r"(?:dossier\s*(?:n[o\xb0Â°])?\s*:?|r[e\xe9]f[e\xe9]rence\s*(?:n[o\xb0Â°])?\s*:?|affaire\s*n[o\xb0Â°]\s*:?)\s*([A-Z0-9][A-Za-z0-9\.\-\/]{2,20})",
        r"[Dd]ossier\s+([A-Z][0-9]{2,4}[.\-]?[0-9]{2,6})",
        # Format type 1992C100001 ou A09-032
        r"\b([A-Z]?\d{4}[A-Z]\d{4,8})\b",
        r"\b([A-Z]\d{2}[.\-]\d{3,5})\b",
    ],
    "section": [
        r"(?:section\s+(?:cadastrale\s+)?n[o\xb0]?|section\s*:)\s*([A-Z]{1,2}\d{0,2})\b",
        r"\bsection\s+([A-Z]{1,2}\d{0,2})\b",
    ],
    "feuille": [
        r"(?:feuille\s+(?:cadastrale\s+)?n[o\xb0]?|feuille\s*:)\s*([A-Z0-9]{1,6})\b",
        r"\bfeuille\s+([A-Z0-9]{1,6})\b",
    ],
    "date": [
        # Contexte avec mot clÃ© Date (ajout)
        r"date\s*[:\-;,\.]?\s*(.+)",
        # Contexte explicite (prioritÃ© 1) â€” "Ã©tabli/dressÃ©/signÃ© le ..."
        r"(?:(?:[e\xe9]tabli|dress[e\xe9]|[e\xe9]dit[e\xe9]|sign[e\xe9]|fait)\s+le|en\s+date\s+du|fait\s+[a\xe0])\s+(\d{1,2}\s*[/\-\.]\s*\d{1,2}\s*[/\-\.]\s*\d{2,4})",
        r"(?:(?:[e\xe9]tabli|dress[e\xe9]|[e\xe9]dit[e\xe9]|sign[e\xe9]|fait)\s+le|en\s+date\s+du)\s+(\d{1,2}\s+(?:janvier|f[e\xe9]vrier|mars|avril|mai|juin|juillet|ao[u\xfb]t|septembre|octobre|novembre|d[e\xe9]cembre)\s+\d{4})",
        # Contexte partiel (prioritÃ© 2) â€” "le JJ/MM/AAAA"
        r"\ble\s+(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})\b",
        r"\ble\s+(\d{1,2}\s+(?:janvier|f[e\xe9]vrier|mars|avril|mai|juin|juillet|ao[u\xfb]t|septembre|octobre|novembre|d[e\xe9]cembre)\s+\d{4})\b",
        # Fallback : date nue en bas de document (format JJ/MM/AAAA)
        r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4})\b",
    ],
    "echelle": [
        r"(?:[e\xe9]chelle|ech\.?)\s*[:\s]*(1\s*/\s*\d{3,5})",
        r"(?:[e\xe9]chelle|ech\.?)\s*[:\s]*(\d{3,5}(?:\s*/\s*\d{3,5})?)",
    ],
    "geometre": [
        r"(?:g[e\xe9]om[e\xe8]tre[\s\-]expert|cabinet\s+de\s+g[e\xe9]om[e\xe8]tre|le\s+soussign[e\xe9]\s+g[e\xe9]om[e\xe8]tre|g[e\xe9]om[e\xe8]tre\s*:|expert\s*:?|dresse\s+par\s*:?|pour\s+mm\.?)[,\s:\-]+(?:M\.|Mme|Monsieur\s+)?([A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-\.]{2,60}?)(?:\s*,|\s*\n|\s*\.|$)",
        r"(?i)\b(?:RACAT\s+et\s+CEYTE)\b",
        r"(?i)\b(?:CAB\.\s+TOPO.*?YTE|CABINET.*?YTE|[CE]?YTE)\b",
    ],
    "signataires": [
        r"(?:certifi[e\xe9]\s+exact|vu\s+et\s+approuv[e\xe9]|soussign[e\xe9](?:e?s)?)[,\s:]+([A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-\.]{2,60}?)(?:\s*,|\s*\n|\s*\.|$)",
    ],
    "proprietaires_anciens": [
        # Formulations explicites avec label clair
        r"(?:anciens?\s+propri[e\xe9]taires?|propri[e\xe9]taire[s]?\s+(?:sortant[s]?|actuel[s]?|vendeur[s]?)|c[e\xe9]dant[s]?|vendeur[s]?)\s*[:\-]?\s*"
        r"((?:M\.|Mme\.?|Monsieur|Madame|Mlle\.?|M\s+et\s+Mme|M\s*\.\s*et\s*Mme|Mme\s+et\s+M\.?)\s*"
        r"[A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-]{2,60}?)(?:\s*[,;]|\s*\n|$)",
        # Requiert par / a la demande de = demandeur = ancien proprietaire sur un PVa
        r"(?:[a\xe0]\s+la\s+demande\s+de|requis\s+par|[a\xe0]\s+la\s+requ[e\xea]te\s+de)\s+"
        r"((?:M\.|Mme\.?|Monsieur|Madame|Mlle\.?)[^\n,;]{2,60}?)(?:[,;]|\n|$)",
        # Vend / cede a 
        r"(?:d[e\xe9]nomm[e\xe9](?:e?s)?|ci[\-\s]dessus\s+nomm[e\xe9](?:e?s)?)\s+"
        r"((?:M\.|Mme\.?|Monsieur|Madame|Mlle\.?)[^\n,;]{2,60}?)(?:[,;]|\n|$)",
        # Propriete de / appartenant a
        r"(?:propri[e\xe9]t[e\xe9]\s+de|appartenant\s+[a\xe0])\s+"
        r"((?:M\.|Mme\.?|Monsieur|Madame|Mlle\.?)[^\n,;]{2,60}?)(?:[,;]|\n|$)",
    ],
    "proprietaires_nouveaux": [
        # Formulations explicites avec label clair
        r"(?:nouveaux?\s+propri[e\xe9]taires?|propri[e\xe9]taire[s]?\s+(?:entrant[s]?|b[e\xe9]n[e\xe9]ficiaire[s]?)|acqu[e\xe9]reurs?|b[e\xe9]n[e\xe9]ficiaires?|acheteurs?)\s*[:\-]?\s*"
        r"((?:M\.|Mme\.?|Monsieur|Madame|Mlle\.?|M\s+et\s+Mme|M\s*\.\s*et\s*Mme|Mme\s+et\s+M\.?)\s*"
        r"[A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-]{2,60}?)(?:\s*[,;]|\s*\n|$)",
        # Au profit de / en faveur de / au benefice de
        r"(?:au\s+profit\s+de|en\s+faveur\s+de|au\s+b[e\xe9]n[e\xe9]fice\s+de|est\s+attribu[e\xe9]\s+[a\xe0])\s+"
        r"((?:M\.|Mme\.?|Monsieur|Madame|Mlle\.?)[^\n,;]{2,60}?)(?:[,;]|\n|$)",
        # Acquis par
        r"(?:acquis\s+par|achet[e\xe9]\s+par|c[e\xe9]d[e\xe9]\s+[a\xe0])\s+"
        r"((?:M\.|Mme\.?|Monsieur|Madame|Mlle\.?)[^\n,;]{2,60}?)(?:[,;]|\n|$)",
    ],
    "indication": [
        # Certifications d'arpentage (extrait la ligne valide)
        r"(?:a\s+[e\xe9]t[e\xe9]\s+[e\xe9]tabli[\s\S]{0,200}?)(en\s+conformit[e\xe9]\s+d['â€™]un\s+piquetage[^\n]*|d['â€™]apr[e\xe8]s\s+les\s+indications[^\n]*|d['â€™]apr[e\xe8]s\s+un\s+plan\s+d['â€™]arpentage[^\n]*)",
        
        r"(?:objet\s*(?:du\s*document)?\s*:|nature\s+des\s+op[e\xe9]rations\s*:)\s*([^\n]{5,150})",
        # Capturer "bornage/piquetage contradictoire du JJ mois AAAA" en entier
        r"((?:piquetage|bornage|reconnaissance\s+de\s+limites|accord\s+amiable)"
        r"(?:\s+contradictoire)?[^\n]{0,80})",
        # Fallback : "objet : ..." sans label structurÃ©
        r"(?:division|lotissement|distraction|remembrement|r[e\xe9]union)[^\n]{0,80}([^\n]{5,100})",
    ],
}

# Regex dynamique pour les gÃ©omÃ¨tres connus
import re
_geo_pattern = r"(?i)\b(" + "|".join([re.escape(g) for g in GEOMETRES_CONNUS]) + r")\b"
CONTEXTUAL_PATTERNS["geometre"].append(_geo_pattern)

# Liste de mots-clÃƒÂ©s pour le fallback d'extraction classique
KEYWORDS: Dict[str, List[str]] = {
    "commune": ["commune", "territoire"],
    "n_ordre": ["nÃ‚Â° d'ordre", "numero d'ordre", "da nÃ‚Â°", "dossier", "affaire", "rÃƒÂ©fÃƒÂ©rence"],
    "n_dossier": ["dossier", "rÃƒÂ©fÃƒÂ©rence"],
    "section": ["section", "sect.", "sect"],
    "feuille": ["feuille"],
    "date": ["etabli le", "dresse le", "edite le", "signe le", "fait le", "date", "fait a"],
    "echelle": ["echelle", "ech.", "ech"],
    "geometre": ["geometre expert", "cabinet", "dresse par", "geometre"],
    "signataires": ["certifie exact", "approuve", "soussigne"],
    "proprietaires_anciens": ["anciens proprietaires", "cedant", "vendeur"],
    "proprietaires_nouveaux": ["nouveaux proprietaires", "acquereur"],
    "indication": ["objet", "nature", "piquetage", "bornage", "division"],
}


def _normalize_ocr_text(text: str) -> str:
    """
    Correction 1 : Normalisation Unicode GLOBALE.
    Unifie les variantes de ponctuations que l'OCR produit sur vieux documents.
    - Apostrophes typographiques (' â€™ Ê¼ `) â†’ apostrophe droite (')
    - Guillemets (Â« Â» â€œ â€) â†’ guillemet droit (")
    - Tirets longs (â€“ â€”) â†’ tiret simple (-)
    - Espace insÃ©cable â†’ espace normale
    Cette normalisation s'applique une seule fois avant toutes les Regex.
    """
    replacements = [
        ("\u2019", "'"), ("\u2018", "'"), ("\u02bc", "'"), ("\u0060", "'"),
        ("\u00ab", '"'), ("\u00bb", '"'), ("\u201c", '"'), ("\u201d", '"'),
        ("\u2013", "-"), ("\u2014", "-"),
        ("\u00a0", " "),  # espace insecable
        ("N'", "N\u00b0"),  # OCR frequemment confond N' et NÂ° sur vieux docs
        ("NÂ°", "N\u00b0"),
        ("n'", "n\u00b0"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def _find_field_contextual(
    field_type: str,
    ocr_results: List[Tuple],
    img_shape: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    """
    Extraction contextuelle : applique les CONTEXTUAL_PATTERNS sur le texte
    complet de la zone OCR (concatÃƒÂ©nÃƒÂ©), puis retrouve la bbox de la valeur.

    Avantages vs l'ancienne _find_field :
    - Lit le label ET la valeur dans la mÃƒÂªme phrase Ã¢â€ â€™ pas de faux voisins.
    - La date n'est capturÃƒÂ©e que si prÃƒÂ©cÃƒÂ©dÃƒÂ©e de "ÃƒÂ©tabli le / dressÃƒÂ© le ..."
      (pas "nÃƒÂ©e le", pas "signÃƒÂ© acte le").
    - La commune n'est capturÃƒÂ©e que si prÃƒÂ©cÃƒÂ©dÃƒÂ©e de "commune de".
    """
    h, w = img_shape
    patterns = CONTEXTUAL_PATTERNS.get(field_type, [])
    if not patterns:
        return None

    # Construire la liste d'items avec bbox
    items = []
    for (bbox, text, prob) in ocr_results:
        if prob < 0.10:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({
            "text": text,
            "bx0": min(xs), "by0": min(ys),
            "bx1": max(xs), "by1": max(ys),
        })

    if not items:
        return None

    # Texte complet de la zone (sÃ©parateurs newline pour garder la structure)
    # â”€â”€ Correction 1 : normalisation Unicode avant TOUTE Regex â”€â”€
    full_text = "\n".join(_normalize_ocr_text(it["text"]) for it in items)

    for pat in patterns:
        m = re.search(pat, full_text, re.IGNORECASE)
        if not m:
            continue
        val_raw = m.group(1).strip()
        # Nettoyage
        val_raw = re.sub(r'\s+', ' ', val_raw).strip(" ,;:-")
        if len(val_raw) < 1:
            continue
        validated = _validate_field(field_type, val_raw)
        if not validated:
            continue

        # Trouver l'item OCR qui contient la valeur pour la bbox
        best_item = None
        for it in items:
            if validated.lower()[:8] in it["text"].lower() or it["text"].lower()[:8] in validated.lower():
                best_item = it
                break
        if not best_item:
            # Chercher l'item qui contient le dÃƒÂ©but de la valeur
            for it in items:
                if any(w2 in it["text"].lower() for w2 in validated.lower().split()[:2] if len(w2) > 3):
                    best_item = it
                    break
        if not best_item and items:
            best_item = items[0]

        zone_f = [
            best_item["bx0"] / w, best_item["by0"] / h,
            best_item["bx1"] / w, best_item["by1"] / h,
        ]
        return {"valeur": validated, "zone": zone_f, "brut": m.group(0).strip()}

    return None

# Ã¢â€â‚¬Ã¢â€â‚¬ Zones de recherche spatialisÃƒÂ©es selon le type de plan Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# Format: [x0_frac, y0_frac, x1_frac, y1_frac] (fractions de la page)
ZONES_PAR_TYPE: Dict[str, Dict[str, List[float]]] = {
    "PVa": {   # PV de bornage/division Ã¢â‚¬â€ texte entiÃƒÂ¨rement tapÃƒÂ©
        "commune":    [0.0, 0.0, 1.0, 0.60],
        "n_ordre":    [0.0, 0.0, 1.0, 0.60],
        "n_dossier":  [0.0, 0.0, 1.0, 0.60],
        "section":    [0.0, 0.0, 1.0, 0.60],
        "feuille":    [0.0, 0.0, 1.0, 0.60],
        "date":       [0.0, 0.0, 1.0, 1.0],
        "echelle":    [0.0, 0.0, 1.0, 1.0],
        "geometre":   [0.0, 0.0, 1.0, 1.0],
        "signataires":[0.0, 0.0, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.0, 1.0, 1.0],
        "proprietaires_nouveaux": [0.0, 0.0, 1.0, 1.0],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 1.0, 1.0],
    },
    "PLa": {   # Document d'arpentage (DA) Ã¢â‚¬â€ Cartouche 1/3 haut
        "commune":    [0.0, 0.0, 0.60, 0.35],  # 1/3 haut, partie gauche
        "n_ordre":    [0.0, 0.0, 1.0, 0.50],   # Full width top 50%
        "section":    [0.0, 0.0, 1.0, 0.50],   # Full width top 50%
        "feuille":    [0.40, 0.0, 1.0, 0.45],  # 1/3 haut, partie droite
        "date":       [0.40, 0.0, 1.0, 0.45],  # 1/3 haut, partie droite
        "echelle":    [0.40, 0.0, 1.0, 0.45],  # 1/3 haut, partie droite
        "geometre":   [0.0, 0.55, 1.0, 1.0],   # bas de page / signatures
        "signataires":[0.0, 0.55, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.25, 1.0, 0.80],
        "proprietaires_nouveaux": [0.0, 0.25, 1.0, 0.80],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 0.60, 0.45],
    },
    "DMPC": {   # DMPC (jaunÃƒÂ¢tre, tapÃƒÂ© + manuscrit)
        "commune":    [0.0, 0.0, 0.70, 0.35],  # en premier ÃƒÂ  gauche
        "section":    [0.0, 0.0, 1.0, 0.50],   # ensuite ÃƒÂ  gauche
        "feuille":    [0.0, 0.0, 0.70, 0.45],  # ensuite ÃƒÂ  gauche
        "echelle":    [0.0, 0.0, 0.70, 0.50],  # ensuite ÃƒÂ  gauche
        "n_ordre":    [0.0, 0.0, 1.0, 0.50],  # en haut ÃƒÂ  droite (DA)
        "n_dossier":  [0.45, 0.0, 1.0, 0.40],
        "date":       [0.0, 0.55, 1.0, 1.0],   # bas de page
        "geometre":   [0.35, 0.55, 1.0, 1.0],  # bas ÃƒÂ  droite (dressÃƒÂ© par)
        "signataires":[0.0, 0.55, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.55, 1.0, 1.0], # bas de page (noms des prop)
        "proprietaires_nouveaux": [0.0, 0.55, 1.0, 1.0],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.55, 1.0, 1.0],   # propositions certifiÃƒÂ© par
    },
    "GENERIC": {   # Fallback
        "commune":    [0.0, 0.0, 0.70, 0.40],
        "n_ordre":    [0.0, 0.0, 1.0, 0.50],
        "section":    [0.0, 0.0, 1.0, 0.50],
        "feuille":    [0.0, 0.0, 0.70, 0.50],
        "date":       [0.0, 0.50, 1.0, 1.0],
        "echelle":    [0.4, 0.0, 1.0, 0.50],
        "geometre":   [0.3, 0.50, 1.0, 1.0],
        "signataires":[0.3, 0.50, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.15, 0.8, 0.75],
        "proprietaires_nouveaux": [0.0, 0.15, 0.8, 0.75],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 0.70, 0.50],
    },
    "CROQUIS": {   # Vieux plans, croquis, bornages sans layout fixe
        "commune":    [0.0, 0.0, 1.0, 1.0],
        "n_ordre":    [0.0, 0.0, 1.0, 1.0],
        "section":    [0.0, 0.0, 1.0, 1.0],
        "feuille":    [0.0, 0.0, 1.0, 1.0],
        "date":       [0.0, 0.0, 1.0, 1.0],
        "echelle":    [0.0, 0.0, 1.0, 1.0],
        "geometre":   [0.0, 0.0, 1.0, 1.0],
        "signataires":[0.0, 0.0, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.0, 1.0, 1.0],
        "proprietaires_nouveaux": [0.0, 0.0, 1.0, 1.0],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 1.0, 1.0],
    },
}


# â”€â”€â”€ Mots-clÃ©s qui ne peuvent PAS Ãªtre des noms de personnes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_SIGNATAIRE_BLACKLIST = {
    "COMMUNE", "SECTION", "FEUILLE", "ECHELLE", "PLAN", "CADASTRE", "DMPC",
    "DATE", "BORNAGE", "ARPENTAGE", "LOTISSEMENT", "DIVISION", "GEOMETRE",
    "GEOMETRES", "GEOMETRES EXPERTS", "GEOMETRES-EXPERTS", "EXPERT", "EXPERTS",
    "RECONNAISSANCE DE LIMITES", "ACCORD AMIABLE", "DESIGNAIION DES PARIIES",
    "DESIGNATION DES PARTIES", "OBJET DE", "CLAUSES GENERALES", "ACCORD DES PARTIES",
    "DOCUMENIS ANALYSES", "DEFINITION DES LIMIIES", "DEBAI CONIRADICIOIRE",
    "ORDRE DES", "DOCUMENT DE CONSERVATION", "DOCUMENT D ARPENTAGE",
    "FRANCE", "ARDECHE", "DROME", "GARD", "LOIRE", "RHONE",
    "RECONAISSANCE", "REIABLISSEMENI DES BORNES",
    "CLAUSES", "GENERALES", "ACCORD", "AMIABLE",
    "VALS LES BAINS", "VALLON PONT", "VALLON PONT D ARC",
    "GUILHERAND GRANGES", "GUILHERAND", "ANNONAY", "AUBENAS", "PRIVAS",
}

def _validate_field(field_type: str, val: str, commune_db=None) -> Optional[str]:
    """Valide et nettoie la valeur d'un champ selon des rÃƒÂ¨gles strictes."""
    import re
    if not val:
        return None
    val = str(val).strip()
    if not val:
        return None
        
    # Les sections et feuilles peuvent faire 1 seul caractÃƒÂ¨re (ex: "C", "1")
    if field_type not in ("section", "feuille") and len(val) < 2:
        return None
    val_norm = val.lower()

    # Rejet des phrases conversationnelles typiques des VLM (ex: "est l'un des Ã©lÃ©ments textuels prÃ©sents dans l'image")
    if any(phrase in val_norm for phrase in ["est l'un", "l'image", "limage", "la photo", "je vois", "il y a", "le texte", "prÃ©sent dans", "liste brute", "mots ou", "nombres sont", "ne peux pas", "trop petite", "trop petit", "illisible", "avec prÃ©cision"]):
        # Si le VLM a mis la rÃ©ponse entre guillemets, on l'extrait
        m = re.search(r'"([^"]+)"|\'([^\']+)\'', val)
        if m:
            val = m.group(1) or m.group(2)
            val_norm = val.lower()
        else:
            return None

    # Rejets universels : si ÃƒÂ§a ressemble ÃƒÂ  un autre label, c'est invalide
    LABEL_INTERDITS = [
        "commune", "section", "parcelle", "ÃƒÂ©chelle", "echelle", "lieudit",
        "contenance", "surface", "gÃƒÂ©omÃƒÂ¨tre", "geometre", "propriÃƒÂ©taire",
        "procÃƒÂ¨s-verbal", "document d'arpentage", "bornage", "plan de", "feuille", "nÂ°", "numero", "date", "d",
    ]
    for interdit in LABEL_INTERDITS:
        if field_type == interdit or (field_type == "echelle" and interdit == "Ã©chelle"):
            continue
        if val_norm == interdit.lower():
            return None

    if field_type == "section":
        # Si l'OCR a fusionnÃ© "SECTION AD"
        m_sec = re.match(r'(?i)^section\s*(?::|\-)?\s*(.+)$', val)
        if m_sec:
            val = m_sec.group(1)
        elif val_norm == "section":
            return None
            
        # Nettoyage des parasites courants de l'OCR sur les lettres manuscrites (barres de Z vues comme _ ou -)
        # Et on retire les espaces au milieu (ex: "A Z" -> "AZ")
        val = re.sub(r'[\s_\.\-]+', '', val)
        
        # Rejeter fermement si le texte contient d'autres mots longs (pour Ã©viter d'extraire "DE" de "COMMUNE DE")
        if len(val) > 4 and not re.match(r'^[A-Za-z\d]+$', val):
            return None
            
        # Match stricte: la chaÃ®ne entiÃ¨re doit Ãªtre une section valide (1 ou 2 lettres, ou jusqu'Ã  3 chiffres)
        m = re.search(r'\b([A-Za-z]{1,2}|\d{1,3})\b', val)
        if m:
            clean = m.group(1).upper()
            if clean in ["DE", "LE", "LA", "DU", "AU", "EN", "ET", "UN", "CE"]:
                return None
            if re.match(r'^[018]$', clean):
                clean = clean.replace("0", "O").replace("1", "I").replace("8", "B")
            return clean
        return None

    elif field_type == "feuille":
        # Format Geofoncier spÃ©cifique (ex: "000 AH 01")
        m_geof = re.search(r'\b\d{3}\s+[A-Z]{1,2}\s+(\d{1,3})\b', val.upper())
        if m_geof:
            return m_geof.group(1).lstrip("0") or "0"
            
        m = re.search(r'\b(\d{1,4}[A-Za-z]?|[A-Za-z]\d{1,3})\b', val)
        if m:
            return m.group(1).upper()
        return None

    elif field_type == "echelle":
        if any(u in val_norm for u in ["ca", "ha", "m2", " a ", "mÂ²"]):
            return None
        m = re.search(r'(1\s*[/:]\s*\d{3,5}|\d{3,5})', val)
        if m:
            return m.group(1).replace(" ", "")
        return None

    elif field_type in ("n_ordre", "n_dossier"):
        if any(u in val_norm for u in [" ha", " a ", " ca", "m2", "nee", "demeurant", "epouse"]):
            return None
        m_recent = re.search(r'\b([A-Z]\d{2}[.\-]?\d{2,5})\b|\b(\d{4}[.\-]\d{2,5})\b', val.upper())
        if m_recent:
            return (m_recent.group(1) or m_recent.group(2)).replace(" ", "")
        m_da = re.search(r'(\b\d{1,5}\s*[.\-]?\s*[A-Z]\b|\b\d{1,5}\b)', val.upper())
        if m_da:
            cleaned = re.sub(r'\s+', ' ', m_da.group(1)).strip()
            if re.match(r'^(19|20)\d{2}$', cleaned):
                return None
            return cleaned
        clean = re.sub(r"[^A-Za-z0-9\-\/\.]", "", val)
        if 2 <= len(clean) <= 20 and re.search(r'\d', clean):
            return clean
        return None


    elif field_type == "date":
        # Normaliser annÃ©es Ã  2 chiffres (14/10/97 -> 14/10/1997)
        # IMPORTANT : ne pas toucher les dates ISO (YYYY-MM-DD) qui ont dÃ©jÃ  4 chiffres
        def _norm_year_date(s: str) -> str:
            # Si la chaÃ®ne contient dÃ©jÃ  une annÃ©e Ã  4 chiffres, ne rien faire
            if re.search(r'\b(19|20)\d{2}\b', s):
                return s.strip()
            def _exp(m_y):
                sep, yy = m_y.group(1), int(m_y.group(2))
                return sep + str(2000 + yy if yy <= 30 else 1900 + yy)
            return re.sub(r'([/\.\-])(\d{2})$', _exp, s.strip())
        m = re.search(
            r'(\d{1,2}\s*[/\-\.]\s*\d{1,2}\s*[/\-\.]\s*\d{2,4}'
            r'|\d{1,2}\s+(?:janvier|f[eÃ©]vrier|mars|avril|mai|juin|juillet|ao[uÃ»]t|septembre|octobre|novembre|d[eÃ©]cembre)\s+\d{4})',
            val_norm
        )
        if m:
            return _norm_year_date(val.strip())
        return None

    elif field_type == "commune":
        # TolÃ©rer si l'OCR a fusionnÃ© "COMMUNE DE SAINT SAUVEUR" en une seule phrase
        # GÃ¨re aussi les fautes d'OCR (COMMUIE, COMMUNIE)
        m_commune = re.match(r'(?i)^commu[ni]e?\s*(?:de\s*)?(?::|\-)?\s*(.+)$', val)
        if m_commune:
            val_clean = m_commune.group(1).strip()
            if len(val_clean) >= 3:
                return val_clean
        # Supprimer les codes entre parenthÃ¨ses comme "(289)"
        val = re.sub(r'\s*\(\d+\)\s*', '', val).strip()
        if not val or val.lower() in ("commune", "commune de") or re.match(r'(?i)^commu[ni]e?$', val):
            return None
        COMMUNES_INTERDITES = {"commune", "territoire", "section", "echelle", "plan", "bornage", "feuille", "date", "d"}
        val_norm = val.lower()
        if val_norm in COMMUNES_INTERDITES:
            return None
        if re.match(r'^(?:07)?\d{3}$', val.strip()):
            return val.strip()
            
        if len(val) < 2 or len(val) > 50:
            return None
        if sum(c.isdigit() for c in val) > 2:
            return None
        if re.search(r'[_\{\}\[\]\\]', val):
            return None
        if re.match(r'(?i)^(les?|des?|de|du|un|une|ce|cette|il|elle|on|nous|vous|ils)$', val):
            return None
            
        # Nettoyage des points de suspension manuscrits "Saint Privat......."
        val = re.sub(r'[\.\s_\-]+$', '', val).strip()
            
        # Strict DB verification if DB is provided
        if commune_db:
            matched = _match_commune(val.strip(), commune_db)
            if matched != val.strip() or any(c["officiel"].upper() == val.strip().upper() for c in commune_db):
                return matched
            return val.strip() # Rejette catÃ©goriquement dÃ©sactivÃ©, on garde ce qui a Ã©tÃ© extrait
            
        return val.strip()

    elif field_type == "geometre":
        blacklisted_phrases = ["expert", "cabinet", "geometre", "gÃ©omÃ¨tre", "dresse", "dressÃ©", "par", "le", "soussigne", "soussignÃ©", "gÃ©omÃ¨tre-expert", "geometre-expert"]
        clean_words = [w for w in val_norm.split() if w not in blacklisted_phrases]
        if not clean_words:
            return None
            
        val_clean = " ".join(clean_words).upper()
        val_clean_norm = re.sub(r"[^A-Z0-9 ]", " ", val_clean).strip()
        
        # 1. TolÃ©rance OCR (Fuzzy Matching) avec la liste stricte
        try:
            from rapidfuzz import process as rfp, fuzz
            if len(val_clean_norm) >= 3:
                result = rfp.extractOne(val_clean_norm, GEOMETRES_CONNUS, scorer=fuzz.WRatio)
                if result and result[1] >= 75:
                    return result[0]
        except ImportError:
            pass
            
        # 2. Fallback exact (si rapidfuzz non disponible)
        for geo in GEOMETRES_CONNUS:
            if geo.lower() in val_norm:
                return geo
                
        # 3. LISTE STRICTE ASSOUPLIE : Si pas dans la liste mais valide, on le renvoie tel quel
        if len(val_clean_norm) >= 3:
             return val_clean_norm
        return None

    elif field_type in ("proprietaires_anciens", "proprietaires_nouveaux"):
        if len(val) < 3 or len(val) > 250:
            return None
        # Rejeter toujours les titres professionnels purs
        _TITRES = [
            "gÃ©omÃ¨tre expert", "geometre expert", "gÃ©omÃ¨tre-expert", "geometre-expert",
            "gÃ©omÃ¨tre", "notaire", "avocat", "huissier", "maire",
            "expert foncier", "expert-foncier", "technicien", "inspecteur",
        ]
        if any(val.lower().strip() == t for t in _TITRES):
            return None
        # Rejeter les phrases descriptives sans marqueur de civilitÃ©
        _MOTS_PHRASES = [
            "parcelles", "cadastr", "domaine", "territoire",
            "appartenant", "sis Ã ", "situÃ©", "lieudit", "lieu-dit",
        ]
        a_civilite = bool(re.search(r'(?i)\b(M\.|Mme\.?|Monsieur|Madame|Mlle\.?|M\s+et\s+Mme)', val))
        if not a_civilite and any(m in val.lower() for m in _MOTS_PHRASES):
            # ASSOUPLISSEMENT: On accepte quand mÃªme la valeur
            pass
        # Tronquer apres les separateurs metier
        val = re.sub(
            r'[;,]\s*(?:propri[eÃ©]taires?|actuel|sortant|entrant|des parcelles|'
            r'de la parcelle|ci[- ]apr[eÃ¨]s|lesquels|qui|dont|lequel|laquelle).*',
            '', val, flags=re.IGNORECASE
        ).strip()
        val = val.rstrip(',; ').strip()
        if len(val) < 3:
            return None
        return val


    elif field_type == "signataires":
        # Ne pas capturer des mots-clÃƒÂ©s du document, des noms de villes ou des labels
        MOTS_INTERDITS_SIGNATAIRE = {
            "vals les bains", "vallon pont", "vallon pont d'arc", "guilherand", "guilherand granges",
            "proces verbal", "procÃƒÂ¨s-verbal", "document d'arpentage", "bornage",
            "section", "commune", "feuille", "echelle", "plan", "cadastre",
            "geometres-experts", "experts", "geometre expert", "geometre",
            "document de conservation", "conservation", "service de conservation",
        }
        if val_norm.strip().lower() in MOTS_INTERDITS_SIGNATAIRE:
            return None
        # Rejeter les phrases (contenant des mots de liaison = ce n'est pas un nom)
        # Rejeter le bruit OCR : accolades, slashs, trop court, trop peu de lettres
        if re.search(r'[{}|]', val) or len(val.strip()) < 4:
            return None
        nb_alpha = sum(1 for c in val if c.isalpha())
        if len(val) > 0 and nb_alpha / len(val) < 0.5:
            return None
        if re.search(r'(?i)\b(bornee?|ce jour|conduit|fait|objet|demande|limite|alignement)\b', val):
            return None
        if len(val) < 3:
            return None
        return val

    # Champs libres (indication, feuille...)
    if len(val) > 150:
        return None
    return val

def _is_valid_signataire(name: str) -> bool:
    """VÃ©rifie qu'un nom est vraiment un nom de personne et pas une phrase du document."""
    name_up = re.sub(r"[^A-Z\s]", " ", name.upper()).strip()
    name_up = re.sub(r"\s+", " ", name_up)
    # Rejet si le nom normalisÃ© est dans la blacklist
    if name_up in _SIGNATAIRE_BLACKLIST:
        return False
    # ASSOUPLISSEMENT: On ne rejette plus les sous-chaÃ®nes (pour Ã©viter de jeter "COMMUNE" dans un nom lÃ©gitime)
    
    # Rejet si aucune majuscule ou trop de caractÃ¨res spÃ©ciaux
    if not re.search(r'[A-Z]', name):
        return False
    # Rejet si ressemble Ã  une phrase (>4 mots â†’ titre de section)
    if len(name.split()) > 4 and not re.search(r'(?i)(M\.|Mme|Monsieur|Madame|Mlle)', name):
        return False
    # Rejet si commence par un chiffre
    if re.match(r'^\d', name.strip()):
        return False
    return True


def _fuzzy_nc(t: str) -> str:
    """Normalisation pour la comparaison fuzzy de communes."""
    nfkd = unicodedata.normalize("NFKD", str(t))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9 ]", " ", s.upper()).strip()


def _score_commune(val: str, commune_db, seuil_exact=78) -> tuple:
    """Retourne (valeur_officielle, score, trouvÃ©_en_base).
    Essaie d'abord une correspondance exacte, puis fuzzy."""
    if not commune_db:
        return val, 0, False
    noms_officiel = {e["officiel"].lower(): e["officiel"] for e in commune_db}
    if val.lower() in noms_officiel:
        return noms_officiel[val.lower()], 100, True
    try:
        from rapidfuzz import process as rfp, fuzz as rfz
        noms_keys = list(noms_officiel.keys())
        best = rfp.extractOne(_fuzzy_nc(val), [_fuzzy_nc(n) for n in noms_keys], scorer=rfz.token_set_ratio)
        if best and best[1] >= seuil_exact:
            return noms_officiel[noms_keys[best[2]]], best[1], True
        return val, best[1] if best else 0, False
    except ImportError:
        return val, 0, False


# â”€â”€â”€ VÃ©rification Autonome (Passe 2 + 3) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _verify_and_revise(champs, all_ocr_page, img_shape, full_text, commune_db=None):
    """
    SystÃ¨me de vÃ©rification en 3 passes :

    PASSE A â€” Corroboration + Format :
      Attribue un score de confiance initial basÃ© sur la frÃ©quence
      d'apparition dans le document et la cohÃ©rence du format.

    PASSE B â€” ItÃ©ration (jusqu'Ã  3 tentatives) :
      Pour chaque champ INCERTAIN ou SUSPECT, relance une recherche
      alternative avec une stratÃ©gie diffÃ©rente. S'arrÃªte dÃ¨s qu'on
      obtient une valeur satisfaisante (â‰¥ 0.65).

    PASSE C â€” CohÃ©rence inter-champs :
      Valide les champs les uns par rapport aux autres.
      Ex: si la commune est trouvÃ©e en bas du doc par "Fait Ã  X le ...",
      elle confirme ou corrige la commune trouvÃ©e en haut.
    """
    verified = {}
    full_low = full_text.lower()
    h, w = img_shape

    print(f"\n  [Verify] â•â• Passe A : Corroboration + Format ({len(champs)} champs) â•â•")

    for field, data in champs.items():
        if not isinstance(data, dict):
            verified[field] = data
            continue
        val = str(data.get("valeur", ""))
        if not val or val.lower() in ("", "none", "nan"):
            verified[field] = data
            continue

        # Filtrage des signataires dÃ¨s la Passe A
        if field == "signataires":
            raw_list = data.get("valeur", [])
            if isinstance(raw_list, list):
                filtered = [n for n in raw_list if _is_valid_signataire(n)]
                if len(filtered) < len(raw_list):
                    print(f"    [A][signataires] FiltrÃ© {len(raw_list)-len(filtered)} entrÃ©es invalides")
                verified[field] = {**data, "valeur": filtered,
                                   "confidence": 0.75, "verification_status": "OK",
                                   "verification_notes": [f"{len(filtered)} signataires valides"]}
                continue

        confidence = 0.70
        notes = []
        revised_val = val

        # Corroboration : combien de fois la valeur apparaÃ®t dans le doc ?
        if len(val) > 3:
            count = full_low.count(val.lower())
            if count > 1:
                bonus = min(0.22, 0.08 * (count - 1))
                confidence += bonus
                notes.append(f"x{count} dans le doc")

        # Validation de format par type de champ
        if field == "commune":
            off_val, score, found = _score_commune(val, commune_db)
            if found:
                confidence += 0.22
                notes.append(f"Base OK ({score}%)")
                if off_val.lower() != val.lower():
                    notes.append(f"Corrige -> '{off_val}'")
                    revised_val = off_val
            else:
                confidence -= 0.30
                notes.append(f"Absent de la base ({score}%)")
            if sum(c.isdigit() for c in val) > 2: confidence -= 0.40; notes.append("Chiffres suspects")
            if len(val) < 3 or len(val) > 60: confidence -= 0.30; notes.append("Longueur anormale")

        elif field in ("n_ordre", "n_dossier"):
            if re.match(r'^[A-Z]\d{2}[\.\-]?\d{2,5}$', val.upper()):
                confidence += 0.25; notes.append("Format moderne OK")
            elif re.match(r'^\d{1,5}[A-Z]$', val.upper()):
                confidence += 0.20; notes.append("Format DA classique OK")
            elif not re.search(r'\d', val):
                confidence -= 0.50; notes.append("Sans chiffre -> suspect")

        elif field == "section":
            if re.match(r'^[A-Z]{1,2}$', val.upper()):
                confidence += 0.25; notes.append("1-2 lettres OK")
            else:
                confidence -= 0.30; notes.append("Format douteux")

        elif field == "feuille":
            if re.match(r'^\d{1,4}[A-Z]?$', val.upper()):
                confidence += 0.25; notes.append("Format OK")
            else:
                confidence -= 0.25; notes.append("Format douteux")

        elif field == "date":
            if re.match(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', val):
                confidence += 0.25; notes.append("JJ/MM/AAAA OK")
                # VÃ©rifier plausibilitÃ© : entre 1900 et 2030
                m_yr = re.search(r'(\d{4})', val)
                if m_yr:
                    yr = int(m_yr.group(1))
                    if not (1900 <= yr <= 2030):
                        confidence -= 0.30; notes.append(f"AnnÃ©e {yr} hors plage")
            elif re.search(r'\d{4}', val):
                confidence += 0.08; notes.append("AnnÃ©e prÃ©sente")
            else:
                confidence -= 0.40; notes.append("Format invalide")

        elif field == "echelle":
            ce = val.replace(" ", "")
            if re.match(r'^1[/:]?\d{3,5}$', ce):
                confidence += 0.25; notes.append("1/X OK")
            elif re.match(r'^\d{3,5}$', ce):
                confidence += 0.08; notes.append("Valeur seule")
            else:
                confidence -= 0.35; notes.append("Format douteux")

        elif field == "geometre":
            # Nettoyage via la liste noire avant Ã©valuation
            clean_geo = _validate_field("geometre", val)
            if clean_geo:
                revised_val = clean_geo
                val = clean_geo
                
                mots = val.strip().split()
                # Blacklist gÃ©omÃ¨tre
                if val.upper().strip() in {"EXPERTS BP", "EXPERTS", "EXPERT", "GEOMETRES"}:
                    confidence -= 0.60; notes.append("Nom gÃ©nÃ©rique -> rejet")
                elif re.match(r'^\d', val.strip()):
                    confidence -= 0.60; notes.append("Commence par chiffre")
                elif len(val) < 3:
                    confidence -= 0.50; notes.append("Trop court")
                elif len(mots) >= 2:
                    confidence += 0.15; notes.append("Multi-mots OK")
            else:
                confidence -= 1.0; notes.append("RejetÃ© par validate_field")
                revised_val = ""
                val = ""

        confidence = round(max(0.0, min(1.0, confidence)), 2)
        status = "OK" if confidence >= 0.65 else ("INCERTAIN" if confidence >= 0.40 else "SUSPECT")
        print(f"    [A][{field}] '{revised_val}' -> {status} ({confidence:.0%})" + (f" | {', '.join(notes)}" if notes else ""))

        verified[field] = {
            **data, "valeur": revised_val,
            "confidence": confidence, "verification_notes": notes,
            "verification_status": status,
        }

    # â•â• PASSE B : ItÃ©ration sur les champs incertains (jusqu'Ã  3 tentatives) â•â•
    uncertain_fields = [f for f, d in verified.items()
                        if isinstance(d, dict) and d.get("verification_status") in ("INCERTAIN", "SUSPECT")
                        and f not in ("parcelles", "signataires")]
    if uncertain_fields:
        print(f"\n  [Verify] â•â• Passe B : Relance itÃ©rative ({len(uncertain_fields)} champs suspects) â•â•")
        STRATEGIES = [
            # StratÃ©gie 1 : contextuel sur toute la page
            lambda f, ocr: _find_field_contextual(f, ocr, (h, w)),
            # StratÃ©gie 2 : keywords sur toute la page
            lambda f, ocr: _find_field(f, ocr, KEYWORDS.get(f, []), (h, w)),
            # StratÃ©gie 3 : GLiNER si disponible
            lambda f, ocr: _semantic_extract(f, ocr, (h, w)) if _GLINER_AVAILABLE else None,
        ]
        for field in uncertain_fields:
            current = verified[field]
            old_val = current["valeur"]
            print(f"    [B] '{field}' ({current['confidence']:.0%}) -> itÃ©rations...")
            for attempt, strategy in enumerate(STRATEGIES, 1):
                try:
                    res = strategy(field, all_ocr_page)
                except Exception:
                    res = None
                if res:
                    alt_raw = res.get("valeur", "")
                    alt_val = _validate_field(field, alt_raw) if alt_raw else None
                    if alt_val and alt_val.lower() != old_val.lower():
                        # VÃ©rifier la qualitÃ© de la nouvelle valeur
                        new_conf = current["confidence"]
                        if field == "commune":
                            _, sc, found = _score_commune(alt_val, commune_db)
                            if found: new_conf = max(0.75, new_conf + 0.20)
                        elif field in ("n_ordre", "n_dossier"):
                            if re.match(r'^[A-Z]\d{2}[\.\-]?\d{2,5}$', alt_val.upper()): new_conf = max(0.85, new_conf)
                        else:
                            new_conf = max(new_conf + 0.15, 0.60)
                        verified[field] = {
                            **current, "valeur": alt_val,
                            "confidence": round(new_conf, 2),
                            "verification_status": "OK" if new_conf >= 0.65 else "INCERTAIN",
                            "verification_notes": current["verification_notes"] + [f"B{attempt}: RÃ©visÃ© '{old_val}' -> '{alt_val}'"],
                        }
                        print(f"      [B{attempt}] RÃ©visÃ© : '{old_val}' -> '{alt_val}' ({new_conf:.0%})")
                        if new_conf >= 0.65:
                            break

    # â•â• PASSE C : CohÃ©rence inter-champs â•â•
    print(f"\n  [Verify] â•â• Passe C : CohÃ©rence inter-champs â•â•")

    # C1. La commune extraite du bas de page ("Fait Ã  X le ...") doit correspondre
    m_fait = re.search(
        r'(?:fait|redige|a)\s+([A-Z][A-Za-z\xc0-\xff\s\-]{2,30})[,\s]+le\s+'
        r'(\d{1,2}[\s/\-\.]\d{1,2}[\s/\-\.]\d{2,4})',
        full_text, re.IGNORECASE
    )
    if m_fait:
        commune_bottom = m_fait.group(1).strip()
        date_bottom = m_fait.group(2).strip()
        off_c, sc_c, found_c = _score_commune(commune_bottom, commune_db)
        if found_c and sc_c >= 80:
            current_comm = verified.get("commune", {})
            current_val = current_comm.get("valeur", "") if isinstance(current_comm, dict) else ""
            if not current_val or current_comm.get("confidence", 0) < 0.70:
                print(f"    [C1] Commune trouvÃ©e via 'Fait Ã ' : '{off_c}' ({sc_c}%)")
                verified["commune"] = {
                    **(current_comm if isinstance(current_comm, dict) else {}),
                    "valeur": off_c, "confidence": 0.82,
                    "verification_status": "OK",
                    "verification_notes": [f"TrouvÃ©e dans 'Fait Ã  {commune_bottom} le {date_bottom}'"],
                    "zone": [0.0, 0.60, 0.80, 0.95],
                }
        # C1b. Confirmer/complÃ©ter la date
        if not verified.get("date") or verified.get("date", {}).get("confidence", 0) < 0.65:
            print(f"    [C1b] Date trouvÃ©e via 'Fait Ã ' : '{date_bottom}'")
            verified["date"] = {
                "valeur": date_bottom, "confidence": 0.82,
                "verification_status": "OK",
                "verification_notes": [f"Extraite de 'Fait Ã  ... le {date_bottom}'"],
                "zone": [0.0, 0.60, 0.80, 0.95],
            }

    # C2. Un n_ordre/n_dossier incertain doit avoir au moins un chiffre
    for fn in ("n_ordre", "n_dossier"):
        fd = verified.get(fn)
        if isinstance(fd, dict) and fd.get("verification_status") != "OK":
            if not re.search(r'\d', str(fd.get("valeur", ""))):
                print(f"    [C2] '{fn}' sans chiffre -> supprimÃ©")
                verified[fn] = {**fd, "valeur": "", "confidence": 0.0, "verification_status": "SUSPECT"}

    # C2b. Si date toujours vide aprÃ¨s passes A+B, forcer un fallback regex brut sur le full_text
    _date_val = str(verified.get("date", {}).get("valeur", "") if isinstance(verified.get("date"), dict) else "")
    if not _date_val or _date_val.lower() in ("", "none", "nan"):
        # Chercher toute date en bas de page (format JJ/MM/AAAA) dans le texte entier
        _date_candidates = re.findall(
            r'\b(\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{4})\b', full_text
        )
        for _dc in _date_candidates:
            # VÃ©rifier que l'annÃ©e est plausible
            _yr = re.search(r'(\d{4})', _dc)
            if _yr and 1950 <= int(_yr.group(1)) <= 2030:
                _dc_clean = re.sub(r'[\s]', '/', _dc).strip()
                print(f"    [C2b] Date retrouvÃ©e par fallback brut : '{_dc_clean}'")
                verified["date"] = {
                    "valeur": _dc_clean, "confidence": 0.60,
                    "verification_status": "INCERTAIN",
                    "verification_notes": ["Fallback: date brute dans le texte"],
                    "zone": [0.0, 0.55, 1.0, 1.0],
                }
                break

    # C3. GÃ©omÃ¨tre : chercher dans tout le texte si toujours suspect
    geo = verified.get("geometre", {})
    if isinstance(geo, dict) and geo.get("verification_status") != "OK":
        # Regex stricte : cherche "dressÃ© par" ou "gÃ©omÃ¨tre expert :" suivi d'un vrai NOM
        # Le nom doit commencer par une majuscule et ne PAS Ãªtre une phrase
        m_geo2 = re.search(
            r'(?:dress[eÃ©]\s+par|g[eÃ©]om[eÃ¨]tre[s\-]?\s*expert\s*[:\-,]|le\s+soussign[eÃ©]\s+g[eÃ©]om)'
            r'[\s,:\-]*(?:M\.?\s*|Monsieur\s*|Mme\s*)?'
            r'([A-ZÃ€-Ã][a-zÃ -Ã¿]{2,}(?:\s+[A-ZÃ€-Ã][A-Za-zÃ -Ã¿]{2,}){0,3})',
            full_text, re.IGNORECASE
        )
        if m_geo2:
            g_val = m_geo2.group(1).strip().rstrip(".,;:")
            g_val = re.sub(r'\s+', ' ', g_val)
            has_liaison = re.search(r'(?i)\b(et|de|le|la|du|un|une|par|sur|sans|avec|soussign|plan|accept|expert\b)', g_val)
            if (
                len(g_val) >= 5
                and len(g_val.split()) <= 4
                and not re.match(r'^\d', g_val)
                and not has_liaison
                and g_val.upper() not in _SIGNATAIRE_BLACKLIST
            ):
                print(f"    [C3] GÃ©omÃ¨tre retrouvÃ© via passe C : '{g_val}'")
                verified["geometre"] = {
                    **(geo if isinstance(geo, dict) else {}),
                    "valeur": g_val, "confidence": 0.75,
                    "verification_status": "OK",
                    "verification_notes": ["TrouvÃ© via Passe C (recherche Ã©largie)"],
                }

    print(f"  [Verify] â•â• VÃ©rification terminÃ©e â•â•\n")
    return verified





# Ã¢â€â‚¬Ã¢â€â‚¬ DÃƒÂ©tection du type depuis le nom de fichier Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
def classify_plan(filepath: str) -> str:
    # 1. Lecture rapide du texte PDF pour prioriser les plans modernes (DGFIP)
    try:
        doc = fitz.open(filepath)
        txt = doc[0].get_text("text").upper()
        doc.close()
        
        # Si c'est un document moderne DGFIP, on force "MODERN_DGFIP" pour qu'il soit routÃ© vers process_modern_plan
        if "DIRECTION GÃ‰NÃ‰RALE DES FINANCES PUBLIQUES" in txt or "DIRECTION GENERALE DES FINANCES PUBLIQUES" in txt:
            return "MODERN_DGFIP"
            
        # D'ARPENTAGE dÃ©signe un Document d'Arpentage (ancien nom du DMPC)
        if "DMPC" in txt or "DOCUMENT MODIFICATIF" in txt or "D'ARPENTAGE" in txt:
            return "DMPC"
        if "PROCÃˆS-VERBAL" in txt or "PROCES-VERBAL" in txt or "PROCÃ‰S-VERBAL" in txt:
            return "PVa"
        if "LOTISSEMENT" in txt or "DIVISION" in txt:
            return "PLa"
    except Exception:
        pass

    # 2. DÃƒÂ©tection depuis le nom de fichier
    name = os.path.basename(filepath).lower()
    if re.search(r'pva|1pva|pvb|pvc|proc[eÃ¨]s.?verbal', name):
        return "PVa"
    if re.search(r'2pla|1pla|pla|lotissement|division', name):
        return "PLa"
    if re.search(r'dmpc|geofoncier_dmpc', name):
        return "DMPC"
    # Format DA typique, ex: 289-891-A ou 296-1050-E
    if re.search(r'^\d{3}[_\-]\d{3,5}[_\-][A-Za-z]', name):
        return "DMPC"
        
    return "GENERIC"


def classify_plan_with_ocr(filepath: str, reader=None) -> str:
    """
    Classification hybride de Phase 1 : nom de fichier + OCR Page 1.
    Effectue une passe OCR complÃ¨te sur la premiÃ¨re page avant le routage.
    """
    base_type = classify_plan(filepath)
    if reader is None:
        return base_type
        
    try:
        import fitz
        import cv2
        import numpy as np
        
        doc = fitz.open(filepath)
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        doc.close()
        
        print(f"  [Routeur] OCR de classification sur la page 1 de {os.path.basename(filepath)}...")
        ocr_results = reader.readtext(img_bgr)
        
        refined_type = _refine_type_plan_from_ocr(base_type, ocr_results)
        return refined_type
    except Exception as e:
        print(f"  [Routeur] Erreur OCR de classification: {e}")
        return base_type


def is_plan_document(filepath: str) -> bool:
    return True   # Tous les fichiers dans inputs/ sont des plans


# Ã¢â€â‚¬Ã¢â€â‚¬ Extraction OCR dans une zone fractionnelle Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
def _ocr_in_zone(reader, img: np.ndarray, zone: List[float]) -> List[Tuple[Any, str, float]]:
    """OCR EasyOCR sur une sous-rÃƒÂ©gion de l'image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(zone[0]*w), int(zone[1]*h), int(zone[2]*w), int(zone[3]*h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return []
    crop = img[y0:y1, x0:x1]
    results = reader.readtext(crop)
    # Remettre les bbox en coordonnÃƒÂ©es absolues
    shifted = []
    for (bbox, text, prob) in results:
        abs_bbox = [[pt[0]+x0, pt[1]+y0] for pt in bbox]
        shifted.append((abs_bbox, text, prob))
    return shifted

# Ã¢â€â‚¬Ã¢â€â‚¬ Extraction SÃƒÂ©mantique (Intelligence Artificielle) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
GLINER_LABELS_MAP = {
    "commune": "nom de commune",
    "section": "section cadastrale",
    "n_ordre": "numÃƒÂ©ro de document d'arpentage DA",
    "n_dossier": "numÃƒÂ©ro de rÃƒÂ©fÃƒÂ©rence dossier",
    "date": "date",
    "echelle": "ÃƒÂ©chelle du plan",
    "geometre": "nom du gÃƒÂ©omÃƒÂ¨tre expert",
    "proprietaires_anciens": "ancien propriÃƒÂ©taire cÃƒÂ©dant",
    "proprietaires_nouveaux": "nouveau propriÃƒÂ©taire acquÃƒÂ©reur",
    "indication": "objet du document bornage"
}

def _semantic_extract(field_type: str, ocr_results: List[Tuple], img_shape: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """Utilise un VLM/NLP pour lire le texte et comprendre oÃƒÂ¹ est l'information."""
    model = get_gliner_model()
    if not model: return None
    
    items = []
    for (bbox, text, prob) in ocr_results:
        if prob < 0.20: continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        items.append({"text": text, "box": [min(xs), min(ys), max(xs), max(ys)]})
        
    if not items: return None
    
    full_text = " | ".join([it["text"] for it in items])
    label_to_find = GLINER_LABELS_MAP.get(field_type)
    if not label_to_find: return None
    
    # On demande ÃƒÂ  l'IA de trouver l'entitÃƒÂ©
    entities = model.predict_entities(full_text, [label_to_find], threshold=0.50)
    if not entities: return None
    
    # On prend la prÃƒÂ©diction la plus confiante
    best_ent = max(entities, key=lambda x: x["score"])
    val = best_ent["text"].strip()
    
    # On passe quand mÃƒÂªme par la validation mÃƒÂ©tier pour ÃƒÂªtre sÃƒÂ»r
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

def _norm(text: str) -> str:
    """Normalise un texte pour le matching de mots-clÃ©s."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(text))
    sans_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r"[^a-z0-9]", " ", sans_accents.lower()).strip()

# Ã¢â€â‚¬Ã¢â€â‚¬ Recherche d'un champ par mots-clÃƒÂ©s dans l'OCR d'une zone Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
def _find_field(
    field_type: str,
    ocr_results: List[Tuple],
    keywords: List[str],
    img_shape: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    """
    Parcourt les rÃƒÂ©sultats OCR pour trouver le label puis capture la valeur.
    GÃƒÂ¨re les cas avec sÃƒÂ©parateurs (:) et sans sÃƒÂ©parateurs (Section C).
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
            
            # 1. Tentative: Valeur sur la mÃƒÂªme ligne
            raw = item["text"]
            val_candidate = ""
            
            # Cas A: SÃƒÂ©parateur prÃƒÂ©sent (ex: "Section : C")
            parts = re.split(r'[:Ã¢â‚¬â€œ\-;,\.]', raw, maxsplit=1)
            if len(parts) > 1:
                # On vÃƒÂ©rifie si le mot clÃƒÂ© est avant le sÃƒÂ©parateur
                if kw_norm in _norm(parts[0]):
                    val_candidate = parts[1].strip()
            
            # Cas B: Pas de sÃƒÂ©parateur (ex: "Section C")
            if not val_candidate:
                # On retire le mot clÃƒÂ© du texte pour voir ce qu'il reste
                # On cherche la position du kw dans le texte original (insensible ÃƒÂ  la casse)
                match = re.search(re.escape(kw), raw, re.IGNORECASE)
                if match:
                    val_candidate = raw[match.end():].strip()
                    # Si le reste commence par un sÃƒÂ©parateur qu'on aurait ratÃƒÂ©
                    val_candidate = re.sub(r'^[:Ã¢â‚¬â€œ\-\s\.]+', '', val_candidate)

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
                if y_dist < -50 or y_dist > 150: break
                
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



# Ã¢â€â‚¬Ã¢â€â‚¬ Matching commune contre la base Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
def _match_commune(text: str, commune_db: Optional[List[Dict]], seuil: int = 80) -> str:
    if not text or not commune_db:
        return text
    try:
        from rapidfuzz import process as rfp, fuzz
    except ImportError:
        return text

    # Handle INSEE code (e.g., '07289' or '289')
    import re
    text_clean = text.strip()
    if re.match(r'^(?:07)?\d{3}$', text_clean):
        code_insee = text_clean if len(text_clean) == 5 else f"07{text_clean}"
        for c in commune_db:
            if str(c.get("code", "")) == code_insee:
                return c["officiel"]
        return text

    def norm_c(t):
        nfkd = unicodedata.normalize("NFKD", str(t))
        s = "".join(c for c in nfkd if not unicodedata.combining(c))
        s = re.sub(r"[-''`]", " ", s)
        return re.sub(r"[^A-Z0-9 ]", " ", s.upper()).strip()

    text_n = norm_c(text)
    # Minimum 4 chars pour eviter "VALS" -> "Saint-AndÃ©ol-de-Vals"
    if not text_n or len(text_n) < 4:
        return text
    noms = [norm_c(e["officiel"]) for e in commune_db]
    # token_set_ratio evite les faux positifs des mots courts isolÃ©s
    result = rfp.extractOne(text_n, noms, scorer=fuzz.token_set_ratio)
    if result and result[1] >= seuil:
        matched = commune_db[result[2]]["officiel"]
        # Exiger un score > 90 si le texte est trÃ¨s court
        if len(text_n) < 6 and result[1] < 90:
            return text
        return matched
    return text


def _clean_vlm_response(val: str) -> str:
    """Nettoyage commun des rÃ©ponses VLM : supprime les phrases parasites."""
    import re as _re
    PREFIXES = [
        r"la photo montre.*?(?:sont|est|:)\s*",
        r"l'image montre.*?(?:sont|est|:)\s*",
        r"this image (?:shows|contains|depicts).*?(?::,)\s*",
        r"il semble.*?:\s*",
        r"les nombres.*?(?:sont|est|:)\s*",
        r"le texte sur l'?image est", r"le texte dans l'?image est",
        r"le texte correspondant .*? est", r"il s'agit de",
        r"ce document", r"voici le texte", r"texte lu",
        r"le texte lu est", r"le texte est", r"texte:", r"valeur",
        r"je peux voir", r"je vois", r"i can see", r"the text (?:reads|says|is)",
        r"based on the image", r"in the image",
    ]
    pattern = r"(?i)^(?:.*?(?:(?:est|:|: |\s)?" + r"|".join(PREFIXES) + r")\s*:?\s*)+"
    val = _re.sub(pattern, "", val).strip()
    val = _re.sub(r"(?i)^(?:texte|le texte|la valeur)?\s*(?:dans|sur)?\s*(?:l'?image|ce document)?\s*(?:est|:)?\s*", "", val).strip()
    val = val.replace('"', '').replace("'", "").strip(" '.:,")
    return val


def _extract_with_vlm(img_bgr: np.ndarray, type_plan: str, validate_fn, crops_data: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extraction directe par Vision-Language Model (Ollama)."""
    import base64
    import cv2
    import json
    import re
    import subprocess
    import os
    
    res_dict = {}
    
    # â”€â”€ NOUVELLE STRATÃ‰GIE : VLM CIBLÃ‰ SUR CROPS (Phase 3) â”€â”€
    if crops_data and len(crops_data) > 0:
        print(f"  [VLM] Traitement ciblÃ© sur {len(crops_data)} crops (Ancrage)...")
        h_img, w_img = img_bgr.shape[:2]
        
        for field, crop_info in crops_data.items():
            z = crop_info["zone"]
            x0, y0 = int(z[0]*w_img), int(z[1]*h_img)
            x1, y1 = int(z[2]*w_img), int(z[3]*h_img)
            crop_img = img_bgr[y0:y1, x0:x1]
            
            if crop_img.size == 0: continue
            
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, buffer = cv2.imencode('.jpg', crop_img, encode_param)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if field == "geometre":
                prompt = f"Extrais UNIQUEMENT le nom de la personne ou du cabinet Ã©crit sur l'image. NE FAIS AUCUNE PHRASE. Ne dis pas 'L'image montre' ou 'Le texte est'. RÃ©ponds juste avec le nom brut. Si c'est illisible, rÃ©ponds 'vide'."
            elif field in ["parcelles", "proprietaires_anciens", "proprietaires_nouveaux"]:
                prompt = f"Extrais UNIQUEMENT les mots ou numÃ©ros utiles liÃ©s Ã  '{field}'. NE FAIS AUCUNE PHRASE DESCRIPTIVE. Ne dis pas 'La photo montre' ni 'Les nombres sont'. Donne la liste brute. Si cest illisible, rÃ©ponds 'vide'."
            else:
                prompt = f"Extrais UNIQUEMENT la valeur de '{field}' sur l'image. NE FAIS AUCUNE PHRASE. Ne dÃ©cris pas l'image. RÃ©ponds juste par le texte lu. Si illisible, rÃ©ponds 'vide'."
                
            payload = {
                "model": "llava",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 64, "seed": 42}
            }
            
            os.makedirs("outputs", exist_ok=True)
            payload_path = os.path.join(os.getcwd(), "outputs", f"vlm_payload_{field}.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            win_payload_path = payload_path.replace("/mnt/c/", "C:\\").replace("/", "\\")
            cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate", "-H", "Content-Type: application/json", "-d", f"@{win_payload_path}"]
            
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
                if res.returncode == 0 and res.stdout.strip():
                    try:
                        ollama_json = json.loads(res.stdout)
                        raw = ollama_json.get("response", "").strip()
                        val = _reparer_encodage(raw)
                        
                        if val and val.lower() not in ["none", "null", "", "inconnu", "vide"]:
                            if "dÃ©solÃ©" in val.lower() or "peux pas" in val.lower() or "sorry" in val.lower() or "illisible" in val.lower():
                                val = ""

                            if val:
                                # On retire les textes parasites de LLaVa
                                PREFIXES_VLM = [
                                    r"la photo montre.*?(?:sont|est|:)\s*",
                                    r"l'image montre.*?(?:sont|est|:)\s*",
                                    r"il semble.*?:\s*",
                                    r"les nombres.*?(?:sont|est|:)\s*",
                                    r"le texte sur l'?image est", r"le texte dans l'?image est", 
                                    r"texte dans l'?image est", r"texte sur l'?image est",
                                    r"le texte correspondant .*? est", r"il s'agit de", 
                                    r"ce document", r"voici le texte", r"texte lu", 
                                    r"le texte lu est", r"le texte est", r"texte:", r"valeur",
                                    r"je peux voir", r"je vois"
                                ]
                                pattern = r"(?i)^(?:.*?(?:(?:est|:|: | \s)?" + r"|".join(PREFIXES_VLM) + r")\s*:?\s*)+"
                                val = re.sub(pattern, "", val).strip()
                                # Au cas oÃ¹ il resterait "texte dans limage est"
                                val = re.sub(r"(?i)^(?:texte|le texte|la valeur)?\s*(?:dans|sur)?\s*(?:l'?image|ce document)?\s*(?:est|:)?\s*", "", val).strip()
                                val = val.replace('"', '').replace("'", "").strip(' \'.:,')
                                
                                # Nettoyage si on a encore une phrase longue descriptive
                                if "la photo montre" in val.lower() or "l'image montre" in val.lower():
                                    val = ""
                                
                                final_val = validate_fn(field, val) if validate_fn else val
                                # Correction du bug: ne pas fallback sur 'val' si validate_fn retourne None (ce qui signifie invalide)
                                
                                if final_val:
                                    res_dict[field] = {
                                        "valeur": final_val,
                                        "zone": z,
                                        "brut": crop_info.get("brut", "") + " -> " + val,
                                        "methode": "vlm_crop",
                                        "confidence": 0.90
                                    }
                                    print(f"    [VLM Crop] {field} -> '{final_val}'")
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                print(f"    [VLM Crop] Erreur sur {field}: {e}")
            finally:
                if os.path.exists(payload_path): os.remove(payload_path)
                
    return res_dict

def _extract_dmpc_specialized(
    ocr_results: List[Tuple],
    w: int,
    h: int,
    validate_fn,
    commune_db=None,
    pdf_path: str = "",
) -> Dict[str, Any]:
    """
    Parseur rigide specifique aux documents DMPC CERFA.

    Extraction de la commune en 3 niveaux de fallback :
      Niveau 1 : Detection du label 'commune' par correspondance fuzzy
                 (tolere les erreurs OCR : COMMUIE, COMMUNIE, COMMUNE DE ...)
                 Puis cherche la valeur en dessous / a droite du label.
      Niveau 2 : Scan direct de la base de communes dans le quadrant
                 haut-gauche (x < 55%, y < 30%) â€” aucun label requis.
      Niveau 3 : Extraction du code commune depuis le nom de fichier
                 (ex: geofoncier_dmpc_07116_000_262 -> code=07116).
    """
    res = {}
    
    # â”€â”€ RÃ‰ACTIVATION DU NOM DE FICHIER POUR LES DOCUMENTS ILLISIBLES â”€â”€
    # Format exact : geofoncier_dmpc_07289_000_677.pdf -> (Insee: 07289, Section: 000, DA: 677)
    if pdf_path:
        filename = os.path.basename(pdf_path).lower()
        m_geof = re.match(r'(?:geofoncier_dmpc_)?(\d{5})_([0-9a-z]{1,3})_(\d{1,5})', filename)
        if m_geof:
            insee_code, section, n_ordre = m_geof.groups()
            
            if commune_db:
                for c in commune_db:
                    if str(c.get("code", "")) == insee_code:
                        res["commune"] = {
                            "valeur": c["officiel"], "zone": [0.0, 0.0, 0.3, 0.1],
                            "brut": f"[filename:{filename}]", "methode": "dmpc_filename_insee", "confidence": 1.0
                        }
                        break
            
            res["section"] = {
                "valeur": section.upper().lstrip('0') or "0", "zone": [0.0, 0.0, 0.3, 0.1],
                "brut": f"[filename:{filename}]", "methode": "dmpc_filename_sec", "confidence": 1.0
            }
            
            res["n_ordre"] = {
                "valeur": n_ordre, "zone": [0.0, 0.0, 0.3, 0.1],
                "brut": f"[filename:{filename}]", "methode": "dmpc_filename_da", "confidence": 1.0
            }
    blocks = [
        {
            "bbox": b, "text": t, "prob": p,
            "x0": min(pt[0] for pt in b), "y0": min(pt[1] for pt in b),
            "x1": max(pt[0] for pt in b), "y1": max(pt[1] for pt in b),
            "cx": sum(pt[0] for pt in b) / 4, "cy": sum(pt[1] for pt in b) / 4,
        }
        for b, t, p in ocr_results if p > 0.1 and t.strip()
    ]

    # â”€â”€ Helpers internes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _levenshtein(a: str, b: str) -> int:
        """Distance de Levenshtein simple entre deux chaines courtes."""
        if a == b:
            return 0
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, lb + 1):
                temp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
        return dp[lb]

    def _is_commune_label(text: str) -> bool:
        """
        Retourne True si le texte ressemble au label 'commune' sur un DMPC.
        Tolerant aux fautes OCR : COMMUIE, COMMUNIE, C0MMUNE, COMMUNE DE, etc.
        """
        t = re.sub(r"[^A-Za-z]", "", text).upper()
        # Correspondance exacte ou partielle
        if "COMMUNE" in t or "COMMUN" in t:
            return True
        # Fuzzy sur les 7 premiers caracteres (robuste aux fautes de fin)
        prefix = t[:7] if len(t) >= 7 else t
        return _levenshtein(prefix, "COMMUNE") <= 2

    def _scan_commune_in_zone(zone_blocks, commune_db):
        """
        Cherche directement un nom de commune connu dans une liste de blocs OCR.
        Utilise rapidfuzz si disponible, sinon Levenshtein maison.
        Retourne (nom_officiel, bloc) ou (None, None).
        """
        if not commune_db:
            return None, None
        try:
            from rapidfuzz import process as rfp, fuzz as rfz
            noms_normalises = [e["normalise"] for e in commune_db]
            noms_officiels = [e["officiel"] for e in commune_db]

            blacklist = {"PLAN", "DOCUMENT", "MODIFICATIF", "PARCELLAIRE", "EXTRAIT", "ECHELLE", "SECTION", "FEUILLE", "DATE", "OBJET", "DOSSIER", "NUMERO", "NORD", "ARRET", "VENTE"}
            for b in zone_blocks:
                t = re.sub(r"[^A-Za-zÃ€-Ã¿\s\-]", " ", b["text"]).strip()
                if len(t) < 4:
                    continue
                t_norm = re.sub(r"[^A-Z0-9 ]", " ", t.upper()).strip()
                
                # Check blacklist pour Ã©viter que "PLAN" ne match la commune "PLAN"
                if any(kw in t_norm.split() for kw in blacklist):
                    continue

                # token_set_ratio est beaucoup plus fiable que WRatio pour ignorer les mots en trop
                best = rfp.extractOne(t_norm, noms_normalises, scorer=rfz.token_set_ratio)
                if best and best[1] >= 85:
                    idx = best[2]
                    print(f"    [DMPC Niveau2] Commune directe : '{b['text']}' -> '{noms_officiels[idx]}' ({best[1]}%)")
                    return noms_officiels[idx], b
        except ImportError:
            pass
        return None, None

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 1. Commune
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    commune_zone_blocks = [b for b in blocks if b["cx"] < w * 0.60 and b["cy"] < h * 0.30]

    # -- Niveau 2 : Detection du label (fuzzy) + valeur en dessous/droite ------
    if "commune" not in res:
        commune_label = next(
            (b for b in commune_zone_blocks if _is_commune_label(b["text"])), None
        )
        if commune_label:
            m_inline = re.split(r'[:\-]', commune_label["text"], maxsplit=1)
            val_inline = m_inline[1].strip() if len(m_inline) > 1 else re.sub(r'(?i)commune\s*(?:de)?\s*:?\s*', '', commune_label["text"]).strip()
            val_valid = validate_fn("commune", val_inline)
            
            # Validation stricte en base
            if val_valid and commune_db:
                matched = _match_commune(val_valid, commune_db)
                if matched != val_valid or any(c["officiel"].upper() == val_valid.upper() for c in commune_db):
                    val_valid = matched
                else:
                    val_valid = None

            if val_valid:
                res["commune"] = {
                    "valeur": val_valid, "zone": [commune_label["x0"]/w, commune_label["y0"]/h, commune_label["x1"]/w, commune_label["y1"]/h],
                    "brut": commune_label['text'], "methode": "dmpc_label_inline", "confidence": 0.95,
                }
            else:
                cands = []
                for b in blocks:
                    if b is commune_label: continue
                    if commune_label["cy"] - h * 0.02 < b["cy"] <= commune_label["cy"] + h * 0.25:
                        if abs(b["cx"] - commune_label["cx"]) < w * 0.25: cands.append(b)
                if not cands:
                    for b in blocks:
                        if b is commune_label: continue
                        if abs(b["cy"] - commune_label["cy"]) <= h * 0.05 and b["cx"] > commune_label["x1"]:
                            if b["cx"] - commune_label["x1"] < w * 0.35: cands.append(b)
                if cands:
                    cands.sort(key=lambda b: b["cy"] if b["cy"] > commune_label["cy"] + h * 0.02 else b["cx"])
                    for cand in cands:
                        val = validate_fn("commune", cand["text"])
                        if val:
                            if commune_db:
                                matched = _match_commune(val, commune_db)
                                if matched != val or any(c["officiel"].upper() == val.upper() for c in commune_db):
                                    val = matched
                                else:
                                    continue # Rejeter ce candidat qui ne matche pas la DB
                            res["commune"] = {
                                "valeur": val, "zone": [cand["x0"]/w, cand["y0"]/h, cand["x1"]/w, cand["y1"]/h],
                                "brut": f"{commune_label['text']} -> {cand['text']}", "methode": "dmpc_label_fuzzy", "confidence": 0.95,
                            }
                            break

    # -- Niveau 3 : Scan direct (si commune non trouvee par label) -------------
    if "commune" not in res and commune_db:
        val_dir, blk_dir = _scan_commune_in_zone(commune_zone_blocks, commune_db)
        if val_dir and blk_dir:
            res["commune"] = {
                "valeur": val_dir, "zone": [blk_dir["x0"]/w, blk_dir["y0"]/h, blk_dir["x1"]/w, blk_dir["y1"]/h],
                "brut": blk_dir["text"], "methode": "dmpc_direct_scan", "confidence": 0.80,
            }

    if "commune" not in res:
        print(f"    [DMPC] Commune introuvable apres les 3 niveaux.")

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 2. N_Ordre : Extraction
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # -- Niveau 1 : Nom de fichier (DÃ‰SACTIVÃ‰ Ã€ LA DEMANDE DE L'UTILISATEUR) --
    # if pdf_path:
    #     m_da = re.search(r'\b\d{3}[_\-]?(\d{3,5}[_\-]?[A-Z])\b', os.path.basename(pdf_path), re.IGNORECASE)
    #     if m_da:
    #         val_valid = validate_fn("n_ordre", m_da.group(1))
    #         if val_valid:
    #             res["n_ordre"] = {
    #                 "valeur": val_valid, "zone": [0.0, 0.0, 0.5, 0.2],
    #                 "brut": f"[filename:{os.path.basename(pdf_path)}]", "methode": "dmpc_filename_da", "confidence": 1.0,
    #             }

    # -- Niveau 2 : Quadrant haut-droit --
    if "n_ordre" not in res:
        ordre_label = next(
            (b for b in blocks if any(k in b["text"].lower() for k in ["ordre", "da n"])
             and b["cy"] < h * 0.45),
            None,
        )
        if ordre_label:
            m_inline = re.split(r'[:]', ordre_label["text"])
            val_inline = m_inline[-1].strip() if len(m_inline) > 1 else re.sub(r'(?i).*?(?:ordre|da\s*n.)\s*:?\s*', '', ordre_label["text"]).strip()
            val_valid = validate_fn("n_ordre", val_inline)
            if val_valid:
                res["n_ordre"] = {
                    "valeur": val_valid, "zone": [ordre_label["x0"]/w, ordre_label["y0"]/h, ordre_label["x1"]/w, ordre_label["y1"]/h],
                    "brut": ordre_label['text'], "methode": "dmpc_label_inline", "confidence": 0.98,
                }
            else:
                cands = []
                for b in blocks:
                    if b is ordre_label:
                        continue
                    if b["cy"] < ordre_label["cy"] - h * 0.10 or b["cy"] > ordre_label["cy"] + h * 0.20:
                        continue
                    if b["cx"] < ordre_label["cx"] - w * 0.20:
                        continue
                    dist = abs(b["cx"] - ordre_label["cx"]) + abs(b["cy"] - ordre_label["cy"]) * 3
                    cands.append((dist, b))
                if cands:
                    cands.sort(key=lambda x: x[0])
                    for i, (_, b) in enumerate(cands):
                        text = b["text"]
                        x1 = b["x1"]
                        # Fusion avec le bloc suivant s'il est trÃ¨s proche (ex: "677" et "J")
                        if i + 1 < len(cands):
                            _, next_b = cands[i+1]
                            # Le bloc suivant doit Ãªtre sur la mÃªme ligne et Ã  droite (tolÃ©rance de lÃ©ger chevauchement)
                            if abs(next_b["cy"] - b["cy"]) < h * 0.03 and next_b["cx"] > b["cx"] and next_b["x0"] - b["x1"] < w * 0.10:
                                text += " " + next_b["text"]
                                x1 = max(x1, next_b["x1"])
                                
                        clean_text = re.sub(r"[\s_]+$", "", text).strip()
                        val = validate_fn("n_ordre", clean_text)
                        if val:
                            res["n_ordre"] = {
                                "valeur": val,
                                "zone": [b["x0"] / w, b["y0"] / h, x1 / w, b["y1"] / h],
                                "brut": f"{ordre_label['text']} -> {text}",
                                "methode": "dmpc_grid",
                                "confidence": 0.98,
                            }
                            break

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 3. Geometre : Quadrant bas-droit
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    geo_label = next(
        (b for b in blocks if any(k in b["text"].lower() for k in ["dress", "cabinet", "soussign", "le geometre"])
         and b["cy"] > h * 0.5),
        None,
    )
    if geo_label:
        cands = []
        for b in blocks:
            if b is geo_label:
                continue
            if abs(b["cx"] - geo_label["cx"]) > w * 0.3:
                continue
            if b["cy"] < geo_label["cy"] - h * 0.10 or b["cy"] > geo_label["cy"] + h * 0.15:
                continue
            dist = abs(b["cx"] - geo_label["cx"]) + abs(b["cy"] - geo_label["cy"]) * 3
            cands.append((dist, b))
        if cands:
            cands.sort(key=lambda x: x[0])
            for _, b in cands:
                val = validate_fn("geometre", b["text"])
                if val:
                    res["geometre"] = {
                        "valeur": val,
                        "zone": [b["x0"] / w, b["y0"] / h, b["x1"] / w, b["y1"] / h],
                        "brut": f"{geo_label['text']} -> {b['text']}",
                        "methode": "dmpc_grid",
                        "confidence": 0.98,
                    }
                    break

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # 4. Section & Feuille : Extraction ciblÃ©e pour DMPC (Moderne & Ancien)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    for field, labels in [("section", ["section"]), ("feuille", ["feuille"])]:
        if field not in res:
            lbl_block = next(
                (b for b in blocks if any(k in b["text"].lower() for k in labels) and b["cy"] < h * 0.45),
                None
            )
            if lbl_block:
                m_inline = re.split(r'[:\-;\.,]', lbl_block["text"])
                val_inline = m_inline[-1].strip() if len(m_inline) > 1 else re.sub(fr'(?i){labels[0]}s?\s*(?:\(s\))?\s*[:;\.,]?\s*', '', lbl_block["text"]).strip()
                if field == "feuille" and len(val_inline) > 5 and val_inline.endswith(("01", "02", "03", "04", "05")):
                    val_inline = val_inline.split()[-1]
                val_valid = validate_fn(field, val_inline)
                if val_valid:
                    res[field] = {
                        "valeur": val_valid, "zone": [lbl_block["x0"]/w, lbl_block["y0"]/h, lbl_block["x1"]/w, lbl_block["y1"]/h],
                        "brut": lbl_block['text'], "methode": "dmpc_label_inline", "confidence": 0.95,
                    }
                else:
                    cands = []
                    for b in blocks:
                        if b is lbl_block: continue
                        if b["cy"] < lbl_block["cy"] - h * 0.1 or b["cy"] > lbl_block["cy"] + h * 0.25:
                            continue
                        if abs(b["cx"] - lbl_block["cx"]) > w * 0.5:
                            continue
                        dist = abs(b["cx"] - lbl_block["cx"]) + abs(b["cy"] - lbl_block["cy"]) * 3
                        cands.append((dist, b))
                    if cands:
                        cands.sort(key=lambda x: x[0])
                        for _, b in cands:
                            val = validate_fn(field, b["text"])
                            if val:
                                res[field] = {
                                    "valeur": val, "zone": [b["x0"]/w, b["y0"]/h, b["x1"]/w, b["y1"]/h],
                                    "brut": f"{lbl_block['text']} -> {b['text']}", "methode": "dmpc_grid", "confidence": 0.95,
                                }
                                break

    return res





def _refine_type_plan_from_ocr(initial_type: str, ocr_results: List[Tuple]) -> str:
    """
    Affine la classification initiale (basÃ©e sur le nom de fichier)
    en analysant la densitÃ© et le contenu du texte OCR de la premiÃ¨re page.
    Ceci permet de diffÃ©rencier dynamiquement les plans modernes (entiÃ¨rement imprimÃ©s)
    des formulaires anciens (hybrides) sans se fier uniquement au nom de fichier.
    """
    high_conf_blocks = [b for b in ocr_results if b[2] > 0.6]
    full_text = " ".join([b[1] for b in high_conf_blocks]).lower()
    
    # 0. Les extraits modernes (DGFIP) sont toujours des plans gÃ©nÃ©riques/modernes
    if initial_type == "MODERN_DGFIP":
        return "GENERIC"
    if "finances publiques" in full_text or "extrait du plan cadastral" in full_text:
        return "GENERIC"
        
    # 1. Signaux textuels forts explicites
    if "procÃ¨s-verbal" in full_text or "proces-verbal" in full_text or "proces verbal" in full_text:
        return "PVa"
    if "document modificatif" in full_text or "d.m.p.c" in full_text or "d'arpentage" in full_text or "d arpentage" in full_text:
        return "DMPC"
    if "lotissement" in full_text or "division" in full_text:
        return "PLa"
    if "croquis" in full_text or "conservation" in full_text:
        return "CROQUIS"  # Les anciens croquis sont traitÃ©s via la grille spatiale globale
        
    # 2. Analyse de densitÃ© pour diffÃ©rencier moderne vs formulaire ancien
    labels = ["commune", "section", "feuille", "dossier", "ordre", "echelle"]
    filled_labels = 0
    empty_labels = 0
    
    for (bbox, text, prob) in ocr_results:
        if prob < 0.2: continue
        t_lower = text.lower()
        for lbl in labels:
            if lbl in t_lower:
                parts = re.split(r'[:\-]', text)
                if len(parts) > 1 and len(parts[1].strip()) > 2:
                    filled_labels += 1
                elif len(text.lower().replace(lbl, "").strip()) > 2:
                    filled_labels += 1
                else:
                    empty_labels += 1
                    
    # S'il y a beaucoup de trous (labels trouvÃ©s seuls par l'OCR, la valeur manuscrite n'est pas lue)
    # c'est la signature d'un formulaire ancien type DMPC.
    if empty_labels > filled_labels and empty_labels >= 2:
        return "DMPC"
        
    # DensitÃ© relative (remplace seuil fixe 80 blocs)
    n_high = len(high_conf_blocks)
    n_all = max(len([b for b in ocr_results if b[2] > 0.2]), 1)
    ratio_hc = n_high / n_all

    if initial_type in ("GENERIC", "CROQUIS"):
        if filled_labels >= 2 and n_high >= 60:
            return "GENERIC"
        if n_high < 40 or (n_high < 70 and ratio_hc < 0.55):
            print(f"  [Classif] Document peu dense ({n_high} blocs HC, ratio={ratio_hc:.2f}) -> CROQUIS")
            return "CROQUIS"

    if filled_labels >= 2 and n_high >= 80:
        return "GENERIC"

    return initial_type

# Ã¢â€â‚¬Ã¢â€â‚¬ Pipeline principal Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

def _extract_with_vlm(img_bgr, type_plan: str, validate_fn, crops_data=None):
    """Extraction VLM ciblÃ©e sur des crops d'ancrage avec modÃ¨le adaptatif."""
    import base64, json, subprocess, os
    import cv2
    res_dict = {}
    if not crops_data:
        return res_dict
    model_name = _get_vlm_model(type_plan)
    print(f"  [VLM] Crops sur {len(crops_data)} champs (modele: {model_name})...")
    h_img, w_img = img_bgr.shape[:2]
    for field, crop_info in crops_data.items():
        z = crop_info["zone"]
        x0, y0 = int(z[0]*w_img), int(z[1]*h_img)
        x1, y1 = int(z[2]*w_img), int(z[3]*h_img)
        crop_img = img_bgr[y0:y1, x0:x1]
        if crop_img.size == 0:
            continue
        _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        prompt = _get_vlm_prompt(type_plan, field)
        timeout_sec = 90 if model_name == "llama3.2-vision" else 60
        payload = {"model": model_name, "prompt": prompt, "images": [img_base64],
                   "stream": False, "options": {"temperature": 0.0, "num_predict": 80, "seed": 42}}
        os.makedirs("outputs", exist_ok=True)
        payload_path = os.path.join(os.getcwd(), "outputs", f"vlm_payload_{field}.json")
        with open(payload_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp)
        cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate",
               "-H", "Content-Type: application/json", "-d", f"@{payload_path}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if res.returncode == 0 and res.stdout.strip():
                ojson = json.loads(res.stdout)
                raw = _reparer_encodage(ojson.get("response", "").strip())
                if raw and raw.lower() not in ["none", "null", "", "inconnu", "vide", "illisible"]:
                    if any(kw in raw.lower() for kw in ["dÃ©solÃ©", "sorry", "cannot", "i cannot"]):
                        raw = ""
                    if raw:
                        raw = _clean_vlm_response(raw)
                    if raw:
                        final_val = validate_fn(field, raw) if validate_fn else raw
                        if final_val:
                            res_dict[field] = {
                                "valeur": final_val, "zone": z,
                                "brut": crop_info.get("brut", "") + " -> " + raw,
                                "methode": f"vlm_crop_{model_name.split(':')[0]}",
                                "confidence": 0.92 if model_name == "llama3.2-vision" else 0.90,
                            }
                            print(f"    [VLM/{model_name.split(':')[0]}] {field} -> '{final_val}'")
        except Exception as e:
            print(f"    [VLM Crop] Erreur {field}: {e}")
        finally:
            if os.path.exists(payload_path):
                os.remove(payload_path)
    return res_dict

def _extract_with_vlm_full_page(img_bgr, fields_to_extract, commune_db=None, type_plan: str = "GENERIC"):
    """Extraction VLM pleine page avec modÃ¨le et prompt adaptatifs."""
    import base64, json, subprocess, os
    import cv2
    res_dict = {}
    model_name = _get_vlm_model(type_plan)
    print(f"  [VLM Pleine Page] {fields_to_extract} (modele: {model_name})")
    h, w = img_bgr.shape[:2]
    max_dim = 2000 if model_name == "llama3.2-vision" else 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_r = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    else:
        img_r = img_bgr
    _, buffer = cv2.imencode('.jpg', img_r, [int(cv2.IMWRITE_JPEG_QUALITY), 87])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    _fdesc = {
        "commune": "nom de la commune", "section": "lettre de section cadastrale",
        "feuille": "numÃ©ro de feuille", "n_ordre": "numÃ©ro d'ordre ou DA",
        "date": "date de dressÃ© (JJ/MM/AAAA)", "echelle": "Ã©chelle (1/XXXX)",
        "geometre": "nom du gÃ©omÃ¨tre-expert", "indication": "objet du document",
        "proprietaires_anciens": "propriÃ©taire cÃ©dant",
        "proprietaires_nouveaux": "nouveau propriÃ©taire",
    }
    _ctx = {
        "PVa": "C'est un procÃ¨s-verbal de bornage tapÃ© Ã  la machine.",
        "CROQUIS": "C'est un ancien plan cadastral (texte manuscrit ou tampon).",
        "DMPC": "C'est un formulaire DMPC avec cartouche structurÃ©.",
    }.get(type_plan, "C'est un document cadastral.")
    fields_desc = ", ".join(_fdesc.get(f, f) for f in fields_to_extract)
    prompt = (
        f"Tu es un expert gÃ©omÃ¨tre. {_ctx} "
        f"Extrais: {fields_desc}. "
        f"Ne devine pas â€” si illisible, mets 'vide'. "
        f"RÃ©ponds UNIQUEMENT en JSON avec les clÃ©s: {', '.join(fields_to_extract)}. "
        f"Valeurs courtes et factuelles."
    )
    timeout_sec = 180 if model_name == "llama3.2-vision" else 120
    payload = {"model": model_name, "prompt": prompt, "images": [img_base64],
               "stream": False, "format": "json",
               "options": {"temperature": 0.0, "num_predict": 512, "seed": 42}}
    os.makedirs("outputs", exist_ok=True)
    payload_path = os.path.join(os.getcwd(), "outputs", "vlm_payload_full_page.json")
    with open(payload_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate",
           "-H", "Content-Type: application/json", "-d", f"@{payload_path}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if res.returncode == 0 and res.stdout.strip():
            ojson = json.loads(res.stdout)
            raw = ojson.get("response", "").strip()
            try:
                data = json.loads(raw)
                for f, val in data.items():
                    if f not in fields_to_extract or not isinstance(val, str):
                        continue
                    if val.lower() in ["none", "null", "", "inconnu", "vide", "illisible"]:
                        continue
                    val = _reparer_encodage(val)
                    if any(kw in val.lower() for kw in ["dÃ©solÃ©", "illisible", "cannot"]):
                        continue
                    val = _clean_vlm_response(val)
                    if not val:
                        continue
                    final_val = _validate_field(f, val, commune_db)
                    if final_val:
                        conf = 0.78 if model_name == "llama3.2-vision" else 0.70
                        res_dict[f] = {
                            "valeur": final_val, "zone": [0.0, 0.0, 1.0, 1.0],
                            "brut": f"VLM FP ({model_name.split(':')[0]}): {val}",
                            "methode": f"vlm_fullpage_{model_name.split(':')[0]}",
                            "confidence": conf,
                        }
                        print(f"    [VLM FP/{model_name.split(':')[0]}] {f} -> '{final_val}'")
            except json.JSONDecodeError as e:
                print(f"    [VLM FullPage JSON Error] {e}")
    except Exception as e:
        print(f"    [VLM FullPage Error] ({model_name}): {e}")
    finally:
        if os.path.exists(payload_path):
            os.remove(payload_path)
    return res_dict



def _extract_pva_textuel(ocr_results, w: int, h: int, validate_fn) -> dict:
    """
    Extraction dÃ©diÃ©e pour les ProcÃ¨s-Verbaux de bornage (PVa).
    Analyse sÃ©quentielle du texte libre et des mots-clÃ©s typiques.
    """
    import re
    res = {}
    full_text = " ".join([b[1] for b in ocr_results])
    
    blocks = [{"text": b[1], "bbox": b[0], "prob": b[2], 
               "cy": sum(p[1] for p in b[0])/4.0, "cx": sum(p[0] for p in b[0])/4.0,
               "y0": min(p[1] for p in b[0]), "y1": max(p[1] for p in b[0]),
               "x0": min(p[0] for p in b[0]), "x1": max(p[0] for p in b[0])} 
              for b in ocr_results]

    # 1. Commune : "commune de [X]" ou "territoire de la commune de [X]"
    m_com = re.search(r'(?i)commune\s+(?:de\s+|d\'|d’)([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-]{2,30}?)(?:\s+section|\s*,|\s*\n|$)', full_text)
    if m_com:
        val = validate_fn("commune", m_com.group(1))
        if val:
            res["commune"] = {"valeur": val, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_com.group(0), "methode": "pva_regex_commune", "confidence": 0.95}

    # 2. Date et Lieu : "Fait Ã  [Lieu], le [Date]"
    # Le lieu Fait Ã  X correspond trÃ¨s souvent Ã  la commune ou Ã  une ville reconnue.
    m_fait = re.search(r'(?i)fait\s+[aà]\s+([A-Za-zÀ-ÿ\s\-]{2,30}?)\s*,\s*le\s+([0-9]{1,2}(?:er)?\s+[a-zéû]+(?:\s+[0-9]{4})?|[0-9]{1,2}\s*[/\-\.]\s*[0-9]{1,2}\s*[/\-\.]\s*[0-9]{2,4})', full_text)
    if m_fait:
        lieu = m_fait.group(1).strip()
        date_str = m_fait.group(2).strip()
        if "commune" not in res:
            val_lieu = validate_fn("commune", lieu)
            if val_lieu:
                res["commune"] = {"valeur": val_lieu, "zone": [0.0, 0.0, 1.0, 1.0], "brut": f"Fait Ã  {lieu}", "methode": "pva_regex_fait_a", "confidence": 0.90}
        val_date = validate_fn("date", date_str)
        if val_date:
            res["date"] = {"valeur": val_date, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_fait.group(0), "methode": "pva_regex_date", "confidence": 0.95}

    # Fallback date si pas de "Fait Ã "
    if "date" not in res:
        m_date = re.search(r'(?i)le\s+([0-9]{1,2}(?:er)?\s+[a-zÃ©Ã»]+\s+[0-9]{4}|[0-9]{1,2}\s*[/\-\.]\s*[0-9]{1,2}\s*[/\-\.]\s*[0-9]{2,4})', full_text)
        if m_date:
            val = validate_fn("date", m_date.group(1))
            if val:
                res["date"] = {"valeur": val, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_date.group(0), "methode": "pva_regex_date", "confidence": 0.85}

    # 3. Section
    m_sect = re.search(r'(?i)section\s+([A-Z]{1,2})', full_text)
    if m_sect:
        val = validate_fn("section", m_sect.group(1))
        if val:
            res["section"] = {"valeur": val, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_sect.group(0), "methode": "pva_regex_section", "confidence": 0.90}

    # 4. GÃ©omÃ¨tre : "Le GÃ©omÃ¨tre", "Le soussignÃ©", etc.
    cands_geo = []
    for b in blocks:
        if b["cy"] > h * 0.4:  # Scanner la moitiÃ© infÃ©rieure
            val = validate_fn("geometre", b["text"])
            if val:
                cands_geo.append((b, val))
    if cands_geo:
        # Prendre le dernier (le plus en bas)
        best = cands_geo[-1]
        res["geometre"] = {
            "valeur": best[1], 
            "zone": [best[0]["x0"]/w, best[0]["y0"]/h, best[0]["x1"]/w, best[0]["y1"]/h], 
            "brut": best[0]["text"], 
            "methode": "pva_scan_geometre", 
            "confidence": 0.85
        }

    return res

def process_plan(pdf_path: str, models=None, commune_db=None) -> dict:
    print(f"  [PlanClassifier] Traitement : {os.path.basename(pdf_path)}")
    reader = models[1] if models and len(models) > 1 else None
    if reader is None:
        print("  [PlanClassifier] Pas de reader OCR disponible.")
        return {"fichier": pdf_path, "type_plan": "GENERIC", "pages": [], "skipped": False}

    type_plan = classify_plan(pdf_path)
    zones_def = ZONES_PAR_TYPE.get(type_plan, ZONES_PAR_TYPE["GENERIC"])
    print(f"  [PlanClassifier] Type dÃƒÂ©tectÃƒÂ© : {type_plan}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [PlanClassifier] Erreur ouverture PDF : {e}")
        return {"fichier": pdf_path, "skipped": True, "raison": str(e)}

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    nb_pages = len(doc)
    print(f"  [PlanClassifier] Document : {nb_pages} page(s)")

    # Ã¢â€â‚¬Ã¢â€â‚¬ StratÃƒÂ©gie : rÃƒÂ´le de chaque page dans le document Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    # RÃƒÂ¨gles mÃƒÂ©tier : oÃƒÂ¹ se trouve chaque type d'information ?
    # entete     : cartouche (commune, section, date, ÃƒÂ©chelle...) Ã¢â€ â€™ Page 1
    # corps      : plan graphique (parcelles, propriÃƒÂ©taires)     Ã¢â€ â€™ Pages du milieu
    # signatures : signatures, gÃƒÂ©omÃƒÂ¨tre, DMPC bas de page        Ã¢â€ â€™ DerniÃƒÂ¨re page
    # all        : document mono-page, tout est au mÃƒÂªme endroit

    CHAMPS_PAR_ROLE: Dict[str, set] = {
        "entete":     {"commune", "n_ordre", "n_dossier", "section",
                       "feuille", "echelle", "indication"},
        "corps":      {"parcelles", "proprietaires_anciens", "proprietaires_nouveaux"},
        "signatures": {"geometre", "signataires", "date"},
        "all":        {"commune", "n_ordre", "n_dossier", "section", "feuille",
                       "date", "echelle", "indication", "parcelles",
                       "proprietaires_anciens", "proprietaires_nouveaux",
                       "geometre", "signataires"},
    }

    def _page_role(pnum: int, total: int) -> str:
        if total == 1:
            return "all"
        if pnum == 1:
            return "entete"
        if pnum == total:
            return "signatures"
        return "corps"

    # Ã¢â€â‚¬Ã¢â€â‚¬ RÃƒÂ©sultat consolidÃƒÂ© unique pour tout le document Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
    # Un seul dict de champs : le meilleur rÃƒÂ©sultat gagne, pas page par page.
    doc_champs: Dict[str, Any] = {}
    images_par_page = []  # Pour gÃƒÂ©nÃƒÂ©rer les .jpg annotÃƒÂ©s

    for pi, page in enumerate(doc):
        page_num = pi + 1
        role = _page_role(page_num, nb_pages)
        champs_attendus = CHAMPS_PAR_ROLE.get(role, set())
        
        # Pour les croquis, on force l'ajout de tous les champs qui pourraient Ãªtre dissÃ©minÃ©s
        if type_plan == "CROQUIS":
            champs_attendus.update({"date", "geometre", "indication", "commune", "echelle", "section", "n_ordre"})
        print(f"  [PlanClassifier] Page {page_num}/{nb_pages} Ã¢â‚¬â€ rÃƒÂ´le: [{role}]")

        # Rendu haute rÃƒÂ©solution
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        # Champs extraits sur cette page (avant merge dans doc_champs)
        champs: Dict[str, Any] = {}

        # â”€â”€ 1. OCR Global de la page â”€â”€
        print("  [PlanClassifier] Lecture OCR globale de la page (CRAFT)...")
        all_ocr_page = reader.readtext(img_bgr)

        # â”€â”€ CLASSIFICATION DYNAMIQUE POST-OCR (Phase 1 de la refonte) â”€â”€
        if pi == 0:
            type_plan = _refine_type_plan_from_ocr(type_plan, all_ocr_page)
            zones_def = ZONES_PAR_TYPE.get(type_plan, ZONES_PAR_TYPE["GENERIC"])
            print(f"  [PlanClassifier] Type affinÃ© par OCR dynamique : {type_plan}")

        # â”€â”€ GRAPHE SPATIAL ET EXPERT DMPC â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        _graph_page_results = {}
        
        # 1. StratÃ©gie Experte DMPC (DÃ©terministe, ultra-fiable)
        if type_plan == "DMPC":
            print(f"  [PlanClassifier] Extraction DMPC classique prioritaire...")
            _dmpc_res = _extract_dmpc_specialized(
                all_ocr_page, w, h, lambda f, v: _validate_field(f, v, commune_db),
                commune_db=commune_db,
                pdf_path=pdf_path,
            )
            if _dmpc_res:
                _graph_page_results.update(_dmpc_res)

        # 1b. StratÃ©gie Experte PVa (Texte libre tapÃ© Ã  la machine)
        if type_plan == "PVa":
            print(f"  [PlanClassifier] Extraction PVa textuelle prioritaire...")
            _pva_res = _extract_pva_textuel(
                all_ocr_page, w, h, lambda f, v: _validate_field(f, v, commune_db)
            )
            if _pva_res:
                _graph_page_results.update(_pva_res)

        # â”€â”€ GÃ‰NÃ‰RATION DES CROPS D'ANCRAGE (Phase 2 & 3) â”€â”€
        crops_data = {}
        if type_plan == "DMPC":
            dmpc_zones = {
                "commune": [0.0, 0.0, 0.40, 0.30],
                "section": [0.0, 0.0, 0.40, 0.30],
                "feuille": [0.0, 0.0, 0.40, 0.30],
                "echelle": [0.0, 0.0, 0.40, 0.30],
                "n_ordre": [0.50, 0.0, 1.0, 0.30],
                "geometre": [0.60, 0.60, 1.0, 1.0],
                "date": [0.60, 0.60, 1.0, 1.0],
                "proprietaires_anciens": [0.0, 0.70, 0.70, 1.0],
                "proprietaires_nouveaux": [0.0, 0.70, 0.70, 1.0],
                "signataires": [0.0, 0.70, 0.70, 1.0],
                "indication": [0.20, 0.60, 0.80, 1.0],
            }
            for k, z in dmpc_zones.items():
                if k in champs_attendus:
                    crops_data[k] = {"zone": z, "texte_ancre": "Zone experte DMPC"}
        elif _GENERATE_CROPS_AVAILABLE and _generate_crops is not None:
            crops_data = _generate_crops(all_ocr_page, list(champs_attendus), (h, w))
            
        print(f"  [PlanClassifier] DEBUG crops_data a {len(crops_data)} elements: {list(crops_data.keys())}")

        # 2. StratÃ©gie VLM ciblÃ©e sur les crops d'ancrage (Phase 3)
        # DÃ©sactivÃ© pour les PVa qui sont des textes tapÃ©s Ã  la machine : l'OCR pur + Regex est bien plus fiable
        # et cela Ã©vite les hallucinations sur des mots isolÃ©s.
        _vlm_res = {}
        if type_plan != "PVa":
            _vlm_res = _extract_with_vlm(img_bgr, type_plan, lambda f, v: _validate_field(f, v, commune_db), crops_data)
        if _vlm_res:
            print(f"  [PlanClassifier] Extraction VLM rÃ©ussie : {list(_vlm_res.keys())}")
            for k, v in _vlm_res.items():
                if k not in _graph_page_results:
                    _graph_page_results[k] = v

        # 2. StratÃ©gie Graphe Spatial GÃ©nÃ©rique (complÃ¨te ce qui manque)
        if _SPATIAL_AVAILABLE and _graph_extract is not None:
            _generic_graph = _graph_extract(
                all_ocr_page,
                list(champs_attendus),
                (h, w),
                validate_fn=lambda f, v: _validate_field(f, v, commune_db),
            )
            for k, v in _generic_graph.items():
                if k not in _graph_page_results:  # Ne pas Ã©craser les rÃ©sultats DMPC experts
                    _graph_page_results[k] = v
            for _gf, _gv in _graph_page_results.items():
                _lbl = _gv.get("label_trouve", "")
                print(f"    [{_gf}] -> '{_gv['valeur']}' (Graphe: {_lbl})")

        # â”€â”€ 2. Extraction par zone (sur la base de l'OCR global) â”€â”€
        # On n'extrait que les champs attendus pour le rÃƒÂ´le de cette page
        for field, zone in zones_def.items():
                if field == "parcelles":
                    continue   # TraitÃƒÂ© sÃƒÂ©parÃƒÂ©ment (dÃƒÂ©tection couleur)
                if field not in champs_attendus:
                    continue   # Ce champ n'est pas sur cette page selon la structure du document
                if field not in CONTEXTUAL_PATTERNS:
                    continue   # Champ sans pattern contextuel dÃƒÂ©fini
                # Si le champ est dÃƒÂ©jÃƒÂ  trouvÃƒÂ© sur une page prÃƒÂ©cÃƒÂ©dente, on ne l'ÃƒÂ©crase pas
                if field in doc_champs and field not in {"date"}:  # date peut se trouver sur 2 pages
                    continue
                # Si le champ a dÃ©jÃ  Ã©tÃ© trouvÃ© lors de la Passe 1 de cette page, on l'ignore
                if field in champs:
                    continue
                
                # Filtrer les rÃƒÂ©sultats de l'OCR global pour ne garder que ceux de la zone
                # zone = [x0_frac, y0_frac, x1_frac, y1_frac]
                x0, y0, x1, y1 = zone[0]*w, zone[1]*h, zone[2]*w, zone[3]*h
                ocr_zone = []
                for (bbox, text, prob) in all_ocr_page:
                    # Centre de la boite
                    cx = sum(p[0] for p in bbox) / 4.0
                    cy = sum(p[1] for p in bbox) / 4.0
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        ocr_zone.append((bbox, text, prob))
                
                result = None

                # 0. Graphe spatial â€” priorite maximale (label -> voisin direct)
                if field in _graph_page_results:
                    result = _graph_page_results[field]

                # 1. Tentative contextuelle (SEULEMENT si le graphe n'a rien trouve)
                # IMPORTANT : ne pas ecraser le resultat du graphe
                if not result:
                    result = None  # reset explicite si graphe n'a rien donne
                # 1. Tentative contextuelle (patterns label+valeur dans la mÃƒÂªme phrase)
                if not result:
                    result = _find_field_contextual(field, ocr_zone, (h, w))
                if result and result.get('methode') != 'graph_inline' and result.get('methode') != 'graph_neighbor':
                    print(f"    [{field}] Ã¢â€ â€™ '{result['valeur']}' (Contextuel)")

                # 2. Fallback robuste par mots-clÃƒÂ©s classiques (sÃƒÂ©parateur ou distance y)
                if not result and field in KEYWORDS and KEYWORDS[field]:
                    result = _find_field(field, ocr_zone, KEYWORDS[field], (h, w))
                    if result:
                        print(f"    [{field}] Ã¢â€ â€™ '{result['valeur']}' (Mots-clÃƒÂ©s)")

                # 3. Fallback IA GLiNER si les approches regex ont ÃƒÂ©chouÃƒÂ©
                if not result and _GLINER_AVAILABLE:
                    result = _semantic_extract(field, ocr_zone, (h, w))
                    if result:
                        print(f"    [{field}] Ã¢â€ â€™ '{result['valeur']}' (GLiNER)")

                if result:
                    val_valid = _validate_field(field, result["valeur"], commune_db)
                    if not val_valid:
                        result = None

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
                    print(f"    [{field}] Ã¢â€ â€™ '{result['valeur']}'")

        # Ã¢â€â‚¬Ã¢â€â‚¬ 3. Parcelles (dÃƒÂ©tection couleur Ã¢â‚¬â€ uniquement sur les pages corps/all) Ã¢â€â‚¬
        _RE_PARC = re.compile(r'^\d{1,5}[A-Za-z]?$')
        if "parcelles" in champs_attendus:
            # --- 3a. Regex contextuel dans le texte OCR complet ---
            full_text_page = "\n".join(r[1] for r in all_ocr_page)
            parcelles_ctx: list = []
            
            # Format classique sur une ligne: "parcelle nÂ° 12, 14"
            for m in re.finditer(
                r'(?:parcelle[s]?\s*n[o\xb0][s]?|n[o\xb0]\s*de\s*parcelle[s]?)\s*[:\-]?\s*'
                r'(\d{1,5}[A-Za-z]?(?:\s*[,/;]\s*\d{1,5}[A-Za-z]?)*)',
                full_text_page, re.IGNORECASE
            ):
                nums = re.findall(r'\d{1,5}[A-Za-z]?', m.group(1))
                parcelles_ctx.extend(nums)

            # Format tableau / liste multi-lignes : "propriÃ©taires des parcelles ci-aprÃ¨s \n 12 \n 14"
            for m in re.finditer(
                r'(?:parcelle[s]?|section[s]?)[^\n]{0,80}\n'
                r'((?:\s*(?:n[o\xb0]\s*)?\d{1,5}[A-Za-z]?(?:\s*[,/;]\s*|\s*\n\s*|\s+|$)){1,15})',
                full_text_page, re.IGNORECASE
            ):
                nums = re.findall(r'\b\d{1,5}[A-Za-z]?\b', m.group(1))
                # Filtrer les annÃ©es pour Ã©viter les faux positifs (dates de naissance, dates d'actes)
                nums = [n for n in nums if not (len(n) == 4 and n.startswith(('19', '20')))]
                parcelles_ctx.extend(nums)

            # --- 3b. Extraction par Couleur (SEULEMENT pour PLa et DMPC graphiques) ---
            parcelles_nouveaux, parcelles_anciens = [], []
            if type_plan != "PVa":
                try:
                    from color_ocr_engine import extract_color_parcels
                    print("    [parcelles] Extraction couleur (rouge/vert)...")
                    color_res = extract_color_parcels(img_bgr)
                    
                    if color_res["nouvelles_parcelles"]:
                        parcelles_nouveaux = [p["valeur"] for p in color_res["nouvelles_parcelles"]]
                        champs["nouvelles_parcelles"] = {
                            "valeur": parcelles_nouveaux,
                            "zone": [0,0,0,0],
                            "methode": "couleur_rouge"
                        }
                    
                    if color_res["anciennes_parcelles"]:
                        parcelles_anciens = [p["valeur"] for p in color_res["anciennes_parcelles"]]
                        champs["anciennes_parcelles"] = {
                            "valeur": parcelles_anciens,
                            "zone": [0,0,0,0],
                            "methode": "couleur_vert"
                        }
                except Exception as e:
                    print(f"    [parcelles] Erreur extraction couleur : {e}")
            else:
                print("    [parcelles] PVa dÃ©tectÃ© â†’ dÃ©tection couleur HSV dÃ©sactivÃ©e (texte tapÃ©)")

            parcelles_finales = []
            for p in parcelles_ctx:
                parcelles_finales.append(f"{p} (Nouveau)")
            for p in parcelles_nouveaux:
                tag = f"{p} (Nouveau)"
                if tag not in parcelles_finales and p not in parcelles_ctx:
                    parcelles_finales.append(tag)
            for p in parcelles_anciens:
                tag = f"{p} (Ancien)"
                if tag not in parcelles_finales:
                    parcelles_finales.append(tag)

            all_bboxes = []
            if 'color_res' in locals():
                all_bboxes.extend([p['bbox'] for p in color_res.get('nouvelles_parcelles', [])])
                all_bboxes.extend([p['bbox'] for p in color_res.get('anciennes_parcelles', [])])
            
            if all_bboxes:
                all_xs = [x for bbox in all_bboxes for x in (bbox[0], bbox[2])]
                all_ys = [y for bbox in all_bboxes for y in (bbox[1], bbox[3])]
                parc_zone = [min(all_xs)/w, min(all_ys)/h, max(all_xs)/w, max(all_ys)/h]
            else:
                parc_zone = [0.0, 0.0, 1.0, 1.0]

            champs["parcelles"] = {"valeur": parcelles_finales, "zone": parc_zone}
            if parcelles_finales:
                print(f"    [parcelles] Ã¢â€ â€™ {parcelles_finales}")
        else:
            # all_ocr_page est dÃ©jÃ  calculÃ© en dÃ©but de boucle â€” on rÃ©utilise
            full_text_page = "\n".join(r[1] for r in all_ocr_page)

        # Ã¢â€â‚¬Ã¢â€â‚¬ 3. Commune : fallback zone large (seulement si en-tÃƒÂªte ou all) Ã¢â€â‚¬Ã¢â€â‚¬
        if "commune" in champs_attendus and "commune" not in doc_champs and "commune" not in champs:
            zone_large = zones_def.get("commune", [0.0, 0.0, 0.6, 0.3])
            ocr_large = _ocr_in_zone(reader, img_bgr, zone_large)
            result_c = _find_field_contextual("commune", ocr_large, (h, w))
            if not result_c and _GLINER_AVAILABLE:
                result_c = _semantic_extract("commune", ocr_large, (h, w))
            if result_c:
                val_brute = result_c["valeur"]
                val_m = _match_commune(val_brute, commune_db, seuil=70) if commune_db else val_brute
                champs["commune"] = {
                    "valeur": val_m,
                    "zone": result_c["zone"],
                    "brut": result_c.get("brut", ""),
                }

        # Ã¢â€â‚¬Ã¢â€â‚¬ 4. Traitements MÃƒÂ©tier spÃƒÂ©cifiques avancÃƒÂ©s (DMPC & PVa) Ã¢â€â‚¬Ã¢â€â‚¬
        if type_plan == "DMPC":
            # Zones du bas (y > 0.52)
            bottom_items = [r for r in all_ocr_page if min(pt[1] for pt in r[0]) > 0.52 * h]
            bottom_text = "\n".join(r[1] for r in bottom_items)

            # A. Proposition certifiÃƒÂ©e par les propriÃƒÂ©taires non barrÃƒÂ©e
            # Le texte barrÃƒÂ© a gÃƒÂ©nÃƒÂ©ralement une faible confiance OCR. On filtre par confiance.
            options_found = {}
            for r in bottom_items:
                if len(r) < 3 or r[2] < 0.25: continue
                txt_low = r[1].lower()
                prob = r[2]
                if "dispense" in txt_low: options_found["dispensÃƒÂ©s de signer (3Ã‚Â°)"] = prob
                if "represente" in txt_low: options_found["reprÃƒÂ©sentÃƒÂ©s (2Ã‚Â°)"] = prob
                if "soussigne" in txt_low: options_found["soussignÃƒÂ©s (1Ã‚Â°)"] = prob
            
            # Si on trouve un nom aprÃƒÂ¨s reprÃƒÂ©sentÃƒÂ©s, c'est forcÃƒÂ©ment l'option choisie
            m_rep = re.search(r'repr[e\xe9]sent[e\xe9]s?\s+(?:par\s+)?([A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-\.]{2,40})', bottom_text, re.IGNORECASE)
            if m_rep:
                prop_found = f"reprÃƒÂ©sentÃƒÂ©s par {m_rep.group(1).strip()} (2Ã‚Â°)"
            elif options_found:
                # Option avec la plus grande confiance de lecture OCR (non barrÃƒÂ©e)
                prop_found = max(options_found.items(), key=lambda x: x[1])[0]
            else:
                prop_found = "soussignÃƒÂ©s (1Ã‚Â°)" # Fallback par dÃƒÂ©faut
            
            certif_str = f"CertifiÃƒÂ© par les propriÃƒÂ©taires : {prop_found}"
            if "indication" not in champs or not champs["indication"].get("valeur"):
                champs["indication"] = {
                    "valeur": certif_str,
                    "zone": [0.0, 0.52, 1.0, 0.95],
                    "brut": certif_str
                }
            else:
                champs["indication"]["valeur"] += f" | {certif_str}"

            # B. Anciens fallbacks n_ordre et geometre supprimÃ©s car ils contournaient le validateur.

            # C. Double vÃ©rification Commune et Date (Fait Ã  ... le ...)
            m_fait = re.search(
                r'(?:fait|r[e\xe9]digu[e\xe9]|\b[a\xe0]\b)\s+([A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-]{2,30})[,\s]+le\s+'
                r'(\d{1,2}\s+[a-z\xc0-\xff]+\s+\d{4}|\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                bottom_text, re.IGNORECASE
            )
            if m_fait:
                comm_bottom = m_fait.group(1).strip()
                date_bottom = m_fait.group(2).strip()
                if "commune" not in champs or not champs["commune"].get("valeur"):
                    comm_matched = _match_commune(comm_bottom, commune_db) if commune_db else comm_bottom
                    champs["commune"] = {
                        "valeur": comm_matched,
                        "zone": [0.0, 0.55, 0.8, 0.95],
                        "brut": m_fait.group(0)
                    }
                if "date" not in champs or not champs["date"].get("valeur"):
                    champs["date"] = {
                        "valeur": date_bottom,
                        "zone": [0.0, 0.55, 0.8, 0.95],
                        "brut": m_fait.group(0)
                    }

        elif type_plan == "PVa":
            # PV de bornage/division : extraction des signataires
            # On laisse la prioritÃ© absolue Ã  LLaVa (qui a compris sÃ©mantiquement le document)
            if "signataires" not in champs or not champs["signataires"].get("valeur"):
                # Liste noire des termes qui ne sont PAS des signataires
                NOMS_EXCLUS_SIGNATAIRES = {
                    "COMMUNE", "SECTION", "FEUILLE", "ECHELLE", "PLAN", "CADASTRE",
                    "DMPC", "DATE", "PROCES VERBAL", "PROCÃƒË†S VERBAL", "BORNAGE",
                    "ARPENTAGE", "LOTISSEMENT", "DIVISION", "GEOMETRE", "GEOMETRES",
                    "GEOMETRES EXPERTS", "GEOMETRES-EXPERTS", "EXPERT",
                    # Noms de villes courantes dans ces documents
                    "VALS LES BAINS", "VALLON PONT", "VALLON PONT D ARC",
                    "GUILHERAND GRANGES", "GUILHERAND", "ANNONAY", "AUBENAS",
                    "ARDECHE", "DROME", "FRANCE",
                }
                signataires_detectes = []
                for item in all_ocr_page:
                    txt = item[1].strip()
                    # --- PrioritÃ© 1 : civilitÃ©s explicites (M., Mme, Monsieur, Madame) ---
                    m_sig = re.search(
                        r'\b(?:M\.?|Mme\.?|Monsieur|Madame|Mlle\.?)\s+'
                        r'([A-Z\xc0-\xdd][A-Za-z\xc0-\xff][A-Za-z\xc0-\xff\s\-]{1,30})',
                        txt
                    )
                    if m_sig:
                        full = m_sig.group(0).strip()
                        if _is_valid_signataire(full) and full not in signataires_detectes:
                            signataires_detectes.append(full)
                    else:
                        # --- PrioritÃ© 2 : pattern contextuel (certifiÃ© exact, soussignÃ©, vu et approuvÃ©) ---
                        m_ctx = re.search(
                            r'(?:certifi[e\xe9]\s+exact|vu\s+et\s+approuv[e\xe9]|soussign[e\xe9](?:e?s)?)'
                            r'[,\s:]+([A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-\.]{3,50})',
                            txt, re.IGNORECASE
                        )
                        if m_ctx:
                            name = m_ctx.group(1).strip().rstrip('.,;:')
                            if _is_valid_signataire(name) and name not in signataires_detectes:
                                signataires_detectes.append(name)
                        # --- PAS de capture gÃ©nÃ©rique en majuscules : trop de bruit ---
                
                if signataires_detectes:
                    signataires_detectes = list(dict.fromkeys(signataires_detectes))
                    champs["signataires"] = {
                        "valeur": signataires_detectes,
                        "zone": [0.0, 0.0, 1.0, 1.0],
                        "brut": ", ".join(signataires_detectes)
                    }

            # Fallback PVa bypassÃ© pour respecter la liste stricte des gÃ©omÃ¨tres.

            # Ã¢â€â‚¬Ã¢â€â‚¬ Fallback Global Infaillible pour GÃƒÂ©omÃƒÂ¨tres Connus Ã¢â€â‚¬Ã¢â€â‚¬
            if "geometre" not in champs:
                full_text_page_low = full_text_page.lower()
                for nom_connu in GEOMETRES_CONNUS:
                    if nom_connu.lower() in full_text_page_low:
                        # Si trouvÃ© n'importe oÃ¹ dans le texte, on l'attribue
                        champs["geometre"] = {
                            "valeur": nom_connu,
                            "zone": [0.0, 0.0, 1.0, 1.0],
                            "brut": f"Scanner global: {nom_connu}"
                        }
                        print(f"    [geometre] Ã¢â€ â€™ '{nom_connu}' (Fallback Global VIP)")
                        break
            
            # Ã¢â€â‚¬Ã¢â€â‚¬ Fallback Global Infaillible pour Section et Feuille Ã¢â€â‚¬Ã¢â€â‚¬
            if "section" not in champs:
                m_sec = re.search(r'(?i)\bsection\s+([A-Z]{1,2})\b', full_text_page)
                if m_sec:
                    champs["section"] = {
                        "valeur": m_sec.group(1).upper(),
                        "zone": [0.0, 0.0, 1.0, 1.0],
                        "brut": m_sec.group(0)
                    }
                    print(f"    [section] Ã¢â€ â€™ '{m_sec.group(1).upper()}' (Fallback Global)")
            if "feuille" not in champs:
                m_feu = re.search(r'(?i)\bfeuille\s+(?:n[oÃ‚Â°])?\s*(\d{1,3})\b', full_text_page)
                if m_feu:
                    champs["feuille"] = {
                        "valeur": m_feu.group(1),
                        "zone": [0.0, 0.0, 1.0, 1.0],
                        "brut": m_feu.group(0)
                    }
                    print(f"    [feuille] Ã¢â€ â€™ '{m_feu.group(1)}' (Fallback Global)")
        # (Logique supprimÃ©e ou ignorÃ©e)

                # â”€â”€ FULL PAGE VLM FALLBACK (Phase 4) â”€â”€
        all_missing = [f for f in champs_attendus if f not in champs or not champs.get(f, {}).get("valeur")]
        if all_missing and type_plan in ["CROQUIS", "PVa", "GENERIC", "DMPC"]:
            print(f"  [PlanClassifier] Champs manquants {all_missing}. Appel VLM Pleine Page...")
            vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db, type_plan=type_plan)
            for k, v in vlm_full.items():
                if k not in champs or not champs.get(k, {}).get("valeur"):
                    champs[k] = v

        # â”€â”€ Cycle 1 : VÃ©rification Autonome â”€â”€
        champs = _verify_and_revise(
            champs,
            all_ocr_page=all_ocr_page,
            img_shape=(h, w),
            full_text=" ".join(r[1] for r in all_ocr_page),
            commune_db=commune_db,
        )

        # â”€â”€ Cycle 2 : Vote inter-stratÃ©gies + DÃ©tection des valeurs fausses â”€â”€â”€â”€â”€
        # Principe : un champ peut Ãªtre "trouvÃ©" mais sÃ©mantiquement incorrect.
        # On extrait chaque champ critique par 3 stratÃ©gies indÃ©pendantes et on
        # compare. Si les rÃ©sultats divergent â†’ conflit â†’ on re-arbitre.
        # Cas dÃ©clencheurs supplÃ©mentaires par rapport au simple "absent" :
        #   - prop_anciens == prop_nouveaux (glissement OCR)
        #   - gÃ©omÃ¨tre ressemble Ã  un lieu (pas un nom propre)
        #   - indication est un fragment (commence par une minuscule ou < 10 chars)
        #   - valeur confirmÃ©e par 1 seule mÃ©thode et confidence < 0.85

        CHAMPS_CRITIQUES = {"commune", "n_ordre", "proprietaires_anciens",
                            "proprietaires_nouveaux", "geometre", "date", "indication"}

        def _est_suspect_semantiquement(field: str, data: dict) -> bool:
            """DÃ©tecte les valeurs trouvÃ©es mais sÃ©mantiquement douteuses."""
            if not isinstance(data, dict):
                return False
            val = str(data.get("valeur", "")).strip()
            if not val:
                return True  # Vide = toujours suspect

            # Confidence faible mÃªme si status OK
            if data.get("confidence", 0) < 0.80:
                return True

            if field == "indication":
                # Fragment = commence par minuscule ou trop court
                if len(val) < 10 or (val[0].islower() and val[0] not in "0123456789"):
                    return True

            if field == "geometre":
                # Doit ressembler Ã  un nom propre (NOM PrÃ©nom ou inverse)
                # Pas uniquement des mots en majuscules qui pourraient Ãªtre des lieux
                mots = val.split()
                all_upper = all(m.isupper() for m in mots)
                has_lower = any(c.islower() for c in val)
                if len(mots) <= 2 and all_upper and not has_lower:
                    return True  # ex: "RTs N", "EXPERTS BP"
                if len(val) <= 7:
                    return True  # Trop court
                # Boilerplate juridique : phrases figÃ©es des pages de signatures
                _BOILERPLATE_GEO = [
                    "soussign", "accepte", "reserve", "certifi", "signature",
                    "date et", "et signature", "approuve", "mentions", "ci-dessus",
                    "sous reserve", "sans reserve", "parties", "expert foncier",
                    "geometre expert", "le present", "toutes reserves",
                ]
                val_low = val.lower()
                if any(bp in val_low for bp in _BOILERPLATE_GEO):
                    return True  # Phrase juridique, pas un nom de cabinet

            return False

        # DÃ©tecter aussi les contradictions inter-champs
        _champs_a_recontroler = set()
        for f in CHAMPS_CRITIQUES:
            data = champs.get(f)
            if data and _est_suspect_semantiquement(f, data):
                _champs_a_recontroler.add(f)
            elif not data or not (data.get("valeur") if isinstance(data, dict) else data):
                _champs_a_recontroler.add(f)

        # Contradiction anciens == nouveaux
        _anc = str((champs.get("proprietaires_anciens") or {}).get("valeur", ""))
        _nouv = str((champs.get("proprietaires_nouveaux") or {}).get("valeur", ""))
        if _anc and _nouv and _anc.lower() == _nouv.lower():
            print(f"  [Vote] âš  prop_anciens == prop_nouveaux ('{_anc}') â†’ relance prop_nouveaux")
            _champs_a_recontroler.add("proprietaires_nouveaux")
            if "proprietaires_nouveaux" in champs:
                champs["proprietaires_nouveaux"]["valeur"] = ""

        # Filtrer aux champs attendus sur cette page
        _champs_a_recontroler = {f for f in _champs_a_recontroler if f in champs_attendus}

        if _champs_a_recontroler:
            print(f"  [Vote] ContrÃ´le multi-stratÃ©gies pour : {sorted(_champs_a_recontroler)}")
            all_ocr_elargi = [
                (bbox, text, prob) for (bbox, text, prob) in all_ocr_page
                if prob >= 0.08 and text.strip()
            ]
            full_text_elargi = " ".join(r[1] for r in all_ocr_elargi)

            # â”€â”€ Correction 3 : Blacklist de rÃ©utilisation de valeurs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Si un champ a dÃ©jÃ  une valeur assignÃ©e avec haute confiance, on l'ajoute
            # Ã  une blacklist pour Ã©viter qu'elle soit rÃ©affectÃ©e Ã  un autre champ.
            # Ex: "LIONNEL ROBERT" est gÃ©omÃ¨tre â†’ ne peut pas Ãªtre propriÃ©taire.
            CHAMPS_EXCLUSIFS = {
                # {champ_source: [champs_cibles interdits]}
                "geometre": ["proprietaires_anciens", "proprietaires_nouveaux", "signataires"],
                "commune":  ["proprietaires_anciens", "proprietaires_nouveaux", "geometre"],
                "section":  ["commune", "geometre"],
            }
            _valeurs_blacklist: set = set()  # set de valeurs (lowercase) dÃ©jÃ  fiablement assignÃ©es
            for src_field, forbidden_fields in CHAMPS_EXCLUSIFS.items():
                src_data = champs.get(src_field)
                if isinstance(src_data, dict):
                    src_conf = src_data.get("confidence", 0)
                    src_val  = str(src_data.get("valeur", "")).strip().lower()
                    # On blackliste si la valeur est assez longue et assignÃ©e avec confiance
                    if src_val and src_conf >= 0.75 and len(src_val) > 4:
                        for tgt in forbidden_fields:
                            if tgt in _champs_a_recontroler:
                                _valeurs_blacklist.add(src_val)

            for field in sorted(_champs_a_recontroler):
                candidats = {}  # {valeur_normalisee: {"conf": float, "methode": str, "raw": dict}}
                val_existante = str((champs.get(field) or {}).get("valeur", "")) if isinstance(champs.get(field), dict) else ""

                # StratÃ©gie 1 : graphe spatial (seuil Ã©largi)
                if _SPATIAL_AVAILABLE and _graph_extract is not None:
                    g2 = _graph_extract(all_ocr_elargi, [field], (h, w), validate_fn=lambda f, v: _validate_field(f, v, commune_db))
                    if field in g2:
                        v = g2[field]["valeur"]
                        candidats[v.lower()] = {"conf": g2[field].get("confidence", 0.80), "methode": "graphe", "raw": g2[field]}

                # StratÃ©gie 2 : contextuel
                res_ctx = _find_field_contextual(field, all_ocr_elargi, (h, w))
                if res_ctx and res_ctx.get("valeur"):
                    v = _validate_field(field, res_ctx["valeur"], commune_db)
                    if v:
                        if field == "commune" and commune_db:
                            v = _match_commune(v, commune_db)
                        candidats[v.lower()] = {"conf": 0.72, "methode": "contextuel", "raw": res_ctx}

                # StratÃ©gie 3 : mots-clÃ©s
                if field in KEYWORDS and KEYWORDS[field]:
                    res_kw = _find_field(field, all_ocr_elargi, KEYWORDS[field], (h, w))
                    if res_kw and res_kw.get("valeur"):
                        v = _validate_field(field, res_kw["valeur"], commune_db)
                        if v:
                            if field == "commune" and commune_db:
                                v = _match_commune(v, commune_db)
                            candidats[v.lower()] = {"conf": 0.70, "methode": "mots_cles", "raw": res_kw}

                # StratÃ©gie 4 : GLiNER (si disponible)
                if _GLINER_AVAILABLE:
                    res_gl = _semantic_extract(field, all_ocr_elargi, (h, w))
                    if res_gl and res_gl.get("valeur"):
                        v = _validate_field(field, res_gl["valeur"], commune_db)
                        if v:
                            if field == "commune" and commune_db:
                                v = _match_commune(v, commune_db)
                            # GLiNER ne remplace que si diffÃ©rent de l'existant
                            k = v.lower()
                            if k not in candidats:
                                candidats[k] = {"conf": 0.75, "methode": "gliner", "raw": res_gl}
                            else:
                                # Accord GLiNER â†’ bonus de confiance
                                candidats[k]["conf"] = min(0.97, candidats[k]["conf"] + 0.12)
                                candidats[k]["methode"] += "+gliner"

                # â”€â”€ Correction 3 : filtrer les candidats blacklistÃ©s â”€â”€
                # Un candidat dont la valeur est dÃ©jÃ  assignÃ©e Ã  un autre champ
                # avec haute confiance est retirÃ© de la course.
                if _valeurs_blacklist and field in ("proprietaires_anciens", "proprietaires_nouveaux", "signataires", "geometre"):
                    candidats_filtres = {
                        k: v for k, v in candidats.items()
                        if not any(k.startswith(bl) or bl.startswith(k) for bl in _valeurs_blacklist)
                    }
                    if candidats_filtres:
                        if len(candidats_filtres) < len(candidats):
                            evicted = set(candidats.keys()) - set(candidats_filtres.keys())
                            print(f"    [Vote][{field}] ðŸš« Blacklist Ã©limine : {evicted}")
                        candidats = candidats_filtres

                # Arbitrage : choisir le candidat avec le plus de votes + confiance max
                if not candidats:
                    print(f"    [Vote][{field}] Aucun candidat trouvÃ©")
                    continue

                # Compter les votes (nombre de mÃ©thodes qui ont convergÃ© sur cette valeur)
                meilleur_k = max(candidats, key=lambda k: (
                    candidats[k]["methode"].count("+") + 1,  # nombre de mÃ©thodes d'accord
                    candidats[k]["conf"]
                ))
                meilleur = candidats[meilleur_k]
                nb_methodes = meilleur["methode"].count("+") + 1
                nb_total = len(candidats)
                accord = nb_methodes > 1

                val_choisie = meilleur["raw"]["valeur"] if "raw" in meilleur else meilleur_k
                conf_finale = meilleur["conf"]

                if val_existante and val_choisie.lower() == val_existante.lower():
                    if nb_methodes > 1:
                        print(f"    [Vote][{field}] âœ“ '{val_choisie}' confirmÃ© ({nb_methodes}/{nb_total} mÃ©thodes)")
                        if isinstance(champs.get(field), dict):
                            champs[field]["confidence"] = min(0.97, conf_finale + 0.08)
                            champs[field]["verification_notes"] = champs[field].get("verification_notes", []) + [f"ConfirmÃ© par {nb_methodes} stratÃ©gies"]
                            champs[field]["verification_status"] = "OK"
                    else:
                        print(f"    [Vote][{field}] âš  '{val_choisie}' trouvÃ© par 1 mÃ©thode seulement")
                else:
                    if accord or conf_finale > (champs.get(field) or {}).get("confidence", 0):
                        print(f"    [Vote][{field}] â†º '{val_existante}' â†’ '{val_choisie}' ({nb_methodes} mÃ©thodes, {conf_finale:.0%})")
                        champs[field] = {
                            **(meilleur.get("raw", {})),
                            "valeur": val_choisie,
                            "confidence": round(conf_finale, 2),
                            "verification_status": "OK" if conf_finale >= 0.65 else "INCERTAIN",
                            "verification_notes": [f"Vote: {nb_methodes}/{nb_total} mÃ©thodes"],
                            "cycle": 2,
                        }
                    else:
                        print(f"    [Vote][{field}] = '{val_existante}' conservÃ© (divergence sans consensus)")

            # Re-validation finale aprÃ¨s vote
            champs = _verify_and_revise(
                champs,
                all_ocr_page=all_ocr_elargi,
                img_shape=(h, w),
                full_text=full_text_elargi,
                commune_db=commune_db,
            )

        # â”€â”€ VÃ©rification de cohÃ©rence inter-champs (4 couches) â”€â”€
        # S'exÃ©cute sur les champs vÃ©rifiÃ©s de cette page avant la fusion
        coherence_report = None
        if _COHERENCE_AVAILABLE:
            coherence_report = _check_coherence(
                champs, type_plan,
                use_llm=True  # Ollama est silencieux si non installÃ©
            )
            # Attacher le rapport Ã  chaque champ concernÃ©
            for field, field_issues in coherence_report.get("issues_by_field", {}).items():
                if field in champs and isinstance(champs[field], dict):
                    existing = champs[field].get("coherence_issues", [])
                    champs[field]["coherence_issues"] = existing + field_issues
            # Stocker le rÃ©sumÃ© global
            champs["_coherence"] = {
                "valeur": coherence_report["summary"],
                "status": coherence_report["status"],
                "score": coherence_report["coherence_score"],
                "counts": coherence_report["counts"],
            }

        # â”€â”€ Fusion dans doc_champs â”€â”€
        for k, v in champs.items():
            if k not in doc_champs or k == "date":
                # On ajoute le numÃ©ro de page pour l'interface de validation
                if isinstance(v, dict):
                    v["page"] = page_num
                doc_champs[k] = v


        # â”€â”€ 6. Image annotÃ©e (affiche les champs doc_champs + champs page) â”€â”€
        # RÃ©utilise all_ocr_page dÃ©jÃ  calculÃ© â€” on Ã©vite un double appel OCR coÃ»teux
        ann = img_bgr.copy()
        for (bbox, text, prob) in all_ocr_page:
            pts = np.array(bbox, np.int32)
            cv2.polylines(ann, [pts], True, (200, 200, 200), 1)
            xt, yt = int(bbox[0][0]), max(int(bbox[0][1]) - 3, 10)
            cv2.putText(ann, text[:25], (xt, yt), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (120, 120, 120), 1, cv2.LINE_AA)
        # Annoter uniquement les champs trouvÃ©s sur cette page
        for field, info in champs.items():
            if not isinstance(info, dict) or "zone" not in info:
                continue
            z = info["zone"]
            if len(z) != 4 or z == [0.0, 0.0, 0.0, 0.0]:
                continue
            x0p, y0p = int(z[0]*w), int(z[1]*h)
            x1p, y1p = int(z[2]*w), int(z[3]*h)
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
        print(f"  [PlanClassifier] Image annotÃ©e : {out_path}")

    doc.close()
    
    # Mapping Nature de l'Acte pour Geofoncier
    def _map_nature_acte(ind: str) -> str:
        if not ind: return ""
        ind = ind.lower()
        if "bornage" in ind: return "BORNAGE"
        if "division" in ind or "distraction" in ind: return "DIVISION_PARCELLAIRE"
        if "lotissement" in ind: return "LOTISSEMENT"
        if "reunion" in ind or "rÃƒÂ©union" in ind: return "REUNION_PARCELLAIRE"
        if "alignement" in ind or "limite" in ind: return "RECONNAISSANCE_LIMITES"
        if "dmpc" in ind or "document d'arpentage" in ind: return "DMPC"
        return "AUTRE"
        
    ind_val = str((doc_champs.get("indication") or {}).get("valeur", ""))
    if ind_val:
        page_val = (doc_champs.get("indication") or {}).get("page", 1)
        doc_champs["nature_acte_geofoncier"] = {
            "valeur": _map_nature_acte(ind_val),
            "zone": [0,0,0,0],
            "brut": ind_val,
            "page": page_val
        }

    # Retour consolidÃƒÂ© : 1 seul niveau de champs pour tout le document
    return {
        "fichier": pdf_path,
        "type_plan": type_plan,
        "nb_pages": nb_pages,
        "champs": doc_champs,                       # Ã¢â€ ï¿½ ConsolidÃƒÂ© document
        "pages": [{"page": 1, "champs": doc_champs}],  # CompatibilitÃƒÂ© Streamlit/CSV
    }


def export_plan_to_csv(res: dict, output_dir: str = "outputs") -> str:
    base = os.path.splitext(os.path.basename(res["fichier"]))[0]
    csv_path = os.path.join(output_dir, f"{base}_plan_resultats.csv")

    rows = []

    def _g(champs, f):
        """Extrait la valeur d'un champ et corrige l'encodage Mojibake avant export."""
        v = champs.get(f, {})
        if isinstance(v, dict):
            val = v.get("valeur", "")
            raw = ", ".join(val) if isinstance(val, list) else str(val)
        else:
            raw = ""
        return _reparer_encodage(raw)

    # 1. Si on a des pages, on les exporte toutes
    if "pages" in res and res["pages"]:
        for pg in res["pages"]:
            champs = pg.get("champs", {})
            rows.append({
                "ID": f"p{pg.get('page', 1)}",
                "Page": pg.get("page", 1),
                "Type_Plan": res.get("type_plan", ""),
                "Commune":                 _g(champs, "commune"),
                "N_Ordre":                 _g(champs, "n_ordre"),
                "N_Dossier":               _g(champs, "n_dossier"),
                "Section":                 _g(champs, "section"),
                "Feuille":                 _g(champs, "feuille"),
                "Date":                    _g(champs, "date"),
                "Echelle":                 _g(champs, "echelle"),
                "Geometre":                _g(champs, "geometre"),
                "Signataires":             _g(champs, "signataires"),
                "Proprietaires_Anciens":   _g(champs, "proprietaires_anciens"),
                "Proprietaires_Nouveaux":  _g(champs, "proprietaires_nouveaux"),
                "Parcelles":               _g(champs, "parcelles"),
                "Nouvelles_Parcelles":     _g(champs, "nouvelles_parcelles"),
                "Anciennes_Parcelles":     _g(champs, "anciennes_parcelles"),
                "Indication":              _g(champs, "indication"),
                "Nature_Acte_Geofoncier":  _g(champs, "nature_acte_geofoncier"),
                "Confirmation_Status": "Ã€ valider",

            })

    # 2. Sinon, ou en fallback, si on a un dictionnaire global "champs"
    if not rows and "champs" in res and res["champs"]:
        champs = res["champs"]
        rows.append({
            "ID": "p1",
            "Page": 1,
            "Type_Plan": res.get("type_plan", ""),
            "Commune":                 _g(champs, "commune"),
            "N_Ordre":                 _g(champs, "n_ordre"),
            "N_Dossier":               _g(champs, "n_dossier"),
            "Section":                 _g(champs, "section"),
            "Feuille":                 _g(champs, "feuille"),
            "Date":                    _g(champs, "date"),
            "Echelle":                 _g(champs, "echelle"),
            "Geometre":                _g(champs, "geometre"),
            "Signataires":             _g(champs, "signataires"),
            "Proprietaires_Anciens":   _g(champs, "proprietaires_anciens"),
            "Proprietaires_Nouveaux":  _g(champs, "proprietaires_nouveaux"),
            "Parcelles":               _g(champs, "parcelles"),
            "Nouvelles_Parcelles":     _g(champs, "nouvelles_parcelles"),
            "Anciennes_Parcelles":     _g(champs, "anciennes_parcelles"),
            "Indication":              _g(champs, "indication"),
            "Nature_Acte_Geofoncier":  _g(champs, "nature_acte_geofoncier"),
            "Confirmation_Status": "Ã€ valider",

        })

    os.makedirs(output_dir, exist_ok=True)
    # utf-8-sig = UTF-8 avec BOM, reconnu nativement par Excel et Geofoncier
    pd.DataFrame(rows).to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig")
    print(f"  [PlanClassifier] CSV exportÃ© : {csv_path}")
    return csv_path

