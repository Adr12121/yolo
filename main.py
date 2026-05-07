    # pyre-ignore-all-errors
# pyright: reportMissingImports=false, reportGeneralTypeIssues=false
import os
import sys

# === SAUVEGARDE AUTOMATIQUE DES LOGS ===
# Peu importe comment le script est lancé, tous les prints() sont aussi
# sauvegardés dans outputs/log_exec.txt pour le débogage.
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(_log_dir, 'log_exec.txt')

class _TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files: f.flush()

_log_file = open(_log_path, 'w', encoding='utf-8')
sys.stdout = _TeeOutput(sys.__stdout__, _log_file)
sys.stderr = _TeeOutput(sys.__stderr__, _log_file)
print(f"[LOG] Sortie enregistrée dans : {_log_path}")
# =======================================

import json
try:
    from semantic_ocr_engine import (
        process_with_semantic_context,
        CorrectionLearner,
        identify_field_type,
    )
    _SEMANTIC_ENGINE_AVAILABLE = True
except ImportError as _e_sem:
    print(f"[WARN] semantic_ocr_engine non disponible : {_e_sem}")
    _SEMANTIC_ENGINE_AVAILABLE = False

try:
    from modern_plan_extractor import (
        process_modern_plan,
        export_modern_plan_to_csv,
        is_modern_plan,
    )
    _MODERN_PLAN_AVAILABLE = True
except ImportError as _e_mod:
    print(f"[WARN] modern_plan_extractor non disponible : {_e_mod}")
    _MODERN_PLAN_AVAILABLE = False

try:
    from plan_classifier import (
        is_plan_document,
        classify_plan,
        process_plan,
        export_plan_to_csv,
    )
    _PLAN_CLASSIFIER_AVAILABLE = True
except ImportError as _e_cls:
    print(f"[WARN] plan_classifier non disponible : {_e_cls}")
    _PLAN_CLASSIFIER_AVAILABLE = False

# Pipeline plans activé
_MODERN_PLAN_AVAILABLE = True
import cv2  # type: ignore
import numpy as np  # type: ignore
from typing import Any, DefaultDict, Dict, List, Tuple, Optional
import pandas as pd  # type: ignore
import fitz  # PyMuPDF  # type: ignore
from ultralytics import YOLO  # type: ignore
import easyocr  # type: ignore
import subprocess
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import torch  # type: ignore
import re
from collections import defaultdict
from kraken import blla, rpred  # type: ignore
from kraken.lib import models as kraken_models  # type: ignore
import unicodedata
from spellchecker import SpellChecker  # type: ignore

try:
    from rapidfuzz import process, fuzz  # type: ignore
except ImportError:
    print("Attention: rapidfuzz n'est pas installé. La correction des communes sera désactivée.")
    process = None

# ============================================================
# MOTS COURANTS FRANÇAIS À IGNORER DANS LE BRUTE-FORCE COMMUNE
# (pour éviter que "POUR", "DANS", "AVEC" scorent frauduleusement)
# ============================================================
_MOTS_COURANTS_FR = {
    'POUR', 'DANS', 'AVEC', 'SANS', 'SOUS', 'TRES', 'BIEN', 'MAIS',
    'AUSSI', 'PLUS', 'TOUT', 'TOUS', 'CETTE', 'ENTRE', 'DONT', 'LORS',
    'VOIE', 'VOIES', 'ROUTE', 'CHEMIN', 'LIEU', 'LIEUX', 'LIEU DIT',
    'PLAN', 'DATE', 'SIGNE', 'SIGNER', 'CERTIFIE', 'APPROUVE', 'VU',
    'LEDIT', 'LADIT', 'DITE', 'DUDIT', 'AUDIT', 'AUXDITS', 'SUSDITS',
    'PAGE', 'PAGES', 'NOTE', 'FOLIO', 'REGISTRE', 'ARCHIVE',
    'JEAN', 'PIERRE', 'PAUL', 'MARIE', 'ANDRE', 'LOUIS', 'HENRI',
    'PREFECTURE', 'DEPARTEMENT', 'CANTON', 'ARRONDISSEMENT',
    'PROPRIETE', 'PROPRIETAIRE', 'PARCELLE', 'SECTION', 'LIEUDIT',
    'ECHELLE', 'DOSSIER', 'REFERENCE', 'NUMERO', 'ORDRE',
    'SURFACE', 'CONTENANCE', 'SUPERFICIE', 'MESURAGE', 'BORNAGE',
    'ANNEXE', 'TITRE', 'OBJET', 'TYPE', 'NATURE', 'ACTE', 'ACTES',
    'FRANC', 'FRANCS', 'METRE', 'METRES', 'HECTARE', 'ARES', 'CENTIARES',
    'PREMIER', 'DEUXIEME', 'TROISIEME', 'JANVIER', 'FEVRIER', 'MARS',
    'AVRIL', 'JUIN', 'JUILLET', 'AOUT', 'SEPTEMBRE', 'OCTOBRE',
    'NOVEMBRE', 'DECEMBRE', 'DUDIT', 'MILLE', 'CENTS', 'VINGT',
}


def normaliser_pour_matching(texte):
    """
    Normalise un texte pour la comparaison fuzzy :
    - Suppression des diacritiques (é→e, è→e, â→a ...)
    - Tirets, apostrophes → espaces
    - Majuscules
    - Suppression de toute ponctuation résiduelle
    - Collapsing des espaces multiples
    """
    if not texte:
        return ""
    # Décomposition Unicode (NFD) puis suppression des caractères combinants (diacritiques)
    nfkd = unicodedata.normalize('NFKD', str(texte))
    sans_accents = ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    # Prétraitement spécifique des abréviations avec ponctuation (C.N.E. -> CNE)
    sans_accents = re.sub(r'\bC\.?N\.?E\.?\b', 'COMMUNE', sans_accents, flags=re.IGNORECASE)
    sans_accents = re.sub(r'\bST\.?\b', 'SAINT', sans_accents, flags=re.IGNORECASE)
    sans_accents = re.sub(r'\bSTE\.?\b', 'SAINTE', sans_accents, flags=re.IGNORECASE)

    # Tirets et apostrophes → espace
    sans_accents = re.sub(r"[-''`]", ' ', sans_accents)
    # Majuscules
    majuscules = sans_accents.upper()
    # Suppression ponctuation (on garde lettres, chiffres, espaces)
    propre = re.sub(r'[^A-Z0-9 ]', ' ', majuscules)
    # Collapse espaces multiples
    propre = ' '.join(propre.split())
    
    # Remplacement final des mots isolés
    mots = propre.split()
    for i in range(len(mots)):
        if mots[i] == 'ST':
            mots[i] = 'SAINT'
        elif mots[i] == 'STE':
            mots[i] = 'SAINTE'
        elif mots[i] == 'CNE':
            mots[i] = 'COMMUNE'
            
    return ' '.join(mots)


def load_commune_db(ardeche_json='ardeche.json', villes_txt='villes_07_26.txt'):
    """
    Charge la base de données des communes Ardèche depuis ardeche.json (source principale
    avec codes INSEE et noms officiels) et ajoute les communes supplémentaires de
    villes_07_26.txt (Drôme/Ardèche) comme fallback.

    Retourne une liste de dicts :
        [{"officiel": "Saint-Jean-Chambre", "normalise": "SAINT JEAN CHAMBRE", "code": "07244"}, ...]
    """
    commune_db = []
    noms_deja_presents = set()

    # --- Source 1 : ardeche.json (noms officiels accentués + codes INSEE) ---
    if os.path.exists(ardeche_json):
        with open(ardeche_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            nom_officiel = entry.get('nom', '').strip()
            code = entry.get('code', '')
            if nom_officiel:
                normalise = normaliser_pour_matching(nom_officiel)
                commune_db.append({
                    'officiel': nom_officiel,
                    'normalise': normalise,
                    'code': code
                })
                noms_deja_presents.add(normalise)
        print(f"[CommuneDB] {len(commune_db)} communes chargées depuis '{ardeche_json}'.")
    else:
        print(f"[CommuneDB] ATTENTION : '{ardeche_json}' introuvable.")

    # --- Source 2 : villes_07_26.txt (complément Drôme + variantes) ---
    # DÉSACTIVÉ : Ce fichier a été généré via OCR et contient des erreurs ("Paddys", "Jomiequer", etc.)
    # qui parasitent le fuzzy matching. On ne garde que la base INSEE officielle 100% propre.
    print(f"[CommuneDB] Fallback sur '{villes_txt}' désactivé (trop de bruit OCR).")
    
    print(f"[CommuneDB] Total : {len(commune_db)} communes référencées.")
    return commune_db


def load_commune_db_nationale(path_json: str = 'communes_france.json') -> list:
    """
    Charge une base de communes nationales (France entière).
    Si le fichier n'existe pas, tente de le télécharger depuis l'API Geo.api.gouv.fr.
    """
    import urllib.request

    if not os.path.exists(path_json):
        url = "https://geo.api.gouv.fr/communes?fields=nom,code&format=json"
        print(f"[CommuneDB] Téléchargement base nationale depuis {url} ...")
        try:
            urllib.request.urlretrieve(url, path_json)
            print(f"[CommuneDB] Base nationale sauvegardée : {path_json}")
        except Exception as e:
            print(f"[CommuneDB] Échec téléchargement base nationale : {e}")
            return []

    try:
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        db = []
        for entry in data:
            nom = entry.get('nom', '').strip()
            code = entry.get('code', '')
            if nom:
                db.append({
                    'officiel': nom,
                    'normalise': normaliser_pour_matching(nom),
                    'code': code,
                })
        print(f"[CommuneDB] Base nationale : {len(db)} communes chargées.")
        return db
    except Exception as e:
        print(f"[CommuneDB] Erreur lecture base nationale : {e}")
        return []


# Compatibilité : alias pour l'ancien code qui appelle load_villes_dictionary()
# Retourne une simple liste de noms normalisés (pour correct_ocr_with_dict)
def load_villes_dictionary(filepath='villes_07_26.txt'):
    """Compatibilité ascendante — retourne la liste des noms normalisés."""
    db = load_commune_db()
    return [e['normalise'] for e in db]


# ============================================================
# CARTE D'ABRÉVIATIONS AUTOMATIQUE
# ============================================================

# Articles / prépositions ignorés lors de la génération des formes courtes
_ARTICLES_GEO = frozenset({
    'LA', 'LE', 'LES', 'DE', 'DU', 'DES', 'EN', 'SUR', 'SOUS',
    'ET', 'L', 'D', 'AU', 'AUX', 'PAR', 'LES', 'ST', 'STE'
})

# Variantes saint / sainte → forme courte
_SAINT_MAP = {'SAINT': 'ST', 'SAINTE': 'STE'}

# Variable globale initialisée au démarrage
_commune_abbrev_map = {}   # { forme_normalisée_courte: nom_officiel }


def _generer_formes_abregees(nom_officiel):
    """
    Génère toutes les formes abrégées plausibles pour un nom de commune officiel.

    Exemples :
      'Alba-la-Romaine'        → ['ALBA', 'ALBA ROMAINE']
      'La Chapelle-sous-Aubenas' → ['CHAPELLE', 'CHAPELLE AUBENAS']
      'Saint-Martin-de-Valamas' → ['ST MARTIN', 'MARTIN', 'SAINT MARTIN']
      'Vals-les-Bains'          → ['VALS', 'VALS BAINS']
    """
    norm = normaliser_pour_matching(nom_officiel)   # ex : 'ALBA LA ROMAINE'
    mots = norm.split()

    formes = set()

    # Remplacer SAINT/SAINTE par ST/STE
    mots_st = [_SAINT_MAP.get(m, m) for m in mots]

    # Mots SANS articles ni prépositions
    mots_sig = [m for m in mots_st if m not in _ARTICLES_GEO and len(m) >= 3]

    if not mots_sig:
        return formes

    # 1. Premier mot significatif (≥ 4 lettres pour éviter 'AN', 'ES'...)
    if len(mots_sig[0]) >= 4:
        formes.add(mots_sig[0])

    # 2. Deux premiers mots significatifs
    if len(mots_sig) >= 2:
        formes.add(f"{mots_sig[0]} {mots_sig[1]}")

    # 3. Forme avec Saint en abrégé + premier mot
    if mots_sig[0] in ('ST', 'STE') and len(mots_sig) >= 2:
        formes.add(f"{mots_sig[0]} {mots_sig[1]}")
        # Aussi le nom seul sans le saint
        if len(mots_sig[1]) >= 4:
            formes.add(mots_sig[1])

    # 4. Variante avec la forme longue « SAINT » (si on a mis ST)
    formes_longues = []
    for f in formes:
        f_long = f.replace('ST ', 'SAINT ').replace('STE ', 'SAINTE ')
        if f_long != f:
            formes_longues.append(f_long)
    formes.update(formes_longues)

    return formes


def build_abbreviation_map(commune_db):
    """
    Construit automatiquement un dictionnaire d'abréviations depuis la base de communes.

    Logique :
      - Pour chaque commune, génère 3-5 formes abrégées via `_generer_formes_abregees`.
      - Si une forme abrégée correspond à UNE SEULE commune → entrée dans la carte.
      - Si une forme abrégée correspond à PLUSIEURS communes → ignorée (ambiguë).

    Retourne :
        dict { forme_normalisée_courte (str) → nom_officiel (str) }
    """
    # Étape 1 : collecter toutes les formes → liste de noms officiels
    candidats = defaultdict(list)  # forme → [officiel1, officiel2, ...]

    for entry in commune_db:
        officiel = entry['officiel']
        # Forme complète normalisée (correspondance exacte insensible à la casse)
        candidats[entry['normalise']].append(officiel)
        # Formes abrégées
        for forme in _generer_formes_abregees(officiel):
            candidats[forme].append(officiel)

    # Étape 2 : ne garder que les formes non ambiguës
    abbrev_map: Dict[str, str] = {}
    n_ambigus: int = 0
    for forme, officiels in candidats.items():
        uniques = list(dict.fromkeys(officiels))  # dédoublonnage, ordre conservé
        if len(uniques) == 1:
            abbrev_map[forme] = uniques[0]
        else:
            n_ambigus += 1  # type: ignore

    print(f"  [AbbrevMap] {len(abbrev_map)} formes d'abréviations uniques générées "
          f"({n_ambigus} formes ambiguës ignorées).")

    return abbrev_map


try:
    from transformers import LogitsProcessor, LogitsProcessorList  # type: ignore
    _logits_processor_available = True
except ImportError:
    _logits_processor_available = False


class CommuneConstrainedLogitsProcessor(LogitsProcessor if _logits_processor_available else object):
    """
    Force le décodeur TrOCR à ne générer QUE des noms de communes valides.

    Mécanisme :
      - On pré-tokenise tous les noms de communes.
      - On construit un TRIE de séquences de tokens.
      - À chaque pas de décodage, on met les logits à -inf pour tous les tokens
        qui ne font PAS partie d'un préfixe valide dans le trie.
      - Résultat : le beam search est contraint à terminer sur un nom de commune.
    """

    def __init__(self, commune_token_seqs, eos_token_id, decoder_start_token_id):
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        # Construire le trie
        self.trie: Dict[int, Any] = {}
        for seq in commune_token_seqs:
            node = self.trie
            for tok in seq:
                if tok not in node:
                    node[tok] = {}  # type: ignore
                node = node[tok]
            # EOS valide dès que la séquence est complète
            node[eos_token_id] = {}  # type: ignore

    def _get_valid_next_tokens(self, generated_so_far):
        """Parcourt le trie et retourne les tokens autorisés après la séquence générée."""
        node = self.trie
        for tok in generated_so_far:
            if tok == self.eos_token_id:
                return {self.eos_token_id}
            if tok == self.decoder_start_token_id:
                continue
            if tok in node:
                node = node[tok]
            else:
                # Hors trie → on autorise uniquement EOS pour terminer proprement
                return {self.eos_token_id}
        valid = set(node.keys())
        # EOS est toujours autorisé si au moins un chemin est complet
        return valid if valid else {self.eos_token_id}

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        for i in range(input_ids.shape[0]):
            gen_part = [
                tok for tok in input_ids[i].tolist()
                if tok != self.decoder_start_token_id
            ]
            valid_tokens = self._get_valid_next_tokens(gen_part)
            # Mettre -inf pour tous les tokens non autorisés
            mask = scores[i].clone().fill_(float('-inf'))  # type: ignore
            for tok_id in valid_tokens:
                if 0 <= tok_id < scores.shape[-1]:
                    mask[tok_id] = scores[i][tok_id]  # type: ignore
            scores[i] = mask  # type: ignore
        return scores


def build_commune_logits_processor(commune_db, ocr_processor):
    """
    Pré-tokenise tous les noms de communes avec le tokenizer TrOCR
    et retourne un CommuneConstrainedLogitsProcessor prêt à l'emploi.
    Retourne None si le tokenizer n'est pas disponible ou échoue.
    """
    if not _logits_processor_available or ocr_processor is None:
        return None
    try:
        tokenizer = ocr_processor.tokenizer
        eos_id   = tokenizer.eos_token_id
        start_id = tokenizer.bos_token_id or eos_id

        commune_token_seqs = []
        for entry in commune_db:
            nom = entry.get('officiel', '')
            if not nom:
                continue
            toks = tokenizer.encode(nom, add_special_tokens=False)
            if toks:
                commune_token_seqs.append(toks)
            # Aussi ajouter des variantes courantes (St → Saint, etc.)
            variantes = [
                nom.replace('Saint-', 'St-').replace('Sainte-', 'Ste-'),
                nom.upper(),
            ]
            for v in variantes:
                tv = tokenizer.encode(v, add_special_tokens=False)
                if tv:
                    commune_token_seqs.append(tv)

        if not commune_token_seqs:
            return None

        print(f"  [ConstrainedDecoding] Trie construit avec {len(commune_token_seqs)} séquences de communes")
        return CommuneConstrainedLogitsProcessor(commune_token_seqs, eos_id, start_id)
    except Exception as e:
        print(f"  [ConstrainedDecoding] Impossible de construire le LogitsProcessor: {e}")
        return None


# ============================================================
# LOGIQUE DE SIMILARITÉ VISUELLE (Weighted Levenshtein)
# ============================================================

_VISUAL_GROUPS = [
    set("IL1J"),      # Bâtons verticaux
    set("NMUVW"),     # Jambages cursifs
    set("O0DQG"),     # Formes rondes
    set("S58"),       # Boucles de S
    set("FT7"),       # Croix / barres horizontales
    set("B6H"),       # Jambages hauts bouclés
    set("PRK"),       # P/R/K souvent confondus en cursive
    set("AOU"),       # Voyelles rondes
    set("ECG"),       # C/E/G ouverts
    set("VY"),        # Descentes
    set("I1EYL")      # Variantes de bâtons avec empattements
]

def visual_similarity(s1, s2):
    """
    [Conservé pour compatibilité] Appelle _char_alignment_score.
    """
    return _char_alignment_score(s1, s2)


def _char_alignment_score(ocr_str: str, commune_str: str) -> float:
    """
    Score d'alignement LOCAL lettre par lettre (algorithme Smith-Waterman).

    Principe : on cherche le meilleur sous-alignement de ocr_str dans commune_str,
    sans punit les abréviations. Chaque caractère contribue indépendamment.

    Scores :
      +2.0  caractères identiques
      +1.5  caractères visuellement proches (même groupe : n/u, l/i, o/0 ...)
      -0.5  substitution sans lien visuel
      -1.0  gap (insertion / suppression)

    Normalisation sur len(ocr_str) → une abréviation parfaite = 100 %.
    Exemples :
      'AUBE'   vs 'AUBENAS'      → ~100 %  (préfixe parfait)
      'VALS'   vs 'VALS LES BAINS' → ~100 %  (préfixe parfait)
      'RUOM'   vs 'RUOMS'        → ~90 %   (un manque)
      'AUBE'   vs 'PRIVAS'       → très bas
    """
    n, m = len(ocr_str), len(commune_str)
    if n == 0 or m == 0:
        return 0.0

    MATCH        =  2.0
    VISUAL_MATCH =  1.5
    MISMATCH     = -0.5
    GAP          = -1.0

    # Grille Smith-Waterman (initialisation à 0 = réinitialisation libre)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    max_score = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c1, c2 = ocr_str[i-1], commune_str[j-1]
            if c1 == c2:
                sub = MATCH
            else:
                is_vis = any(c1 in g and c2 in g for g in _VISUAL_GROUPS)
                sub = VISUAL_MATCH if is_vis else MISMATCH
            dp[i][j] = max(
                0.0,
                dp[i-1][j-1] + sub,
                dp[i-1][j]   + GAP,
                dp[i][j-1]   + GAP,
            )
            if dp[i][j] > max_score:
                max_score = dp[i][j]

    # Normalisation sur le score idéal = MATCH * len(ocr_str)
    ideal = MATCH * n
    return min(100.0, (max_score / ideal) * 100.0) if ideal > 0 else 0.0

# Variable globale : on le construit UNE SEULE FOIS en début de traitement
_commune_logits_processor = None


# ============================================================
# FONCTIONS UTILITAIRES DE QUALITÉ OCR (Modèles statistiques)
# Définies ICI, avant match_commune_multi_hypotheses qui les utilise.
# Aucune liste noire — la confiance émerge des propriétés mathématiques du texte.
# ============================================================

def _ocr_informativeness(text: str) -> float:
    """
    Mesure le contenu informationnel d'un texte OCR via l'entropie de Shannon.
    Retourne [0.0, 1.0] : 0 = pas de lettre, 1 = texte riche et varié.
    """
    import math
    if not text or len(text.strip()) == 0 or text.strip() == '[VIDE]':
        return 0.0
    alpha_chars = [c.lower() for c in text if c.isalpha()]
    n_alpha = len(alpha_chars)
    if n_alpha == 0:
        return 0.0
    len_score = min(1.0, n_alpha / 5.0)
    freq: dict = {}
    for c in alpha_chars:
        freq[c] = freq.get(c, 0) + 1
    total = float(n_alpha)
    entropy = -sum((f / total) * math.log2(f / total) for f in freq.values())
    n_distinct = len(freq)
    max_entropy = math.log2(n_distinct) if n_distinct > 1 else 0.0
    entropy_score = (entropy / max_entropy) if max_entropy > 0.0 else 0.0
    return float(0.40 * len_score + 0.60 * entropy_score)


def _length_coherence_score(ocr_text: str, commune_norm: str) -> float:
    """
    Cohérence de longueur OCR↔commune via distribution log-normale.
    Le TrOCR lit ~77% des caractères réels (mu=log(1.3), sigma=0.65).
    Retourne [0.05, 1.0].
    """
    import math
    n_ocr = sum(1 for c in ocr_text if c.isalpha())
    n_comm = sum(1 for c in commune_norm if c.isalpha())
    if n_ocr == 0:
        return 0.08
    if n_comm == 0:
        return 0.5
    ratio = n_comm / float(n_ocr)
    mu_log, sigma_log = math.log(1.3), 0.65
    log_ratio = math.log(ratio) if ratio > 0 else -10.0
    z = (log_ratio - mu_log) / sigma_log
    return max(0.05, math.exp(-0.5 * z * z))


def _apply_ocr_quality_modifier(raw_score: float, ocr_text: str, commune_entry: dict) -> float:
    """
    Applique le modificateur statistique au score brut.
    score_final = score_brut × informativité × cohérence_longueur
    """
    info = _ocr_informativeness(ocr_text)
    length_coh = _length_coherence_score(ocr_text, commune_entry.get('normalise', ''))
    modifier = info * length_coh
    modified = raw_score * modifier
    if modifier < 0.80:
        print(f"    [QualityMod] {raw_score:.0f}% × info={info:.3f} × len={length_coh:.3f} → {modified:.1f}%")
    return modified


def match_commune_ardeche(texte_ocr: str, commune_db: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compatibilité : appelle match_commune_multi_hypotheses avec une seule hypothèse.
    """
    return match_commune_multi_hypotheses([(texte_ocr, 1.0)], commune_db)


def match_commune_multi_hypotheses(hypotheses: List[Tuple[str, float]], commune_db: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Matching de commune sur plusieurs hypothèses OCR pondérées.
    Utilise une approche en 3 étapes :
    1. Matching exact via abréviations.
    2. Filtrage global avec rapidfuzz (top 15).
    3. Ré-évaluation fine via similarité visuelle (Weighted Levenshtein).
    """
    if not commune_db or process is None:
        return {'officiel': 'Non identifiée', 'code': '', 'score': 0, 'methode': 'erreur',
                'brut': hypotheses[0][0] if hypotheses else '', 'hypotheses_ocr': ''}

    noms_normalises = [e['normalise'] for e in commune_db]
    brut_principal  = hypotheses[0][0] if hypotheses else ''

    # 1. NIVEAU 1 : Carte d'abréviations
    for texte_ocr, beam_prob in hypotheses:
        texte_ocr_clean = texte_ocr
        for pfix in ("[COL_COMMUNE] -> ", "[COL_COMMUNE]", "-> "):
            if texte_ocr_clean.startswith(pfix):
                texte_ocr_clean = texte_ocr_clean[len(pfix):].strip()
        texte_norm = normaliser_pour_matching(texte_ocr_clean)
        
        if texte_norm in _commune_abbrev_map:
            nom_officiel = _commune_abbrev_map[texte_norm]
            idx = next((i for i, e in enumerate(commune_db) if e['officiel'] == nom_officiel), 0)
            return {
                'officiel': nom_officiel, 'code': commune_db[idx]['code'],
                'score': 100, 'methode': 'abreviation', 'brut': texte_ocr,
                'hypotheses_ocr': f"abbrev:'{texte_ocr}'->'{nom_officiel}'"
            }

    # 2. FILTRES DE BRUIT GÉNÉRAUX
    _TEXTES_NON_COMMUNE = {
        'COMMUNE', 'COMMUNES', 'COMMUNE DE', 'LA COMMUNE', 'NOM DE LA COMMUNE', 
        'DESIGNATION', 'DONNEUR D ORDRE', 'DATE', 'DOSSIER', 'PRIX', 'FACTURATION',
        'LOTISSEMENT', 'CHEMIN', 'PARCELLE', 'DIVISION', 'PLAN', 'TOPO', 'BORNAGE'
    }

    best_score, best_idx, best_brut = 0, 0, brut_principal
    best_methode = 'fuzzy'
    hyp_log_parts = []

    for texte_ocr, beam_prob in hypotheses:
        texte_ocr_clean = texte_ocr
        for pfix in ("[COL_COMMUNE] -> ", "[COL_COMMUNE]", "-> "):
            if texte_ocr_clean.startswith(pfix):
                texte_ocr_clean = texte_ocr_clean[len(pfix):].strip()
        texte_norm = normaliser_pour_matching(texte_ocr_clean)
        
        if not texte_norm or len(texte_norm) < 2 or texte_norm in _TEXTES_NON_COMMUNE:
            continue
        if sum(c.isdigit() for c in texte_norm) >= len(texte_norm) * 0.6:
            continue

        # ÉTAPE A : Filtrage Grossier (Rapidfuzz Top 15) pour la rapidité
        results = process.extract(texte_norm, noms_normalises, scorer=fuzz.WRatio, limit=15) # type: ignore
        # ÉTAPE A2 : token_sort_ratio en complément (utile pour les communes multi-mots réordonnés)
        results_tsr = process.extract(texte_norm, noms_normalises, scorer=fuzz.token_sort_ratio, limit=10) # type: ignore
        # Fusion des deux listes (dédoublonnage par index, garde le meilleur score)
        seen_idx = {}
        for match_str, sc, mi in results:
            seen_idx[mi] = (match_str, sc, mi)
        for match_str, sc, mi in results_tsr:
            if mi not in seen_idx or sc > seen_idx[mi][1]:
                seen_idx[mi] = (match_str, sc, mi)
        results = list(seen_idx.values())
        
        hyp_best_cand_score = 0
        hyp_best_cand_idx = 0
        hyp_best_cand_methode = 'fuzzy'

        for match_str, fuzz_score, match_idx in results:
            # ÉTAPE B : Alignement Smith-Waterman lettre par lettre
            # (remplacement du Levenshtein global : robuste aux abréviations)
            sw_score = _char_alignment_score(texte_norm, match_str)

            # ÉTAPE C : Bonus préfixe exact (Juvin → Juvinas) et noms courts
            methode_cand = 'fuzzy'
            if len(texte_norm) >= 4 and match_str.startswith(texte_norm):
                sw_score = min(100.0, sw_score + 10.0)
                methode_cand = 'prefixe'
            if len(texte_norm) <= 5 and texte_norm == match_str:
                sw_score = min(100.0, sw_score + 20.0)
                methode_cand = 'prefixe'

            # NOTE : Pas de pénalité de longueur — Smith-Waterman la gère nativement
            vis_score = sw_score

            if vis_score > hyp_best_cand_score:
                hyp_best_cand_score = vis_score
                hyp_best_cand_idx = match_idx
                hyp_best_cand_methode = methode_cand

        # ÉTAPE E : Fusion Probabiliste avec BeamProb
        # FinalScore = 70% ressemblance visuelle + 30% confiance modèle TrOCR
        score_brut = (hyp_best_cand_score * 0.7) + (float(beam_prob) * 30.0)

        # ── Modificateur de qualité statistique (informativité × cohérence longueur) ──
        # Appelé ici directement pour chaque hypothèse, avant comparaison avec best_score.
        # Cela évite que 'veEt' → 'Colombier-le-Vieux' survive avec un score élevé.
        best_entry_for_mod = commune_db[hyp_best_cand_idx] if hyp_best_cand_idx < len(commune_db) else {}
        score_final = _apply_ocr_quality_modifier(score_brut, texte_norm, best_entry_for_mod)

        hyp_log_parts.append(f"'{texte_ocr}'({beam_prob:.2f})->{noms_normalises[hyp_best_cand_idx]}({score_final:.0f}% {hyp_best_cand_methode})")

        if score_final > best_score:
            best_score, best_idx, best_brut, best_methode = score_final, hyp_best_cand_idx, texte_ocr, hyp_best_cand_methode

    if best_score < 45: # Seuil minimum de santé
        return {'officiel': 'Non identifiée', 'code': '', 'score': 0, 'methode': 'aucun', 'brut': brut_principal, 'hypotheses_ocr': " | ".join(hyp_log_parts)}

    meilleure = commune_db[best_idx]
    return {'officiel': meilleure['officiel'], 'code': meilleure['code'], 'score': int(best_score), 'methode': best_methode, 'brut': best_brut, 'hypotheses_ocr': " | ".join(hyp_log_parts)}



def correct_ocr_with_dict(texte: str, dictionnaire: List[str], seuil: int = 88) -> str:
    """Correction générique d'un texte OCR contre un dictionnaire de noms normalisés.
    Utilisé pour les corrections non-communes (noms de lieux Drôme, etc.)."""
    if not dictionnaire or process is None or len(texte.strip()) < 3:
        return texte

    if len(texte) > 30:
        return texte

    t_clean = normaliser_pour_matching(texte)

    chiffres = sum(c.isdigit() for c in t_clean)
    if chiffres > 2:
        return texte

    if len(t_clean) < 3:
        return texte

    result = process.extractOne(t_clean, dictionnaire, scorer=fuzz.WRatio)  # type: ignore
    if result:
        meilleur_match, score, _ = result
        if score >= seuil:
            print(f"      [Correction Dict] '{texte}' -> '{meilleur_match}' (score: {score:.1f})")
            return meilleur_match

    return texte

def correct_cadastral_rules(texte):
    """Applique des règles métiers (Regex) pour corriger les erreurs OCR fréquentes sur les plans cadastraux."""
    if not texte:
        return texte
        
    texte_corrige = texte

    # 1. Règle Échelle (Echelle 1/1000, Ech 1/500, etc.)
    # L'OCR lit parfois "1f1000", "l/1000", "1/1O00"
    def repl_echelle(m):
        prefix = m.group(1)
        val = m.group(2)
        val = val.replace('l', '1').replace('I', '1').replace('f', '/').replace('t', '/').replace('O', '0').replace('o', '0')
        return prefix + val
    
    texte_corrige = re.sub(r'((?:[Ee]chelle|[Ee]ch\.?)\s*)([A-Za-z0-9/]+)\b', repl_echelle, texte_corrige)

    # 2. Règle Section (Section A, Sect AB, etc.)
    # L'OCR lit "Section 0" au lieu de "O", "1" au lieu de "I", "8" au lieu de "B"
    def repl_section(m):
        prefix = m.group(1)
        val = m.group(2)
        val = val.replace('0', 'O').replace('1', 'I').replace('8', 'B').replace('5', 'S')
        return prefix + val

    texte_corrige = re.sub(r'((?:[Ss]ection|[Ss]ect\.?)\s*)([0-9A-Za-z]{1,2})\b', repl_section, texte_corrige)

    # 3. Règle Numéros de Parcelles (Numéros isolés de 1 à 4 chiffres)
    def repl_numero(m):
        val = m.group(0)
        # Uniquement si le bloc contient au moins un chiffre et ne contient que des caractères ambigus
        # On limite aux blocs de 1 à 5 caractères pour éviter de casser des mots longs
        if 1 <= len(val) <= 5 and any(c.isdigit() for c in val) and re.match(r'^[0-9OolISsB]+$', val):
            val_corrige = val.replace('O', '0').replace('o', '0').replace('l', '1').replace('I', '1').replace('S', '5').replace('s', '5').replace('B', '8')
            if val_corrige.isdigit():
                return val_corrige
        return val

    # Applique à tous les blocs alphanumériques (y compris avec accents pour ne pas couper les mots en deux)
    texte_corrige = re.sub(r'\b[A-Za-zÀ-ÿ0-9]+\b', repl_numero, texte_corrige)

    # 4. Règle Dossier (Ex: 2024-00123)
    # Souvent OCRisé avec des 'O' au lieu de '0'
    def repl_dossier(m):
        val = m.group(0)
        return val.replace('O', '0').replace('o', '0')
    
    texte_corrige = re.sub(r'\b\d{4}[- ]\d+[A-ZÀ-Ÿ]?\b', repl_dossier, texte_corrige)

    # 5. Règle Mots Clés Fréquents (Département, Géomètre, etc.)
    # Correction des "é" souvent pris pour des "&" ou d'autres caractères par Tesseract
    texte_corrige = re.sub(r'(?i)\bd[&cCeE]partement\b', 'Département', texte_corrige)
    texte_corrige = re.sub(r'(?i)\bg[&cCeE]om[eèéêë&c]tre\b', 'Géomètre', texte_corrige)
    texte_corrige = re.sub(r'(?i)\bpropri[&cCeE]t[&cCeE]?\b', 'Propriété', texte_corrige)
    texte_corrige = re.sub(r'(?i)\bpropri[&cCeE]taire\b', 'Propriétaire', texte_corrige)
    texte_corrige = re.sub(r'(?i)\bc0mmune\b', 'Commune', texte_corrige)
    texte_corrige = re.sub(r'(?i)\bparce[l1I]{1,2}es?\b', 'Parcelle', texte_corrige)
    texte_corrige = re.sub(r'(?i)\bsecti0n\b', 'Section', texte_corrige)
    
    return texte_corrige

def correct_spelling(texte, spell):
    """Corrige l'orthographe des mots de plus de 4 lettres tout en conservant la casse."""
    if not texte: return texte
    def repl(m):
        mot = m.group(0)
        corr = spell.correction(mot.lower())
        if corr and corr != mot.lower():
            if mot.isupper(): return corr.upper()
            elif mot.istitle(): return corr.capitalize()
            return corr
        return mot
    # On corrige uniquement les assemblages de lettres
    return re.sub(r'\b[A-Za-zÀ-ÿ]{4,}\b', repl, texte)

def run_tesseract_windows(roi_img, custom_config, output_type='text'):
    """Exécute Tesseract (installé sur Windows) depuis WSL pour une image donnée.
    output_type: 'text' pour string simple, 'tsv' pour DataFrame avec bounding boxes."""
    # Sauvegarde temp de l'image
    temp_img_path = os.path.abspath(os.path.join('outputs', 'temp_tess.png'))
    cv2.imwrite(temp_img_path, roi_img)
    
    # Préfixe pour le fichier texte généré par Tesseract
    temp_prefix = os.path.abspath(os.path.join('outputs', 'temp_tess'))
    
    # Convertir les chemins WSL -> Windows pour l'executable Windows
    win_img_path = subprocess.check_output(['wslpath', '-w', temp_img_path]).decode().strip()
    win_prefix = subprocess.check_output(['wslpath', '-w', temp_prefix]).decode().strip()
    
    tess_exe = "/mnt/c/Users/Topo_4/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
    tessdata_dir = r'C:\Users\Topo_4\AppData\Local\Programs\Tesseract-OCR\tessdata'
    
    # Construction de la commande
    # --tessdata-dir est passé directement en argument CLI car TESSDATA_PREFIX
    # ne se propage pas correctement depuis WSL vers un .exe Windows
    cmd = [tess_exe, win_img_path, win_prefix, '--tessdata-dir', tessdata_dir] + custom_config.split()
    if output_type == 'tsv':
        cmd.append('tsv')
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if output_type == 'tsv':
        res = pd.DataFrame()
        tsv_file = temp_prefix + '.tsv'
        if os.path.exists(tsv_file):
            try:
                res = pd.read_csv(tsv_file, sep='\t', quoting=3)
            except Exception:
                pass
            os.remove(tsv_file)
    else:
        res = ""
        txt_file = temp_prefix + '.txt'
        if os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                res = f.read().strip()
            os.remove(txt_file)
        
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
        
    return res

def setup_directories():
    """Crée les dossiers d'entrée et de sortie s'ils n'existent pas."""
    os.makedirs('inputs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

def load_models(nom_modele="agomberto/trocr-large-handwritten-fr"):
    """Charge les modèles YOLO, EasyOCR et TrOCR."""
    print("Chargement du modèle YOLOv8 pour la détection...")
    # Utilisation de yolov8n.pt pour détecter les zones. 
    # Note : Le modèle de base détecte des objets généraux. Pour un usage réel, 
    # il faudrait fine-tuner YOLO pour détecter spécifiquement "texte imprimé" et "texte manuscrit".
    yolo_model = YOLO('yolov8n.pt')

    print("Chargement d'EasyOCR pour le texte imprimé...")
    # Initialisation de EasyOCR (en français et anglais), utilise le GPU si disponible
    easyocr_reader = easyocr.Reader(['fr', 'en'], gpu=torch.cuda.is_available())

    print("Chargement de TrOCR (version Française affinée agomberto)...")
    print(f"Chargement de TrOCR (modèle : {nom_modele})...")
    processor = TrOCRProcessor.from_pretrained(nom_modele)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(nom_modele)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trocr_model.to(device)
    
    print("Chargement du modèle de reconnaissance PyLaia (Kraken) pour Indice de Confiance...")
    pylaia_model_path = "catmus_large.mlmodel"
    if os.path.exists(pylaia_model_path):
        pylaia_model = kraken_models.load_any(pylaia_model_path)
    else:
        print("ATTENTION: Modèle PyLaia catmus_large.mlmodel introuvable. Score de confiance désactivé.")
        pylaia_model = None

    print("Chargement du Correcteur Orthographique...")
    spell = SpellChecker(language='fr')
    # Ajout du vocabulaire technique métier
    mots_metier = ['contenance', 'lieudit', 'parcelle', 'section', 'echelle', 'dmpc', 'geofoncier', 'superficie', 'cadastre', 'limite', 'borne', 'mur', 'mitoyen']
    spell.word_frequency.load_words(mots_metier)

    # --- VLM POST-CORRECTION (Vision Language Model) ---
    # Qwen2-VL-2B-Instruct : modèle multimodal capable d'analyser une image ET du texte.
    # Il joue le rôle de paléographe : il voit la cellule manuscrite et raisonne sur la commune.
    # Retourné sous la forme d'un tuple (vlm_model, vlm_processor) pour rester compatible
    # avec le reste du pipeline sans changer les signatures.
    try:
        print("Chargement du VLM paléographe (Qwen2-VL-2B-Instruct)...")
        import sys
        import subprocess
        try:
            import accelerate
            import qwen_vl_utils
        except ImportError:
            print("[INSTALL] 'accelerate' ou 'qwen-vl-utils' manquant dans l'environnement actuel. Installation automatique en cours...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate", "qwen-vl-utils", "torchvision"])
            
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  # type: ignore
        vlm_model_id = "Qwen/Qwen2-VL-2B-Instruct"
        vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
        
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                vlm_model_id,
                device_map="auto",
                quantization_config=quant_config
            )
            print(" -> VLM Qwen2-VL-2B chargé en mode 8-bit (économie de RAM).")
        except ImportError:
            vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                vlm_model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print(" -> VLM Qwen2-VL-2B chargé de base (FP16). (Installez bitsandbytes pour 8-bit).")
            
        vlm_model.eval()
        llm_pipeline = (vlm_model, vlm_processor)  # tuple (model, processor)
    except Exception as e:
        print(f"ATTENTION: Impossible de charger le VLM Qwen2-VL ({e}). L'arbitrage visuel sera désactivé.")
        llm_pipeline = None

    return yolo_model, easyocr_reader, processor, trocr_model, device, spell, pylaia_model, llm_pipeline

def read_document(file_path):
    """Lit un PDF ou une image et retourne une liste d'images (tableaux NumPy au format BGR)."""
    images = []
    if file_path.lower().endswith('.pdf'):
        # On lit le PDF avec PyMuPDF
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # CRITIQUE : on rend le PDF en 3x pour augmenter la résolution effective.
            # Sans ça, la résolution est trop basse (~72 DPI) et Tesseract ne peut pas
            # lire le texte imprimé correctement (confiance 0%). Avec 3x, on obtient
            # ~216 DPI et Tesseract lit à 90%+ de confiance.
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat)
            # Convertit le pixmap en tableau numpy lisible par OpenCV
            img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            images.append(img)
    else:
        # On lit l'image normalement avec OpenCV
        img = cv2.imread(file_path)
        if img is not None:
            images.append(img)
    return images

def deskew_image(image):
    """Calculer l'angle d'inclinaison du texte et redresser l'image horizontale."""
    # Convertir en NG et inverser (texte blanc sur fond noir)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Trouver les coordonnées de tous les pixels > 0 (le texte)
    coords = np.column_stack(np.where(gray > 0))
    if len(coords) < 10: # Pas assez de texte pour calculer un angle
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    
    # Correction de l'angle renvoyé par OpenCV
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # S'il n'y a quasi pas d'angle, on ne touche à rien pour éviter le flou de rotation
    if abs(angle) < 0.5:
        return image
        
    # Rotation de l'image pour la redresser
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_roi_for_ocr(roi, mode='hybrid'):
    """Prétraitement OpenCV — retourne un tuple (variante_principale, roi_color_clean).
    La variante principale est la version la plus lisible pour TrOCR.
    """
    if roi.size == 0:
        return roi, roi

    # 0. Redressement du texte (Deskew)
    roi = deskew_image(roi)

    # 1. Agrandissement massif (Upscaling x3)
    roi = cv2.resize(roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2. Nettoyage du bruit
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # 3. Amélioration du contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0 if mode == 'htr' else 2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(blur)

    # Variante principale : niveaux de gris CLAHE en RGB (TrOCR préfère le gris propre)
    roi_color_clean = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB)

    # Padding
    roi_padded = cv2.copyMakeBorder(roi_color_clean, 20, 20, 20, 20,
                                    cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return roi_padded, roi_color_clean


def _build_preprocessing_variants(roi_brute: np.ndarray) -> List[np.ndarray]:
    """
    Génère 4 variantes de prétraitement différentes d'une même ROI de cellule manuscrite.
    Chaque variante est optimisée pour mettre en évidence un aspect différent du tracé.

    Retourne une liste de tableaux numpy RGB (prêts pour TrOCR via PIL.Image.fromarray).
    """
    variants: List[np.ndarray] = []

    # Upscale commun x3
    roi_up = cv2.resize(roi_brute, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
    if len(roi_up.shape) == 3:
        gray = cv2.cvtColor(roi_up, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_up.copy()

    def _to_rgb_padded(img_gray: np.ndarray) -> np.ndarray:
        """Convertit en RGB et ajoute un padding blanc de 20px."""
        rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        return cv2.copyMakeBorder(rgb, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # ─── Variante 1 : CLAHE léger (contraste lissé, fond gris) ───
    # Idéale pour les ecritures à l'encre noire sur papier jauni
    clahe_soft = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    blur1 = cv2.bilateralFilter(gray, 9, 55, 55)
    v1 = clahe_soft.apply(blur1)
    variants.append(_to_rgb_padded(v1))

    # ─── Variante 2 : CLAHE fort + égalisation d'histogramme ───
    # Maximise le contraste entre l'encre et le fond
    clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
    v2_clahe = clahe_strong.apply(gray)
    v2 = cv2.equalizeHist(v2_clahe)
    variants.append(_to_rgb_padded(v2))

    # ─── Variante 3 : Binarisation adaptative douce (texte sombre sur fond blanc) ───
    # Binarisation sur le CLAHE : élimine les taches de papier tout en gardant les liaisons cursives
    clahe_med = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v3_clahe = clahe_med.apply(cv2.GaussianBlur(gray, (3, 3), 0))
    v3_thresh = cv2.adaptiveThreshold(v3_clahe, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 25, 8)
    # IMPORTANT : TrOCR attend texte SOMBRE sur fond CLAIR
    # adaptiveThreshold THRESH_BINARY conserve le fond blanc et le texte noir
    variants.append(_to_rgb_padded(v3_thresh))

    # ─── Variante 4 : Denoise fort (filtre NLMeans) ───
    # Utile pour les scans très bruités ou les photocopies de faible qualité
    try:
        v4_denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
        clahe_v4 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        v4 = clahe_v4.apply(v4_denoised)
        variants.append(_to_rgb_padded(v4))
    except Exception:
        variants.append(variants[0])  # Fallback sur la variante 1

    return variants

def process_document(file_path, models):
    """Pipeline sans YOLO : Détection EasyOCR + Prétraitement + Double OCR (Imprimé/Manuscrit)"""
    # Note : On ignore yolo_model puisqu'on utilise le détecteur d'EasyOCR
    _, easyocr_reader, processor, trocr_model, device, spell, *_ = models
    
    images = read_document(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    results_data = []

    for idx, img in enumerate(images):
        print(f"\nTraitement de la page/image {idx + 1}...")
        
        # Image pour dessiner les résultats (sans écraser l'originale)
        img_annotated = img.copy()
        
        print("  -> Détection des boîtes de texte avec CRAFT (EasyOCR)...")
        # On utilise readtext avec detail=1 pour récupérer les coordonnées (bbox) sans se soucier du texte lu pour l'instant
        ocr_boxes = easyocr_reader.readtext(img)
        
        page_results = []
        
        for r in ocr_boxes:
            bbox, text_brut, conf_brute = r
            
            # Si EasyOCR a détecté un truc absurde avec une confiance < 15%, on ignore complètement cette boîte (bruit)
            if conf_brute < 0.15:
                continue
                
            # Conversion des 4 points (polygon) en rectangle classique (x1, y1, x2, y2)
            (tl, tr, br, bl) = bbox
            x1 = max(0, int(min(tl[0], bl[0])))
            y1 = max(0, int(min(tl[1], tr[1])))
            x2 = min(img.shape[1], int(max(tr[0], br[0])))
            y2 = min(img.shape[0], int(max(bl[1], br[1])))
            
            # Extraction de la petite région (ROI - Region Of Interest)
            roi = img[y1:y2, x1:x2]
            
            if roi.size == 0 or roi.shape[0] < 3 or roi.shape[1] < 3:
                continue

            # --- ETAPE 2 : PRÉTRAITEMENT OPENCV ---
            # On nettoie l'image (lignes, hachures, contraste)
            _, roi_color_clean = preprocess_roi_for_ocr(roi)
            
            # --- ETAPE 3 : LECTURE (OCR MANUSCRIT SEUL) ---
            # Par TrOCR (spécialisé pour texte manuscrit / difficile)
            # TrOCR a besoin d'une image RGB (PIL) propre. roi_color_clean est déjà RGB.
            pil_img = Image.fromarray(roi_color_clean)
            
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
            # Les paramètres (max_new_tokens, no_repeat_ngram_size) aident à stabiliser le modèle
            generated_ids = trocr_model.generate(
                pixel_values, 
                max_new_tokens=20,
                no_repeat_ngram_size=3
            )
            texte_manuscrit = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # --- SAUVEGARDE ET AFFICHAGE ---
            
            # Dessine la boîte
            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Écrit le texte trouvé par TrOCR (en rouge)
            cv2.putText(img_annotated, texte_manuscrit, (x1, max(0, y1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            page_results.append({
                "box": [x1, y1, x2, y2],
                "lecture_trocr": texte_manuscrit,
                "confiance_boite_initiale": float(conf_brute)
            })
        
        # Sauvegarde
        output_img_path = os.path.join('outputs', f"{base_name}_page_{idx+1}_annote.jpg")
        cv2.imwrite(output_img_path, img_annotated)
        results_data.append({"page": idx+1, "detections": page_results})
        
    json_path = os.path.join('outputs', f"{base_name}_resultats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nExtraction terminée pour {file_path}.")
    print(f"-> Résultat visuel : {output_img_path}")
    print(f"-> Résultat texte  : {json_path}")


def extract_kpis_from_layout(results_data, villes_dict=None, commune_db=None):
    """
    Parcourt les résultats spatialement pour extraire les KPIs principaux.
    Stratégie en cascade pour la commune :
      1. Pivot textuel (en-tête "Commune de", etc.)
      2. Zone haute de page (quart supérieur) pour les carnets sans en-tête explicite
      3. Brute-force pondera sur mots/phrases (filtre mots courants FR)
      4. Garantie absolue : match_commune_ardeche() retourne TOUJOURS une commune
    """
    kpis: Dict[str, Any] = {
        "commune": "Inconnu",
        "commune_score": 0,
        "commune_code": "",
        "geometre": "Inconnu",
        "n_dossier": "Inconnu",
        "echelle": "Inconnu",
        "proprietaire": "Inconnu",
        "cadastre_section": "Inconnu",
        "cadastre_parcelle": "Inconnu",
        "ordre_document": "Inconnu"
    }
    
    # Mots-clés pivots (variantes — enrichis pour les carnets manuscrits)
    pivots: Dict[str, List[str]] = {
        "commune": [
            "commune de", "commune :", "commune:", "commue de", "commune du",
            "ville de", "territoire de la commune", "territoire de",
            "departem", "departement", "arrondissement",  # parfois lié à la commune dans les carnets
        ],
        "geometre": ["géomètre expert", "géomètre-expert", "dessiné par", "cabinet", "bureau d'études", "geometre"],
        "n_dossier": ["dossier n°", "dossier :", "n° d'inscription", "affaire n°", "n° de dossier", "référence"],
        "echelle": ["echelle", "échelle", "ech :"],
        "proprietaire": ["propriétaire", "demandeur", "propriété de", "m.", "mme", "sci"],
        "cadastre_section": ["section", "section :", "section cadastrale"],
        "cadastre_parcelle": ["parcelle n°", "parcelle :", "parcelles :", "numéro :"],
        "ordre_document": ["ordre :", "n° d'ordre", "document d'ordre"]
    }

    # Mots à exclure
    exclusions = ["soussigné", "accepté", "certifie", "vu pour", "approuvé", "dument", "le présent", "opérations de", "conformément"]

    # On aplatit toutes les lignes de toutes les pages avec leur info spatiale
    all_lines: List[Dict[str, Any]] = []
    for p_idx, p in enumerate(results_data):
        # On préfère utiliser les détections brutes (colonnes indépendantes) plutôt que les lignes fusionnées
        detections = p.get("raw_detections", p["detections"])
        for det in detections:
            det["page_idx"] = p_idx
            all_lines.append(det)

    for i, line in enumerate(all_lines):
        txt_raw = line["texte"]
        txt_low = txt_raw.lower()
        
        for key, keywords in pivots.items():
            for kw in keywords:
                if kw in txt_low:
                    valeur = ""
                    # 1. On regarde si la valeur est APRES sur la même ligne (via séparateur | ou simple suite)
                    parts = txt_raw.split('|')
                    found_in_same = False
                    for p_idx_part, part in enumerate(parts):
                        if kw in part.lower():
                            if p_idx_part + 1 < len(parts):
                                val_candidate = parts[p_idx_part+1].strip()
                                if len(val_candidate) > 1:
                                    valeur = val_candidate
                                    found_in_same = True
                                    break
                            else:
                                start_idx = part.lower().find(kw) + len(kw)
                                suffix = part[start_idx:].strip().lstrip(':').strip()
                                if len(suffix) > 2:
                                    valeur = suffix
                                    found_in_same = True
                                    break
                    
                    # 2. Recherche Géométrique (En dessous dans la colonne)
                    if not found_in_same:
                        box_pivot = line.get("bbox") or line.get("box")
                        if box_pivot:
                            px_min, py_min, px_max, py_max = box_pivot
                            cx_pivot = (px_min + px_max) / 2
                            
                            best_candidate_text = None
                            min_y_diff = float('inf')
                            
                            for other_line in all_lines:
                                if other_line.get("page_idx") != line.get("page_idx"):
                                    continue
                                
                                other_box = other_line.get("bbox") or other_line.get("box")
                                if not other_box:
                                    continue
                                
                                ox_min, oy_min, ox_max, oy_max = other_box
                                cx_other = (ox_min + ox_max) / 2
                                cy_other = (oy_min + oy_max) / 2
                                
                                tolerance_x = max((px_max - px_min) * 0.8, 50)
                                if cy_other > py_max and abs(cx_other - cx_pivot) < tolerance_x:
                                    y_diff = cy_other - py_max
                                    if y_diff < min_y_diff and y_diff < 300: # Seuil vertical (ex: 300 pixels)
                                        # On vérifie si ce candidat est un autre en-tête pivot
                                        is_pivot = False
                                        for other_kws in pivots.values():
                                            for okw in other_kws:  # type: ignore
                                                if okw in other_line["texte"].lower():
                                                    is_pivot = True; break
                                            if is_pivot: break
                                        
                                        if not is_pivot:
                                            min_y_diff = y_diff
                                            best_candidate_text = other_line["texte"].strip()
                                            
                            if best_candidate_text:
                                valeur = best_candidate_text
                                found_in_same = True
                                
                    # 3. Fallback: Si rien à droite ou géométriquement, on regarde la ligne texte brute en dessous
                    if not found_in_same and i + 1 < len(all_lines):
                        next_line = all_lines[i+1].get("texte", "")  # type: ignore
                        is_pivot = False
                        for other_kws in pivots.values():
                            for okw in other_kws:  # type: ignore
                                if okw in next_line.lower():
                                    is_pivot = True; break
                        if not is_pivot:
                            valeur = next_line.strip()

                    # Nettoyage final
                    if valeur:
                        val_low = valeur.lower()
                        # Exclusion de phrases juridiques ou trop longues
                        if len(valeur) > 100 or any(ex in val_low for ex in exclusions):
                            continue

                        # Cas spécial Commune : on utilise match_commune_ardeche pour une validation qualité
                        if key == "commune" and commune_db:
                            res = match_commune_ardeche(valeur, commune_db)
                            valeur = res['officiel']
                            score_candidat = res['score']
                            # On ne garde que si meilleur que l'actuel
                            if kpis.get("commune") == "Inconnu" or score_candidat > kpis.get("commune_score", 0):
                                kpis["commune"] = valeur
                                kpis["commune_score"] = score_candidat
                                kpis["commune_code"] = res['code']
                        else:
                            # On garde la valeur si elle est plus pertinente que l'actuelle
                            if kpis[key] == "Inconnu" or (len(valeur) > 3 and kpis[key] == "Inconnu"):  # type: ignore
                                kpis[key] = valeur  # type: ignore

    # ================================================================
    # === GARANTIE COMMUNE ARDÈCHE — Extraction en cascade ===
    # Si le pivot textuel n'a pas résolu la commune, on applique
    # 3 stratégies supplémentaires avant la garantie finale.
    # ================================================================
    if commune_db and process:

        commune_courante_score = kpis.get("commune_score", 0)

        # ---- Stratégie 2 : Zone Haute de Page (quart supérieur) ----
        # Dans les vieux carnets, la commune est souvent en en-tête
        # sans mot-clé 'commune de' — on extrait les textes du quart haut.
        print("  -> [CASCADE] Stratégie 2 : Recherche dans la zone haute de page...")
        for line in all_lines:
            if line.get('page_idx', 999) != 0:
                continue  # Seulement première page
            box = line.get('bbox') or line.get('box')
            if not box:
                continue
            _, y1_l, _, y2_l = box
            cy_line = (y1_l + y2_l) / 2.0
            # Hauteur max de la page (approx)
            max_y = max(
                (((l.get('bbox') or l.get('box') or [0, 0, 0, 1000])[3])
                 for l in all_lines if l.get('page_idx', 999) == 0),
                default=1000
            )
            quart = max_y / 4.0
            if cy_line <= quart:
                texte_zone = line['texte'].strip()
                if 3 < len(texte_zone) < 60:
                    res = match_commune_ardeche(texte_zone, commune_db)
                    if res['score'] > commune_courante_score:
                        commune_courante_score  = res['score']
                        kpis["commune"]         = res['officiel']
                        kpis["commune_score"]   = res['score']
                        kpis["commune_code"]    = res['code']
                        print(f"     -> [Zone haute] '{texte_zone}' -> '{res['officiel']}' ({res['score']}%)")

        # ---- Stratégie 3 : Brute-force Pondéré (filtre mots courants) ----
        print("  -> [CASCADE] Stratégie 3 : Brute-force pondéré (filtre mots courants)...")
        for line in all_lines:
            texte_ligne = line["texte"].strip()
            texte_norm_ligne = normaliser_pour_matching(texte_ligne)

            # -- Test ligne entière --
            if 4 <= len(texte_norm_ligne) <= 45 and texte_norm_ligne not in _MOTS_COURANTS_FR:
                res = match_commune_ardeche(texte_ligne, commune_db)
                if res['score'] > commune_courante_score:
                    commune_courante_score = res['score']
                    kpis["commune"]        = res['officiel']
                    kpis["commune_score"]  = res['score']
                    kpis["commune_code"]   = res['code']

            # -- Test mot isolé --
            for mot in texte_ligne.split():
                mot_norm = normaliser_pour_matching(mot)
                if len(mot_norm) < 4 or mot_norm in _MOTS_COURANTS_FR:
                    continue
                res_m = match_commune_ardeche(mot, commune_db)
                if res_m['score'] > commune_courante_score:
                    commune_courante_score = res_m['score']
                    kpis["commune"]        = res_m['officiel']
                    kpis["commune_score"]  = res_m['score']
                    kpis["commune_code"]   = res_m['code']

            # -- Test bigrammes (ex : "St Andéol" → "Saint-Andéol-de-Berg") --
            mots = texte_ligne.split()
            for k in range(len(mots) - 1):
                bigramme = mots[k] + ' ' + mots[k + 1]
                bg_norm = normaliser_pour_matching(bigramme)
                if len(bg_norm) < 5 or bg_norm in _MOTS_COURANTS_FR:
                    continue
                res_b = match_commune_ardeche(bigramme, commune_db)
                if res_b['score'] > commune_courante_score:
                    commune_courante_score = res_b['score']
                    kpis["commune"]        = res_b['officiel']
                    kpis["commune_score"]  = res_b['score']
                    kpis["commune_code"]   = res_b['code']

        # ---- Garantie Absolue (Stratégie 4) ----
        # Même si aucune stratégie n'a trouvé un bon score, on garantit
        # TOUJOURS une commune en sortie (le meilleur match du document).
        if kpis["commune"] == "Inconnu" or not kpis["commune"]:
            texte_total = " ".join([
                l["texte"].strip()
                for l in all_lines
                if 2 < len(l["texte"].strip()) < 60
            ])
            if texte_total:
                res_final = match_commune_ardeche(texte_total, commune_db)
                kpis["commune"]       = res_final['officiel']
                kpis["commune_score"] = res_final['score']
                kpis["commune_code"]  = res_final['code']
                print(f"     -> [Garantie Absolue] Commune forcée : '{res_final['officiel']}' ({res_final['score']}%)")
            else:
                kpis["commune"] = "Commune non identifiée"

        print(f"  -> Commune finale : '{kpis['commune']}' "
              f"(INSEE: {kpis['commune_code']}, confiance: {kpis['commune_score']}%)")

    elif villes_dict and process:
        # Mode compatibilité ascendante (vieux appel sans commune_db)
        best_commune = None
        best_score   = 0
        valeur_actuelle = kpis.get("commune", "Inconnu")
        if valeur_actuelle and valeur_actuelle != "Inconnu":
            match_direct = process.extractOne(valeur_actuelle, villes_dict, scorer=fuzz.WRatio)
            if match_direct:
                best_commune = match_direct[0]
                best_score   = match_direct[1]
        for line in all_lines:
            texte_ligne = line["texte"].strip()
            if 3 < len(texte_ligne) < 45:
                for mot in texte_ligne.split():
                    if len(mot) >= 4:
                        match_res = process.extractOne(mot, villes_dict, scorer=fuzz.WRatio)
                        if match_res and match_res[1] > best_score:
                            best_score   = match_res[1]
                            best_commune = match_res[0]
                match_res_ligne = process.extractOne(texte_ligne, villes_dict, scorer=fuzz.WRatio)
                if match_res_ligne and match_res_ligne[1] > best_score:
                    best_score   = match_res_ligne[1]
                    best_commune = match_res_ligne[0]
        if best_commune:
            print(f"  -> Commune identifiée (compat) : {best_commune} (score: {best_score:.0f}%)")
            kpis["commune"] = best_commune
        else:
            kpis["commune"] = "Commune non identifiée"

    return kpis


def classify_document(results_data: List[Dict[str, Any]], kpis: Optional[Dict[str, Any]] = None) -> str:
    """Fonction identification du type de document utilisant les KPIs et le texte global."""
    texte_global: str = ""
    for page in results_data:
        for det in page["detections"]:
            texte_global += " " + det["texte"].lower()  # type: ignore
            
    mots_cles = {
        "Document d'Arpentage (DMPC)": ["arpentage", "dmpc", "chemise verte", "division", "document de modification", "arpentage"],
        "Plan de Bornage": ["bornage", "reconnaissance", "limites", "amiable", "pv", "procès-verbal", "borné ce jour", "reconnaissance de limites"],
        "Plan Topographique": ["topographique", "etat des lieux", "releve", "courbes de niveau", "levé topographique"],
        "Plan de Division": ["division", "lotissement", "création de lots"]
    }
    
    scores: Dict[str, int] = {k: 0 for k in mots_cles}
    for type_doc, keywords in mots_cles.items():
        for kw in keywords:
            if kw in texte_global:  # type: ignore
                scores[type_doc] += 1  # type: ignore
    
    # Renforcement via KPIs
    if kpis is not None:
        if kpis.get("ordre_document") != "Inconnu":  # type: ignore
            scores["Document d'Arpentage (DMPC)"] += 2
        if "dmpc" in str(kpis.get("n_dossier", "")).lower():  # type: ignore
            scores["Document d'Arpentage (DMPC)"] += 2

    meilleur_type = max(scores, key=lambda k: scores[k])
    if scores[meilleur_type] == 0:
        return "Type Inconnu"
    return meilleur_type


def _load_prompt(name: str) -> str:
    """Charge un fichier de prompt depuis le dossier prompts/ voisin de ce script."""
    prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts', f'{name}.txt')
    with open(prompt_path, encoding='utf-8') as f:
        return f.read().strip()


def llm_commune_correction(hypotheses_ocr, dictionnaire_candidats, llm_pipeline, roi_image=None):
    """
    Arbitrage paléographique par VLM (Qwen2-VL-2B-Instruct).
    
    Le VLM reçoit :
    - L'image brute de la cellule manuscrite (roi_image, numpy BGR)
    - Les hypothèses OCR textuelles
    - La liste restreinte des communes candidates
    
    Il répond avec des balises XML structurées :
    <analyse>raisonnement paléographique</analyse>
    <commune>Nom Officiel</commune>
    
    Si roi_image est None ou si le VLM échoue → fallback texte-seul.
    """
    if not llm_pipeline or not hypotheses_ocr or not dictionnaire_candidats:
        return None

    # Décompacter le tuple VLM
    if not isinstance(llm_pipeline, tuple) or len(llm_pipeline) != 2:
        return None   # Type inattendu → abandon sécurisé
    vlm_model, vlm_processor = llm_pipeline

    # --- Top-10 candidats par similarité fuzzy avec la 1ère hypothèse ---
    try:
        from rapidfuzz import process as rp, fuzz as rf
        top_matches = rp.extract(
            hypotheses_ocr[0][0],
            [c['officiel'] for c in dictionnaire_candidats],
            scorer=rf.WRatio, limit=10
        )
        candidats_noms = [m[0] for m in top_matches]
    except Exception:
        candidats_noms = [c['officiel'] for c in dictionnaire_candidats[:10]]

    hypoth_list      = [h[0] for h in hypotheses_ocr[:4]]
    liste_hyp_str    = ', '.join(f'"{h}"' for h in hypoth_list)
    liste_cand_str   = ', '.join(f'"{c}"' for c in candidats_noms)

    # ------------------------------------------------------------------ #
    # BRANCHE A : Mode Vision (VLM) — image disponible                    #
    # ------------------------------------------------------------------ #
    if roi_image is not None and roi_image.size > 0:
        try:
            import re as _re
            from PIL import Image as _PILImage  # type: ignore

            # Convertir le numpy BGR → PIL RGB
            roi_pil = _PILImage.fromarray(roi_image[:, :, ::-1])  # BGR→RGB

            # ------ Prompts chargés depuis prompts/ ------
            system_text = _load_prompt('vlm_vision_system')
            user_text = (
                f"Hypothèses erronées générées par le premier système OCR : {liste_hyp_str}\n"
                f"Candidats probables (Dictionnaire officiel) : {liste_cand_str}\n\n"
                "En te basant sur l'image fournie et ta liste de candidats, quelle est la commune exacte ?"
            )

            # ------ Construction du message multimodal Qwen2-VL ------
            messages = [
                {"role": "system", "content": system_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": roi_pil},
                        {"type": "text",  "text": user_text},
                    ],
                },
            ]

            # Tokenisation et inférence
            text_input = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Qwen2-VL accepte les PIL Image directement via image_inputs
            image_inputs = [roi_pil]
            inputs = vlm_processor(
                text=[text_input],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
            # Placer les tenseurs sur le même device que le modèle
            target_device = next(vlm_model.parameters()).device
            inputs = {k: v.to(target_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = vlm_model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=False,
                )
            # Décoder uniquement les tokens générés (pas le prompt)
            input_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, input_len:]
            reponse_vlm = vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True
            )[0].strip()

            print(f"      [VLM PALÉOGRAPHE] Réponse brute : {reponse_vlm[:120]}")

            # ------ Extraction de la balise <commune> ------
            m = _re.search(r'<commune>\s*(.+?)\s*</commune>', reponse_vlm, _re.IGNORECASE | _re.DOTALL)
            if m:
                reponse_propre = m.group(1).strip().strip('.,;:!?')
            else:
                # Fallback si le modèle n'a pas utilisé les balises : prendre la dernière ligne non-vide
                lignes = [l.strip() for l in reponse_vlm.split('\n') if l.strip()]
                reponse_propre = lignes[-1] if lignes else ""

            # ------ Validation de la réponse contre le dictionnaire ------
            # 1) Correspondance exacte (insensible à la casse)
            for c in dictionnaire_candidats:
                if c['officiel'].lower() == reponse_propre.lower():
                    print(f"      [VLM ARBITRE ✓] '{hypoth_list[0]}' → '{c['officiel']}' (exact)")
                    return {
                        'officiel': c['officiel'],
                        'code': c['code'],
                        'score': 88,
                        'methode': 'VLM_Paleographe',
                        'brut': hypoth_list[0],
                        'hypotheses_ocr': "Arbitré par VLM Paléographe"
                    }

            # 2) Fuzzy match de sécurité (le VLM peut écrire "Saint-Andéol" vs "Saint-Andéol-de-Berg")
            try:
                from rapidfuzz import process as rp2, fuzz as rf2
                fuzzy_result = rp2.extractOne(
                    reponse_propre,
                    [c['officiel'] for c in dictionnaire_candidats],
                    scorer=rf2.WRatio
                )
                if fuzzy_result and fuzzy_result[1] >= 88:
                    commune_validee = fuzzy_result[0]
                    c_valid = next((c for c in dictionnaire_candidats if c['officiel'] == commune_validee), None)
                    if c_valid:
                        print(f"      [VLM ARBITRE ~] '{hypoth_list[0]}' → '{c_valid['officiel']}' (fuzzy {fuzzy_result[1]:.0f}%)")
                        return {
                            'officiel': c_valid['officiel'],
                            'code': c_valid['code'],
                            'score': int(fuzzy_result[1] * 0.9),  # Légère pénalité fuzzy
                            'methode': 'VLM_Paleographe_fuzzy',
                            'brut': hypoth_list[0],
                            'hypotheses_ocr': "Arbitré par VLM Paléographe (fuzzy)"
                        }
            except Exception:
                pass

            print(f"      [VLM ARBITRE ✗] Réponse '{reponse_propre}' non trouvée dans le dictionnaire.")

        except Exception as e_vlm:
            print(f"      [Erreur VLM Paléographe] {e_vlm}")

    # ------------------------------------------------------------------ #
    # BRANCHE B : Fallback texte-seul (si pas d'image ou erreur VLM)      #
    # ------------------------------------------------------------------ #
    try:
        import re as _re
        system_text_txt = _load_prompt('vlm_texte_system')
        user_text_txt = (
            f"Hypothèses OCR : {liste_hyp_str}\n"
            f"Candidats : {liste_cand_str}\n\n"
            "Quelle est la commune exacte ?"
        )
        messages_txt = [
            {"role": "system", "content": system_text_txt},
            {"role": "user",   "content": user_text_txt},
        ]
        text_input_txt = vlm_processor.apply_chat_template(
            messages_txt, tokenize=False, add_generation_prompt=True
        )
        inputs_txt = vlm_processor(
            text=[text_input_txt], padding=True, return_tensors="pt"
        )
        target_device = next(vlm_model.parameters()).device
        inputs_txt = {k: v.to(target_device) if hasattr(v, 'to') else v for k, v in inputs_txt.items()}

        with torch.no_grad():
            gen_ids_txt = vlm_model.generate(**inputs_txt, max_new_tokens=30, do_sample=False)
        rep_txt = vlm_processor.batch_decode(
            gen_ids_txt[:, inputs_txt["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

        m_txt = _re.search(r'<commune>\s*(.+?)\s*</commune>', rep_txt, _re.IGNORECASE)
        reponse_txt = m_txt.group(1).strip() if m_txt else rep_txt.split('\n')[0].strip()

        for c in dictionnaire_candidats:
            if c['officiel'].lower() == reponse_txt.lower():
                print(f"      [VLM TEXTE ✓] '{hypoth_list[0]}' → '{c['officiel']}'")
                return {
                    'officiel': c['officiel'],
                    'code': c['code'],
                    'score': 82,
                    'methode': 'VLM_Texte_Fallback',
                    'brut': hypoth_list[0],
                    'hypotheses_ocr': "Arbitré par VLM (mode texte)"
                }
    except Exception as e_txt:
        print(f"      [Erreur VLM Texte Fallback] {e_txt}")

    return None



def vlm_first_commune_extraction(roi_image, commune_db, llm_pipeline, ocr_text_hint=""):
    """
    VLM-FIRST v2 — 3 passes progressives.
    Passe 1 : lecture libre avec top-10 candidats dans le prompt.
    Passe 2 : choix forcé parmi top-5 WRatio si P1 insuffisante.
    Passe 3 : sauvetage avec seuil assoupli (65%).
    """
    if not llm_pipeline or roi_image is None or roi_image.size == 0:
        return {'officiel': 'Inconnu', 'score': 0, 'methode': 'erreur_vlm', 'brut': '', 'hypotheses_ocr': ''}

    if not isinstance(llm_pipeline, tuple) or len(llm_pipeline) != 2:
        return {'officiel': 'Inconnu', 'score': 0, 'methode': 'erreur_vlm_type', 'brut': '', 'hypotheses_ocr': ''}

    vlm_model, vlm_processor = llm_pipeline
    import re as _re
    from PIL import Image as _PILImage
    from rapidfuzz import process as rp2, fuzz as rf2

    roi_pil = _PILImage.fromarray(roi_image[:, :, ::-1])

    # Longueur physique estimée depuis l'indice OCR (lettres alpha seulement)
    clean_hint = _re.sub(r'[^a-zA-Z\u00C0-\u017E]', '', ocr_text_hint)
    target_len = len(clean_hint) if len(clean_hint) > 2 else 0

    # ── Top-10 candidats fuzzy inclus dans le prompt passe 1 ──
    all_noms_off = [c['officiel'] for c in commune_db]
    all_noms_norm = [c['normalise'] for c in commune_db]
    hint_search = ocr_text_hint if ocr_text_hint else clean_hint
    if hint_search:
        top10_raw = rp2.extract(normaliser_pour_matching(hint_search), all_noms_norm, scorer=rf2.WRatio, limit=10)
        top10_noms = []
        for n_norm, _, _ in top10_raw:
            e = next((c for c in commune_db if c['normalise'] == n_norm), None)
            if e:
                top10_noms.append(e['officiel'])
    else:
        top10_noms = all_noms_off[:10]
    top10_str = ', '.join(f'"{n}"' for n in top10_noms)

    len_hint_str = f" Le mot manuscrit fait physiquement environ {target_len} lettres." if target_len > 0 else ""
    system_text = (
        "Tu es un paléographe expert en manuscrits administratifs français du XIXe siècle.\n"
        f"Communes les plus probables d'après le contexte : [{top10_str}].\n"
        f"{len_hint_str}\n"
        "RÈGLES :\n"
        "- Si le mot lu correspond exactement ou presque à l'une des communes listées, donne ce nom officiel.\n"
        "- Rejette les mots courants ('le', 'la', 'les', 'de', 'et', 'bains', 'saint' seul).\n"
        "- Lis lettre par lettre, ignore les étoiles et parasites.\n"
        "- Mets ta réponse entre <commune> et </commune>. Rien d'autre.\n"
        "- Si l'image est vide, illisible, ou ne contient pas de texte manuscrit clair, réponds STRICTEMENT <commune>VIDE</commune>."
    )
    user_text = "Quel est le nom de la commune écrite dans cette image ?"

    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": roi_pil},
                {"type": "text",  "text": user_text},
            ],
        },
    ]

    try:
        import torch
        text_input = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Qwen2-VL : image passée dans images=[]
        inputs = vlm_processor(text=[text_input], images=[roi_pil], padding=True, return_tensors="pt")
        target_device = next(vlm_model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=150, do_sample=False)
        
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1]:]
        reponse_vlm = vlm_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

        # Extraction via balise
        m = _re.search(r'<commune>\s*(.+?)\s*</commune>', reponse_vlm, _re.IGNORECASE | _re.DOTALL)
        if m:
            reponse_propre = m.group(1).strip().strip('.,;:!?')
        else:
            lignes = [l.strip() for l in reponse_vlm.split('\n') if l.strip()]
            reponse_propre = lignes[-1] if lignes else ""

        print(f"      [VLM-FIRST PASSE 1] Extrait : '{reponse_propre}'")

        if reponse_propre.upper() in ('VIDE', 'ILLISIBLE', 'NON IDENTIFIÉE', 'INCONNU', 'RIEN'):
            print(f"      [VLM-FIRST] Image jugée vide ou illisible par le VLM.")
            return {'officiel': 'Non identifiée', 'code': '', 'score': 0, 'methode': 'VLM_illisible', 'brut': reponse_propre, 'hypotheses_ocr': ''}

        if len(reponse_propre) <= 2 and ocr_text_hint:
             # Force back to the best hint if the VLM answers something ridiculously short like 'et'
             reponse_propre = ocr_text_hint.split()[0] if ocr_text_hint else reponse_propre

        from rapidfuzz import process as rp2, fuzz as rf2
        
        comp_len = target_len if target_len > 0 else len(reponse_propre)
        
        # ── Matching passe 1 : WRatio sur noms normalisés (gère tirets, accents) ──
        rep_norm = normaliser_pour_matching(reponse_propre)
        hint_norm = normaliser_pour_matching(ocr_text_hint) if ocr_text_hint else ""
        comp_len = target_len if target_len > 0 else len(rep_norm.replace(' ', ''))

        # Mots génériques à rejeter
        _mots_gen = {'LES', 'LE', 'LA', 'DE', 'DU', 'DES', 'ET', 'EN', 'SUR', 'SOUS',
                     'LES BAINS', 'BAINS', 'SAINT', 'SAINTE'}
        if rep_norm in _mots_gen and hint_norm:
            rep_norm = hint_norm  # Tomber sur l'indice OCR si réponse VLM trop générique
            reponse_propre = ocr_text_hint

        combined: dict = {}
        for n_norm, sc, _ in rp2.extract(rep_norm, all_noms_norm, scorer=rf2.WRatio, limit=15):
            combined[n_norm] = max(combined.get(n_norm, 0), int(sc))
        for n_norm, sc, _ in rp2.extract(rep_norm, all_noms_norm, scorer=rf2.ratio, limit=10):
            combined[n_norm] = max(combined.get(n_norm, 0), int(sc * 0.95))
        if hint_norm:
            for n_norm, sc, _ in rp2.extract(hint_norm, all_noms_norm, scorer=rf2.WRatio, limit=10):
                combined[n_norm] = max(combined.get(n_norm, 0), int(sc * 0.90))
        merged = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        # Seuil P1 : 75%, contrainte longueur ±5 chars
        for n_norm, sc in merged:
            entry = next((c for c in commune_db if c['normalise'] == n_norm), None)
            if not entry:
                continue
            n_len = len(n_norm.replace(' ', ''))
            len_ok = (comp_len == 0) or (abs(comp_len - n_len) <= 5)
            if sc >= 75 and len_ok:
                print(f"      [VLM-FIRST P1 ✓] '{reponse_propre}' → '{entry['officiel']}' ({sc}%)")
                return {
                    'officiel': entry['officiel'], 'code': entry['code'],
                    'score': int(sc), 'methode': 'VLM_P1',
                    'brut': reponse_propre, 'hypotheses_ocr': f"VLM P1 ({sc}%)"
                }

        # ── PASSE 2 : Choix forcé parmi top-5 candidats ──
        top5_off = []
        for n_norm, _ in merged[:8]:
            e = next((c for c in commune_db if c['normalise'] == n_norm), None)
            if e:
                top5_off.append(e['officiel'])
            if len(top5_off) >= 5:
                break
        if not top5_off:
            top5_off = top10_noms[:5]

        liste_p2 = " | ".join(top5_off)
        print(f"      [VLM-FIRST P2] Choix restreint : [{liste_p2}]")

        system_text_2 = (
            "Tu es un expert paléographe.\n"
            f"Ta première lecture a donné : '{reponse_propre}'.\n"
            f"Voici les 5 seuls choix possibles : [{liste_p2}].\n"
            "Observe l'image encore : longueur du mot, première lettre, boucles.\n"
            f"{'Indice longueur : environ ' + str(target_len) + ' lettres.' if target_len > 0 else ''}\n"
            "Réponds UNIQUEMENT avec le nom choisi entre <commune> et </commune>.\n"
            "Si l'image est vide ou vraiment impossible à lire, réponds : <commune>VIDE</commune>."
        )
        messages_2 = [
            {"role": "system", "content": system_text_2},
            {"role": "user", "content": [
                {"type": "image", "image": roi_pil},
                {"type": "text",  "text": "Choisissez parmi les options :"},
            ]},
        ]
        text_input_2 = vlm_processor.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
        inputs_2 = vlm_processor(text=[text_input_2], images=[roi_pil], padding=True, return_tensors="pt")
        inputs_2 = {k: v.to(target_device) if hasattr(v, 'to') else v for k, v in inputs_2.items()}
        with torch.no_grad():
            gen_2 = vlm_model.generate(**inputs_2, max_new_tokens=40, do_sample=False)
        rep_2_raw = vlm_processor.batch_decode(
            gen_2[:, inputs_2["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        m2 = _re.search(r'<commune>\s*(.+?)\s*</commune>', rep_2_raw, _re.IGNORECASE | _re.DOTALL)
        rep_2 = m2.group(1).strip().strip('.,;:!?') if m2 else \
            ([l.strip() for l in rep_2_raw.split('\n') if l.strip()] or [""])[-1]
        print(f"      [VLM-FIRST P2] Réponse : '{rep_2}'")

        if rep_2.upper() in ('VIDE', 'ILLISIBLE', 'NON IDENTIFIÉE', 'INCONNU', 'RIEN'):
            print(f"      [VLM-FIRST P2] Image jugée vide ou illisible par le VLM.")
            return {'officiel': 'Non identifiée', 'code': '', 'score': 0, 'methode': 'VLM_P2_illisible', 'brut': rep_2, 'hypotheses_ocr': ''}

        if "ILLISIBLE" not in rep_2.upper() and len(rep_2) > 2:
            rep_2_norm = normaliser_pour_matching(rep_2)
            res_p2 = rp2.extractOne(rep_2_norm, all_noms_norm, scorer=rf2.WRatio)
            if res_p2:
                n_norm_p2, sc_p2, _ = res_p2
                e_p2 = next((c for c in commune_db if c['normalise'] == n_norm_p2), None)
                if e_p2:
                    n_len_p2 = len(n_norm_p2.replace(' ', ''))
                    len_ok_p2 = (comp_len == 0) or (abs(comp_len - n_len_p2) <= 5)
                    seuil_p2 = 70 if len_ok_p2 else 82
                    if sc_p2 >= seuil_p2:
                        print(f"      [VLM-FIRST P2 ✓] '{rep_2}' → '{e_p2['officiel']}' ({sc_p2}%)")
                        return {
                            'officiel': e_p2['officiel'], 'code': e_p2['code'],
                            'score': int(sc_p2 * 0.95), 'methode': 'VLM_P2',
                            'brut': rep_2, 'hypotheses_ocr': f"VLM P2 ({sc_p2}%)"
                        }

        # ── PASSE 3 : Sauvetage — meilleur candidat P1 seuil 65% ──
        for n_norm, sc in merged[:5]:
            e = next((c for c in commune_db if c['normalise'] == n_norm), None)
            if not e:
                continue
            n_len = len(n_norm.replace(' ', ''))
            len_ok = (comp_len == 0) or (abs(comp_len - n_len) <= 6)
            if sc >= 65 and len_ok:
                print(f"      [VLM-FIRST P3 ✓] Sauvetage '{e['officiel']}' ({sc}%)")
                return {
                    'officiel': e['officiel'], 'code': e['code'],
                    'score': int(sc * 0.85), 'methode': 'VLM_P3_rescue',
                    'brut': reponse_propre, 'hypotheses_ocr': f"VLM P3 rescue ({sc}%)"
                }

        print(f"      [VLM-FIRST ECHEC] Aucune passe concluante.")

    except Exception as e:
        print(f"      [Erreur VLM First] {e}")

    return {'officiel': 'Inconnu', 'score': 0, 'methode': 'echec_vlm',
            'brut': reponse_propre if 'reponse_propre' in locals() else '', 'hypotheses_ocr': ''}


# ============================================================
# MODULE 2 : WRITER STYLE LEARNER
# Apprentissage persistant de l'écriture manuscrite par géomètre.
# Utilise des vecteurs HOG simplifiés pour la corrélation lettre-pixel.
# ============================================================

class WriterStyleLearner:
    """
    Apprentissage persistant de l'écriture manuscrite d'un géomètre.

    Principe :
    - Quand une commune est identifiée avec score >= SEUIL_CERTITUDE (85%),
      on extrait un vecteur de caractéristiques visuelles (HOG simplifié + hist pixels)
      de la ROI de la cellule manuscrite.
    - Ces vecteurs sont sauvegardés en JSON sur disque, par géomètre.
    - Lors d'une détection incertaine, on compare visuellement la cellule
      aux ancres connues via similarité cosinus → bonus de score.
    - Les corrections OCR apprises (texte brut → commune réelle) sont aussi sauvegardées.
    """

    SEUIL_CERTITUDE = 92  # Score minimum pour enregistrer une ancre (relevé pour éviter les faux positifs)

    def __init__(self, geometre_id: str, base_dir: str = "writer_styles"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(script_dir, base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.geometre_id = geometre_id
        self.data_path = os.path.join(self.base_dir, f"{geometre_id}.json")
        self.confirmed: Dict[str, List[Dict]] = {}  # commune → [{ocr_text, feature_vec, ts}]
        self.letter_corrections: Dict[str, str] = {}  # ocr_brut → commune_officielle
        self.stats: Dict[str, int] = {"total_confirmed": 0, "total_learned": 0}
        self.load()

    def load(self) -> None:
        """Charge les données persistées depuis le JSON sur disque."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                self.confirmed = raw.get("confirmed", {})
                self.letter_corrections = raw.get("letter_corrections", {})
                self.stats = raw.get("stats", {"total_confirmed": 0, "total_learned": 0})
                n_ancres = sum(len(v) for v in self.confirmed.values())
                print(f"  [WriterLearner] Chargé '{self.geometre_id}' : "
                      f"{len(self.confirmed)} communes connues, {n_ancres} ancres visuelles.")
            except Exception as e:
                print(f"  [WriterLearner] Erreur chargement '{self.data_path}': {e}")
        else:
            print(f"  [WriterLearner] Nouveau profil créé pour '{self.geometre_id}'.")

    def save(self) -> None:
        """Persiste les données sur disque."""
        data = {
            "geometre_id": self.geometre_id,
            "confirmed": self.confirmed,
            "letter_corrections": self.letter_corrections,
            "stats": self.stats
        }
        try:
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [WriterLearner] Erreur sauvegarde : {e}")

    def _extract_features(self, roi: np.ndarray) -> List[float]:
        """
        Extrait un vecteur de caractéristiques HOG simplifié d'une ROI.
        Taille fixe : 64x16 pixels → gradient orienté en 8 directions × (4×1) cellules
        Retourne un vecteur de 32 floats normalisé.
        """
        if roi is None or roi.size == 0:
            return [0.0] * 64

        try:
            # Convertir en niveaux de gris
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()

            # Redimensionner à taille fixe pour comparaison
            resized = cv2.resize(gray, (64, 32), interpolation=cv2.INTER_AREA)

            # Normaliser le contraste
            resized = cv2.equalizeHist(resized)

            # Calculer les gradients
            gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)
            angle = np.arctan2(gy, gx)  # -π à π

            # HOG simplifié : 8 orientations × cellules 16×16 px (4 cellules au total)
            n_bins = 8
            bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
            features = []

            cell_h, cell_w = 16, 16
            for cy in range(0, 32, cell_h):
                for cx in range(0, 64, cell_w):
                    m_cell = magnitude[cy:cy+cell_h, cx:cx+cell_w]
                    a_cell = angle[cy:cy+cell_h, cx:cx+cell_w]
                    hist, _ = np.histogram(a_cell, bins=bin_edges, weights=m_cell)
                    norm = np.linalg.norm(hist)
                    features.extend((hist / (norm + 1e-6)).tolist())

            # Ajouter un histogramme de niveaux de gris simple (16 bins)
            gray_hist, _ = np.histogram(resized, bins=16, range=(0, 256))
            gray_norm = np.linalg.norm(gray_hist)
            features.extend((gray_hist / (gray_norm + 1e-6)).tolist())

            return features

        except Exception as e:
            print(f"  [WriterLearner._extract_features] Erreur : {e}")
            return [0.0] * 64

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Similarité cosinus entre deux vecteurs."""
        a = np.array(v1, dtype=np.float64)
        b = np.array(v2, dtype=np.float64)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(a, b) / denom)

    def record_confirmed(self, commune: str, roi: np.ndarray, ocr_text: str) -> None:
        """
        Enregistre une ancre visuelle pour une commune confirmée (score >= 85%).
        - Extrait le vecteur de features de la ROI
        - Ajoute au dictionnaire confirmed[commune]
        - Ajoute la correction OCR si le texte brut diffère du nom officiel
        - Sauvegarde sur disque
        """
        features = self._extract_features(roi)
        if not any(f != 0.0 for f in features):
            return  # ROI non exploitable, on ignore

        entry = {
            "ocr_text": ocr_text,
            "feature_vec": features,
            "timestamp": str(pd.Timestamp.now())
        }

        if commune not in self.confirmed:
            self.confirmed[commune] = []

        # Éviter les doublons visuels exacts (même texte OCR)
        # On garde max 10 ancres par commune pour ne pas surcharger le JSON
        existing_ocr = [e["ocr_text"] for e in self.confirmed[commune]]
        if ocr_text not in existing_ocr:
            self.confirmed[commune].append(entry)
            if len(self.confirmed[commune]) > 10:
                self.confirmed[commune] = self.confirmed[commune][-10:]
            self.stats["total_confirmed"] = self.stats.get("total_confirmed", 0) + 1
            print(f"  [WriterLearner] ✓ Ancre enregistrée : '{commune}' "
                  f"(OCR brut: '{ocr_text}') — total: {self.stats['total_confirmed']}")

        # Enregistrer la correction OCR si utile
        ocr_norm = normaliser_pour_matching(ocr_text)
        commune_norm = normaliser_pour_matching(commune)
        if ocr_norm and ocr_norm != commune_norm:
            self.letter_corrections[ocr_norm] = commune
            self.stats["total_learned"] = self.stats.get("total_learned", 0) + 1

        self.save()

    def get_visual_boost(self, roi: np.ndarray, commune_db: List[Dict]) -> Dict[str, float]:
        """
        [DÉSACTIVÉ - HOG sur papier d'archive capture la texture du papier/lignes,
         pas les lettres → résultat : toutes les cellules ressemblent à la même commune]

        La méthode retourne toujours un dict vide pour ne pas interférer avec le scoring.
        Le learner conserve uniquement les corrections textuelles (check_ocr_correction).
        """
        return {}

    def check_ocr_correction(self, ocr_text: str) -> Optional[str]:
        """
        Vérifie si ce texte OCR a déjà été corrigé dans les sessions précédentes.
        Retourne le nom officiel de la commune si trouvé, None sinon.
        """
        ocr_norm = normaliser_pour_matching(ocr_text)
        if ocr_norm in self.letter_corrections:
            return self.letter_corrections[ocr_norm]
        return None


# ============================================================
# MODULE 3 : RÉSOLUTION AVANCÉE DES ABRÉVIATIONS / INCONNUS
# ============================================================

def resolve_unknown_commune(
    ocr_text: str,
    commune_db: List[Dict],
    writer_learner: Optional['WriterStyleLearner'] = None,
    roi: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Tentative de résolution d'une commune non identifiée par le matching principal.

    Cascade de 5 étapes :
    1. Vérification des corrections OCR apprises (WriterStyleLearner)
    2. Matching par préfixe dans la base officielle
    3. Filtrage par longueur du texte OCR
    4. Smith-Waterman + boost visuel HOG
    5. Retourne le meilleur candidat (ou 'Non identifiée' si score < 40%)
    """
    if not ocr_text or not commune_db:
        return {'officiel': 'Non identifiée', 'score': 0, 'methode': 'vide', 'brut': ocr_text}

    txt_norm = normaliser_pour_matching(ocr_text)
    if len(txt_norm) < 2:
        return {'officiel': 'Non identifiée', 'score': 0, 'methode': 'trop_court', 'brut': ocr_text}

    # ── ÉTAPE 1 : Corrections OCR apprises ──
    if writer_learner:
        correction = writer_learner.check_ocr_correction(ocr_text)
        if correction:
            len_ocr = len(txt_norm.replace(' ', ''))
            len_off = len(correction.replace(' ', ''))
            if len_ocr >= max(3, int(len_off * 0.35)):
                idx = next((i for i, e in enumerate(commune_db) if e['officiel'] == correction), -1)
                if idx >= 0:
                    print(f"    [ResolveCom] APPRIS : '{ocr_text}' → '{correction}'")
                    return {
                        'officiel': correction,
                        'code': commune_db[idx]['code'],
                        'score': 92,
                        'methode': 'correction_apprise',
                        'brut': ocr_text
                    }
            else:
                print(f"    [ResolveCom] APPRIS ignoré (texte '{ocr_text}' trop court pour '{correction}')")

    # ── ÉTAPE 2 : Recherche par PRÉFIXE ──
    # Expansion des abréviations connues
    txt_expanded = txt_norm
    if txt_norm.startswith('ST ') or txt_norm == 'ST':
        txt_expanded = 'SAINT' + txt_norm[2:]
    elif txt_norm.startswith('STE ') or txt_norm == 'STE':
        txt_expanded = 'SAINTE' + txt_norm[3:]

    prefix_matches = []
    for entry in commune_db:
        nom_norm = entry['normalise']
        # Correspondance si la commune commence par le texte OCR (ou son expansion)
        if nom_norm.startswith(txt_expanded) or nom_norm.startswith(txt_norm):
            prefix_matches.append(entry)
        # Aussi tester si le 1er mot significatif de la commune == texte OCR
        elif txt_norm and nom_norm.split()[0] == txt_norm.split()[0] and len(txt_norm) >= 4:
            prefix_matches.append(entry)

    n_prefix = len(prefix_matches)
    print(f"    [ResolveCom] Préfixe '{txt_norm}' → {n_prefix} candidats")

    if n_prefix == 1:
        entry = prefix_matches[0]
        len_ocr = len(txt_norm.replace(' ', ''))
        len_off = len(entry['normalise'].replace(' ', ''))
        
        if len_ocr >= 5 or len_ocr >= len_off * 0.45:
            score_mod = _apply_ocr_quality_modifier(85.0, txt_norm, entry)
            if score_mod >= 45.0:
                print(f"    [ResolveCom] PRÉFIXE UNIQUE → '{entry['officiel']}' ({score_mod:.1f}%)")
                return {
                    'officiel': entry['officiel'],
                    'code': entry['code'],
                    'score': int(score_mod),
                    'methode': 'prefixe_unique',
                    'brut': ocr_text
                }
            else:
                print(f"    [ResolveCom] PRÉFIXE UNIQUE '{entry['officiel']}' rejeté par QualityMod ({score_mod:.1f}%)")
        else:
            print(f"    [ResolveCom] PRÉFIXE UNIQUE '{entry['officiel']}' ignoré ('{txt_norm}' trop court)")

    # ── ÉTAPE 3 : Filtrage par LONG    UEUR ──
    # La longueur du texte OCR est proportionnelle à la longueur du nom de commune
    if n_prefix > 1:
        len_ocr = len(txt_norm.replace(' ', ''))  # Longueur sans espaces
        filtered = []
        for entry in prefix_matches:
            len_commune = len(entry['normalise'].replace(' ', ''))
            # Tolérance HYPER large pour les abréviations : jusqu'à 5x plus longue
            if len_commune >= len_ocr * 0.5 and len_commune <= len_ocr * 5.0:
                filtered.append(entry)
        if filtered:
            prefix_matches = filtered
            print(f"    [ResolveCom] Après filtre longueur ({len_ocr} chars) → {len(prefix_matches)} candidats")

    # ── ÉTAPE 4 : Smith-Waterman sur candidats avec filtre longueur estimée ──
    #
    # Principe : même si l'OCR lit les mauvais caractères, le NOMBRE de caractères
    # qu'il génère est un bon proxy de la longueur réelle de la commune manuscrite.
    # On supprime la borne HAUTE pour les abréviations massives (ex: "St J de S" pour "Saint Julien du Serre")
    #
    len_ocr_chars = len(txt_norm.replace(' ', ''))  # Longueur estimée sans espaces

    if prefix_matches:
        candidates = prefix_matches
        best_methode = 'sw_prefixe'
    else:
        # Pas de préfixe → filtrer la base complète par longueur estimée
        # Tolérance : [0.40x, INFINI] de la longueur OCR
        if len_ocr_chars >= 2:
            candidates = [
                e for e in commune_db
                if len(e['normalise'].replace(' ', '')) >= int(len_ocr_chars * 0.40)
            ]
            print(f"    [ResolveCom] Filtre longueur souple ({len_ocr_chars} chars OCR) "
                  f"→ {len(candidates)}/{len(commune_db)} communes candidates")
            if not candidates:
                candidates = commune_db  # Sécurité : si filtre trop strict, garder tout
        else:
            candidates = commune_db
        best_methode = 'sw_longueur'

    best_score = 0.0
    best_idx = 0

    for entry in candidates:
        sw = _char_alignment_score(txt_norm, entry['normalise'])
        # Astuce Abréviation : tolérance forte via RapidFuzz
        abrev_score = fuzz.token_set_ratio(txt_expanded, entry['normalise'])
        partial_score = fuzz.partial_ratio(txt_expanded, entry['normalise'])
        
        # Le score brut est le max entre SW pur et une pondération abréviation
        score_fusion = max(sw, (abrev_score + partial_score) / 2.0)
        score_brut = min(100.0, score_fusion)

        # ── Modificateur de qualité statistique ──
        # Réduit automatiquement le score si l'OCR est non-informatif (symboles, 1 char)
        # ou si la longueur de la commune est statistiquement incompatible avec l'OCR.
        total = _apply_ocr_quality_modifier(score_brut, txt_norm, entry)

        if total > best_score:
            best_score = total
            best_idx = commune_db.index(entry) if entry in commune_db else 0

    # ── ÉTAPE 5 : Résultat ──
    # Le seuil de 40% s'applique APRÈS le modificateur de qualité :
    # un OCR de mauvaise qualité ne peut plus passer ce seuil même si son score brut était haut.
    if best_score < 40.0:
        print(f"    [ResolveCom] Score après qualité trop bas ({best_score:.1f}%) → Non identifiée")
        return {'officiel': 'Non identifiée', 'score': 0, 'methode': 'echec_qualite_ocr', 'brut': ocr_text}

    meilleure = commune_db[best_idx]
    print(f"    [ResolveCom] → '{meilleure['officiel']}' (score={best_score:.0f}%, méthode={best_methode})")
    return {
        'officiel': meilleure['officiel'],
        'code': meilleure['code'],
        'score': int(best_score),
        'methode': best_methode,
        'brut': ocr_text
    }


# ============================================================
# MODULE 1 : DÉTECTION COLONNE → CASES (morphologie robuste)
# ============================================================

def detect_commune_cells(
    img_cv: np.ndarray,
    tess_df: 'pd.DataFrame',
    easyocr_boxes: List,
    header_y_max: int = 300
) -> List[Dict[str, Any]]:
    """
    Détecte les cellules manuscrites de colonne(s) COMMUNE dans une image de page.

    Étapes séquentielles :
    1. Localiser les en-têtes "COMMUNE" (Tesseract + EasyOCR)
    2. Pour chaque en-tête :
       a. Morphologie verticale → bornes gauche/droite de la colonne
       b. 3 méthodes combinées pour les lignes horizontales :
          - Méthode A : Morphologie (lignes noires nettes)
          - Méthode B : Projection horizontale (lignes grises)
          - Méthode C : Hough Transform probabiliste (lignes cassées)
       c. Chaque paire de lignes horizontales consécutives = 1 case
    3. Retourne la liste complète des cellules avec leurs coordonnées absolues.

    Retourne :
        [{'col_id': int, 'cell_idx': int, 'bbox': [x1,y1,x2,y2],
          'header_bbox': [hx1,hy1,hx2,hy2]}, ...]
    """
    img_h, img_w = img_cv.shape[:2]
    cells: List[Dict[str, Any]] = []
    candidate_headers: List[Tuple[int, int, int, int]] = []  # (x1, x2, y1, y2)

    # ── ÉTAPE 1A : En-têtes via Tesseract ──
    if isinstance(tess_df, pd.DataFrame) and not tess_df.empty:
        df_valid = tess_df.dropna(subset=['text'])
        for _, row in df_valid.iterrows():
            conf = int(row['conf']) if str(row['conf']).isdigit() else -1
            texte = str(row['text']).strip()
            if conf < 30:
                continue
            y_row = int(row['top'])
            h_row = int(row['height'])
            if y_row > header_y_max:
                continue
            txt_norm_tess = normaliser_pour_matching(texte)
            if ('commune' in texte.lower() or (process and fuzz.WRatio(txt_norm_tess, 'COMMUNE') >= 75)) and h_row < 100:
                x1, y1 = int(row['left']), y_row
                x2, y2 = x1 + int(row['width']), y1 + h_row
                candidate_headers.append((x1, x2, y1, y2))
                print(f"  [CellDetect] En-tête COMMUNE (Tesseract) X=[{x1},{x2}] Y=[{y1},{y2}]")

    # ── ÉTAPE 1B : En-têtes via EasyOCR ──
    for r in easyocr_boxes:
        bbox_pts, text_brut, conf_brute = r
        if conf_brute < 0.20:
            continue
        (tl, tr, br, bl) = bbox_pts
        x1 = int(min(tl[0], bl[0]))
        y1 = int(min(tl[1], tr[1]))
        x2 = int(max(tr[0], br[0]))
        y2 = int(max(bl[1], br[1]))
        if y1 > header_y_max:
            continue
        txt_norm_h = normaliser_pour_matching(text_brut)
        if process and (fuzz.WRatio(txt_norm_h, 'COMMUNE') >= 75 or 'commune' in text_brut.lower().strip()):
            if (y2 - y1) < 100:
                # Vérifier qu'on n'a pas déjà un en-tête très proche
                already = any(abs(hx1 - x1) < 60 and abs(hy1 - y1) < 30
                              for hx1, _, hy1, _ in candidate_headers)
                if not already:
                    candidate_headers.append((x1, x2, y1, y2))
                    print(f"  [CellDetect] En-tête COMMUNE (EasyOCR) X=[{x1},{x2}] Y=[{y1},{y2}]")

    if not candidate_headers:
        print("  [CellDetect] Aucun en-tête COMMUNE trouvé.")
        return []

    # ── ÉTAPE 2 : Pour chaque en-tête, détecter colonne + cases ──
    for col_id, (hx1, hx2, hy1, hy2) in enumerate(candidate_headers):
        hcx = (hx1 + hx2) / 2.0

        # ── 2A : Bornes gauche/droite de la colonne (Géométrie Fixe) ──
        # Nouvelle approche 100% robuste : aucune détection d'image aléatoire pour les colonnes.
        # Puisque les traits manuscrits sont trop faibles, on s'aligne mathématiquement sur
        # la boîte de l'en-tête (hx1, hx2) trouvée par YOLO.
        # Marge : 25px à gauche pour le C de "Commune", 70px à droite pour les noms très longs.
        col_x1 = max(0, hx1 - 25)
        col_x2 = min(img_w, hx2 + 70)

        print(f"  [CellDetect] Col {col_id} délimitée : X=[{col_x1}, {col_x2}] (largeur={col_x2-col_x1}px)")

        # ── 2B : Lignes horizontales (3 méthodes combinées) ──
        x_cell_start = int(max(0, col_x1))
        x_cell_end = int(min(img_w, col_x2))
        y_cell_start = int(hy2)
        y_cell_end = int(min(img_h, img_h))  # Jusqu'en bas de la page

        cell_roi = img_cv[y_cell_start:y_cell_end, x_cell_start:x_cell_end]
        if cell_roi.size == 0:
            continue

        gray_cell = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
        h_lines_y: List[int] = []  # Positions Y absolues des lignes horizontales

        # == MÉTHODE B (Exclusive pour les Registres) : Espaces Inter-Mots ==
        # Sur les cahiers de terrain, on IGNORE totalement les lignes physiques (MÉTHODE A/C DÉSACTIVÉE)
        # car le géomètre écrit souvent "par-dessus" les lignes. On découpe UNIQUEMENT au niveau du blanc !
        
        try:
            # Pour ignorer les lignes physiques de colonne (gauche/droite)
            # On recadre de 15% à 85% au centre de la colonne
            col_width_c = x_cell_end - x_cell_start
            x_b_start = int(col_width_c * 0.15)
            x_b_end = int(col_width_c * 0.85)
            gray_b_core = gray_cell[:, x_b_start:x_b_end]

            if gray_b_core.shape[1] > 20: 
                # Lissage pour atténuer le grain du papier
                gray_b_blur = cv2.medianBlur(gray_b_core, 3)
                # Binarisation adaptative (immunisée contre les ombres et gradients d'illumination de la photo)
                # Capte magnifiquement les boucles estompées des p, y, g, l etc.
                thresh_ink = cv2.adaptiveThreshold(
                    gray_b_blur, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 
                    blockSize=51, C=12
                )

                # Masquer les restes de lignes verticales (pour isoler les lignes d'écriture)
                kernel_v_rm = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
                vert_lines = cv2.morphologyEx(thresh_ink, cv2.MORPH_OPEN, kernel_v_rm)
                thresh_ink = cv2.subtract(thresh_ink, vert_lines)
                
                # Masquer UNIQUEMENT les lignes horizontales parfaites du cahier imprimé !
                # Une ligne de cahier est fine (1-2 px max) et très très longue (strictement horizontale)
                # En faisant une érosion morphologique légère, on détruit ces lignes fines de cahier, mais l'encre manuscrite survit.
                kernel_erosion = np.ones((2, 2), np.uint8)
                thresh_ink = cv2.erode(thresh_ink, kernel_erosion, iterations=1)

                # Dilater l'encre fortement en horizontal = une bande continue par ligne de texte
                kernel_b = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int((x_b_end - x_b_start) * 0.4)), 1))
                dilated_ink = cv2.dilate(thresh_ink, kernel_b, iterations=1)

                row_sum_b = np.sum(dilated_ink, axis=1) / 255.0

                # On est plus strict sur l'encre puisque c'est purement au niveau du texte (lignes verticales enlevées)
                seuil_vide = (x_b_end - x_b_start) * 0.05

                in_gap = False
                gap_start = 0
                for y_i, val in enumerate(row_sum_b):
                    if val <= seuil_vide:
                        if not in_gap:
                            in_gap = True
                            gap_start = y_i
                    else:
                        if in_gap:
                            in_gap = False
                            gap_center = (gap_start + y_i) // 2
                            # On ne crée un split que si le gap est assez grand (tolérance forte pour mots sur 2 lignes)
                            if (y_i - gap_start) >= 12:
                                h_lines_y.append(gap_center + y_cell_start)
                
                # Gérer le dernier gap s'il va jusqu'en bas
                if in_gap and (len(row_sum_b) - gap_start) >= 2:
                    h_lines_y.append((gap_start + len(row_sum_b)) // 2 + y_cell_start)

        except Exception as e_b_new:
            print(f"    [CellDetect] Méthode B (Densité V) erreur: {e_b_new}")

        # == FUSION et DÉDOUBLONNAGE des lignes (tolérance ±8 px) ==
        h_lines_y.sort()
        merged_lines: List[int] = []
        for ly in h_lines_y:
            if not merged_lines or abs(ly - merged_lines[-1]) > 8:
                merged_lines.append(ly)
            else:
                # Garder la médiane du cluster
                merged_lines[-1] = (merged_lines[-1] + ly) // 2

        # Toujours ajouter l'en-tête comme première ligne et le bas de page si nécessaire
        if not merged_lines or merged_lines[0] > y_cell_start + 20:
            merged_lines.insert(0, y_cell_start)
        if merged_lines[-1] < img_h - 50:
            merged_lines.append(min(img_h, merged_lines[-1] + 250))  # Ligne fantôme finale

        print(f"  [CellDetect] Col {col_id} → {len(merged_lines)-1} lignes horizontales fusionnées")

        # ── 2C : Construire les cellules ──
        for cell_idx in range(len(merged_lines) - 1):
            cy1 = merged_lines[cell_idx]
            cy2 = merged_lines[cell_idx + 1]
            cell_height = cy2 - cy1

            # Filtrer les cellules trop petites (< 10px) ou trop grandes (> 250px)
            if cell_height < 10 or cell_height > 250:
                continue

            # Finies les zones aléatoires, la géométrie est parfaitement droite !
            cells.append({
                'col_id': col_id,
                'cell_idx': cell_idx,
                'bbox': [int(col_x1), int(cy1), int(col_x2), int(cy2)],
                'header_bbox': [hx1, hy1, hx2, hy2]
            })

        print(f"  [CellDetect] Col {col_id} → {sum(1 for c in cells if c['col_id']==col_id)} cases valides")

    return cells


def process_document_hybrid(file_path, models, villes_dict, commune_db):
    """
    Pipeline hybride Tesseract + EasyOCR/TrOCR + VLM avec :
    - Détection de colonnes COMMUNE par morphologie (Module 1)
    - Apprentissage persistant WriterStyleLearner (Module 2)
    - Résolution avancée des abréviations (Module 3)
    - Boucle itérative + second passage de confirmation (Module 4)
    - Identification du géomètre comme clé d'apprentissage (Module 5)
    """
    try:
        from kraken import blla  # type: ignore
    except ImportError:
        print("Erreur : Kraken n'est pas installé. Lancez 'pip install kraken'.")
        return

    import time
    t0_fichier = time.time()

    _, easyocr_reader, processor, trocr_model, device, spell, pylaia_model, llm_pipeline = models

    images = read_document(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # ── MODULE 5 : Identification du géomètre ──
    # On utilise le préfixe du nom de fichier comme géomètre_id.
    # Exemple : livretFernand_1.pdf → geometre_id = "livretFernand"
    geometre_id = re.sub(r'[_\-]?\d+$', '', base_name).strip() or base_name
    writer_learner = WriterStyleLearner(geometre_id)
    print(f"  [Module5] Apprentissage chargé pour géomètre : '{geometre_id}'")

    results_data: List[Dict[str, Any]] = []
    annotated_pages: List[Any] = []

    # Mémoriser les cellules détectées sur la page 1 pour les réutiliser si la page 2+ n'a pas d'en-tête
    global_cells_template: List[Dict[str, Any]] = []

    for idx, img_cv in enumerate(images):
        print(f"\nTraitement de la page/image {idx + 1} (Dual-Pipeline v6)...")
        img_annotated = img_cv.copy()
        page_results: List[Dict[str, Any]] = []

        # ── ETAPE 1 : TESSERACT GLOBAL ──
        gray_full = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        custom_tess_config = r'--oem 3 --psm 3 -l fra'
        tess_df = run_tesseract_windows(gray_full, custom_tess_config, output_type='tsv')

        if isinstance(tess_df, pd.DataFrame) and not tess_df.empty:
            tess_df = tess_df.dropna(subset=['text'])
            for _, row in tess_df.iterrows():
                conf = int(row['conf']) if str(row['conf']).isdigit() else -1
                texte = str(row['text']).strip()
                if conf > 65 and texte != '':
                    x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
                    texte_corr = correct_cadastral_rules(texte)
                    page_results.append({
                        "bbox": [x, y, x+w, y+h],
                        "texte": texte_corr,
                        "confiance": f"{conf}%",
                        "type_ocr": "Tesseract (Imprimé)"
                    })

        # ── ETAPE 2 : EASYOCR (CRAFT) + TrOCR ──
        print("  -> Détection des boîtes manuscrites (CRAFT)...")
        ocr_boxes = easyocr_reader.readtext(img_cv.copy())

        for r in ocr_boxes:
            bbox, text_brut, conf_brute = r
            if conf_brute < 0.15:
                continue
            (tl, tr, br, bl) = bbox
            x = max(0, int(min(tl[0], bl[0])))
            y = max(0, int(min(tl[1], tr[1])))
            x_max = min(img_cv.shape[1], int(max(tr[0], br[0])))
            y_max = min(img_cv.shape[0], int(max(bl[1], br[1])))
            w = x_max - x
            h = y_max - y
            roi = img_cv[max(0, y-8):min(img_cv.shape[0], y+h+5),
                         max(0, x-5):min(img_cv.shape[1], x+w+10)]
            if roi.size == 0:
                continue
            _, roi_clean = preprocess_roi_for_ocr(roi, mode='htr')
            pixel_values = processor(Image.fromarray(roi_clean), return_tensors="pt").pixel_values.to(device)
            gen_ids = trocr_model.generate(pixel_values, max_new_tokens=20)
            txt_trocr = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            txt_trocr = correct_cadastral_rules(txt_trocr)
            page_results.append({
                "bbox": [x, y, x+w, y+h],
                "texte": txt_trocr,
                "confiance": "N/A",
                "type_ocr": "TrOCR"
            })

        # ── MODULE 1 : DÉTECTION COLONNE → CASES ──
        commune_cells = detect_commune_cells(img_cv, tess_df, ocr_boxes)

        if commune_cells:
            global_cells_template = commune_cells.copy()
            print(f"  [Module1] {len(commune_cells)} cellules COMMUNE détectées (template mémorisé).")
        elif global_cells_template:
            # Réutiliser le template de la page précédente (même géométrie de tableau)
            # On décale simplement les Y si nécessaire (en général même mise en page)
            commune_cells = global_cells_template.copy()
            print(f"  [Module1] Pas de nouvel en-tête trouvé → template réutilisé ({len(commune_cells)} cellules).")

        # Dessiner les colonnes et cellules en vert (debug visuel)
        # Les rectangles visuels utilisent les mêmes marges que _read_cell_ocr
        # pour montrer exactement ce qui est envoyé à TrOCR (pas les lignes de tableau).
        drawn_cols: set = set()
        for cell in commune_cells:
            cx1, cy1, cx2, cy2 = cell['bbox']
            # On aligne le dessin visuel (boîte verte fine) sur ce que TrOCR va RÉELLEMENT lire
            vis_cx1 = max(0, cx1 - 25)
            vis_cx2 = min(img_cv.shape[1] - 1, cx2 + 35 + int((cx2-cx1) * 0.25))
            vis_cy1 = max(0, cy1 - 6)
            vis_cy2 = min(img_cv.shape[0] - 1, cy2 + 6 + int((cy2-cy1) * 0.40))
            
            col_id = cell['col_id']
            if col_id not in drawn_cols:
                cv2.line(img_annotated, (cx1, 0), (cx1, img_cv.shape[0]), (0, 200, 0), 2)
                cv2.line(img_annotated, (cx2, 0), (cx2, img_cv.shape[0]), (0, 200, 0), 2)
                drawn_cols.add(col_id)
            cv2.rectangle(img_annotated, (vis_cx1, vis_cy1), (vis_cx2, vis_cy2), (0, 220, 100), 1)

        # ══════════════════════════════════════════════════════════════════
        #  MODULE 4 — PREMIER PASSAGE : LECTURE + BOUCLE ITÉRATIVE
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  [Module4] === PREMIER PASSAGE — {len(commune_cells)} cellules ===")

        # Résultats par cellule : {cell_idx_global: result_dict}
        cell_results: Dict[int, Dict[str, Any]] = {}

        def _is_cell_empty(roi: np.ndarray) -> bool:
            """Retourne True si la cellule est vide (efface les traits du tableau avant vérification)."""
            if roi is None or roi.size == 0:
                return True
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
            
            # 1. Si le papier est parfaitement uniforme, c'est vide
            if np.std(gray) < 15.0:
                return True
                
            # 2. Gommage des traits du tableau (Magie OpenCV)
            # Binariser pour avoir l'encre en blanc et le fond en noir
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Repérer les traits verticaux du tableau
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            lignes_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
            
            # Repérer les traits horizontaux du tableau
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            lignes_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
            
            # Effacer les traits du tableau de l'image !
            thresh_clean = cv2.subtract(thresh, cv2.add(lignes_v, lignes_h))
            
            # 3. Compter l'encre restante (qui est censée être du texte)
            ink_pixels = np.count_nonzero(thresh_clean)
            if ink_pixels < 120:  # S'il reste moins de 120 pixels d'encre, c'est vide
                return True
                
            # 4. Complexité des contours (pour éviter de lire une tache d'encre ronde)
            edges = cv2.Canny(thresh_clean, 50, 150)
            if np.count_nonzero(edges) < 80:
                return True

            return False
        # === PASSE GLOBALE (L'entièreté du document) ===
        # Le lecteur OCR global "comprend" les lignes dans leur ensemble
        print(f"  [Global] Lancement d'une passe OCR globale sur l'entièreté de la page (itération contextuelle)...")
        global_ocr_data = []
        try:
            reader = models.get('easyocr')
            if reader is not None:
                # Contraste fort pour aider EasyOCR à lire l'encre rouge/pâle
                gray_for_easy = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                gray_for_easy = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_for_easy)
                global_ocr_data = reader.readtext(gray_for_easy)
                print(f"  [Global] {len(global_ocr_data)} blocs de texte trouvés globalement.")
        except Exception as e_glob:
            print(f"  [Global] Erreur passe globale: {e_glob}")

        def _read_cell_ocr(cell: Dict, img_cv: np.ndarray,
                            col_bound_x1: int = 0, col_bound_x2: int = 999999) -> Tuple[List[Tuple[str, float]], np.ndarray]:
            """
            Lit le contenu d'une cellule et retourne (liste_hypotheses, roi_brute).

            Stratégie multi-hypothèses v7 :
            - 4 variantes de prétraitement image
            - Beam search TrOCR (4 faisceaux × 4 variantes = jusqu'à 16 lectures)
            - EasyOCR comme lecteur secondaire indépendant
            - Dédoublonnage et tri par poids de confiance

            Retourne :
                hypotheses : List[(texte, prob)] triée par prob décroissante
                roi_full   : ROI visuelle complète pour l'annotation
            """
            cx1, cy1, cx2, cy2 = cell['bbox']
            cell_h = cy2 - cy1
            cell_w = cx2 - cx1

            # ── Expansion du ROI (Horizontal + Vertical) ──
            # IMPORTANT : Le ROI est ancré dans les bornes de la colonne (col_bound_x1/x2)
            # pour ne JAMAIS déborder sur les colonnes adjacentes (Date, Désignation).
            pad_x_left  = 5   # Réduit de 25px à 5px — les caractères de début sont gérés par col_bound
            pad_x_right = 5   # Réduit de 60px à 5px — ancrage dur sur la borne droite de colonne
            pad_y = 4  # Marge verticale réduite

            cy1_crop = max(0, cy1 - pad_y)
            cy2_crop = min(img_cv.shape[0] - 1, cy2 + pad_y + int(cell_h * 0.15))  # +15% au lieu de +40%
            # Bornes horizontales : jamais en dehors de la colonne détectée
            cx1_crop = max(0, max(col_bound_x1, cx1 - pad_x_left))
            cx2_crop = min(img_cv.shape[1] - 1, min(col_bound_x2, cx2 + pad_x_right))

            roi_brute = img_cv[cy1_crop:cy2_crop, cx1_crop:cx2_crop]
            roi_full  = img_cv[max(0, cy1):min(img_cv.shape[0], cy2),
                               max(0, cx1):min(img_cv.shape[1], cx2)]

            if roi_brute.size == 0:
                return [], roi_full

            # ── Filtre vide ──
            if _is_cell_empty(roi_brute):
                print(f"    [CellOCR] Cellule vide détectée → ignorée")
                return [('[VIDE]', 1.0)], roi_full

            # ── Recadrage horizontal fin par projection de densité ──
            try:
                gray_roi = cv2.cvtColor(roi_brute, cv2.COLOR_BGR2GRAY)
                _, thresh_ink = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # SUPPRESSION des lignes de marge (verticales physiques) pour ne pas fausser le recadrage du texte
                kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(cell_h * 0.5))))
                lignes_v = cv2.morphologyEx(thresh_ink, cv2.MORPH_OPEN, kernel_v)
                thresh_txt = cv2.subtract(thresh_ink, lignes_v)

                proj_col = np.sum(thresh_txt, axis=0).astype(np.float32)
                # On baisse légèrement le seuil puisque le gros du bruit des colonnes est retiré
                seuil_ink = thresh_txt.shape[0] * 255 * 0.015 
                ink_cols  = np.where(proj_col >= seuil_ink)[0]
                
                if len(ink_cols) >= 3:
                    ink_x1 = max(0, int(ink_cols[0])  - 4)
                    ink_x2 = min(roi_brute.shape[1], int(ink_cols[-1]) + 4)
                    if ink_x2 - ink_x1 >= 20 and (ink_x2 - ink_x1) >= cell_w * 0.15:
                        roi_brute = roi_brute[:, ink_x1:ink_x2]
            except Exception:
                pass

            # ════════════════════════════════════════════════════════════
            # LECTURE MULTI-HYPOTHÈSES
            # ════════════════════════════════════════════════════════════
            all_hypotheses: List[Tuple[str, float]] = []

            # ── A : TrOCR avec beam search sur 4 variantes de prétraitement ──
            variants = _build_preprocessing_variants(roi_brute)
            for v_idx, variant_img in enumerate(variants):
                try:
                    pil_img = Image.fromarray(variant_img)
                    pixel_values = processor(pil_img, return_tensors='pt').pixel_values.to(device)

                    # Beam search : 4 faisceaux, retourne les 4 meilleures hypothèses
                    gen_out = trocr_model.generate(
                        pixel_values,
                        max_new_tokens=25,
                        num_beams=4,
                        num_return_sequences=4,
                        early_stopping=True,
                        output_scores=False,
                    )
                    # Poids décroissant par rang : beam 1 = 1.0, beam 2 = 0.80, etc.
                    beam_weights = [1.0, 0.80, 0.65, 0.55]
                    # Poids décroissant par variante : v1 = 1.0, v2 = 0.90, v3 = 0.85, v4 = 0.80
                    variant_weight = [1.0, 0.90, 0.85, 0.80][v_idx] if v_idx < 4 else 0.75

                    texts = processor.batch_decode(gen_out, skip_special_tokens=True)
                    for rank, txt in enumerate(texts):
                        txt_clean = correct_cadastral_rules(txt.strip())
                        if txt_clean:
                            prob = beam_weights[rank] * variant_weight
                            all_hypotheses.append((txt_clean, prob))
                except Exception as e_v:
                    print(f"    [TrOCR v{v_idx}] erreur: {e_v}")

            # ── B : EasyOCR secondaire (lecture indépendante) ──
            try:
                # EasyOCR sur la ROI originale (avant upscaling)
                easy_results = easyocr_reader.readtext(roi_brute, detail=1)
                for _, txt_easy, conf_easy in easy_results:
                    if txt_easy and conf_easy >= 0.25:
                        txt_easy_clean = correct_cadastral_rules(str(txt_easy).strip())
                        if txt_easy_clean:
                            # Poids EasyOCR pondéré par sa confiance (max 0.75 pour être sous TrOCR)
                            all_hypotheses.append((txt_easy_clean, float(conf_easy) * 0.75))
            except Exception as e_easy:
                print(f"    [EasyOCR secondaire] erreur: {e_easy}")

            # ── C : Intégration du Contexte Global ──
            # On vérifie si un texte trouvé par la passe globale tombe dans cette cellule
            for (bbox_g, text_g, conf_g) in global_ocr_data:
                gx1 = min([p[0] for p in bbox_g])
                gy1 = min([p[1] for p in bbox_g])
                gx2 = max([p[0] for p in bbox_g])
                gy2 = max([p[1] for p in bbox_g])
                
                # Check intersection avec la cellule (cy1_crop, cy2_crop, cx1_crop, cx2_crop)
                inter_y = min(gy2, cy2_crop) - max(gy1, cy1_crop)
                inter_x = min(gx2, cx2_crop) - max(gx1, cx1_crop)
                
                h_g = gy2 - gy1
                w_g = gx2 - gx1
                
                # Tolérance : la boîte globale doit correspondre à au moins 40% de la zone
                if h_g > 0 and w_g > 0 and inter_y > h_g * 0.4 and inter_x > w_g * 0.4:
                    if text_g and conf_g > 0.10:
                        txt_g_clean = correct_cadastral_rules(str(text_g).strip())
                        if txt_g_clean:
                            all_hypotheses.append((txt_g_clean, float(conf_g) * 0.95)) # Fort poids car vision globale contextuelle
                            print(f"    [Global Match] Intégration de '{txt_g_clean}' issu de la passe globale.")

            # ── Dédoublonnage et tri ──
            # Fusionner les textes identiques en gardant le poids max
            seen: Dict[str, float] = {}
            for txt_h, prob_h in all_hypotheses:
                key = normaliser_pour_matching(txt_h)
                if key not in seen or prob_h > seen[key]:
                    seen[key] = prob_h
            # Reconstruire la liste unique, triée par prob décroissante
            unique_hyps: List[Tuple[str, float]] = []
            for txt_h, prob_h in all_hypotheses:
                key = normaliser_pour_matching(txt_h)
                if key in seen:
                    unique_hyps.append((txt_h, seen.pop(key)))  # pop = une seule fois par clé
            unique_hyps.sort(key=lambda x: x[1], reverse=True)

            if unique_hyps:
                best_txt = unique_hyps[0][0]
                print(f"    [CellOCR] {len(unique_hyps)} hypothèses — meilleure: '{best_txt}'")
            else:
                print(f"    [CellOCR] Aucune hypothèse produite")

            return unique_hyps if unique_hyps else [('', 1.0)], roi_full

        # Compteur anti-répétition : {commune: count} pour la page courante
        page_commune_counts: Dict[str, int] = {}

        def _apply_first_letter_bonus(ocr_text: str, match_result: Dict) -> Dict:
            """Bonus de +12 points si la première lettre de l'OCR correspond
               à la première lettre de la commune matchée."""
            if not ocr_text or not match_result.get('officiel'):
                return match_result
            if match_result.get('officiel') in ('Non identifiée', 'Inconnu', 'Vide', '[VIDE]'):
                return match_result
            ocr_norm = normaliser_pour_matching(ocr_text)
            commune_norm = normaliser_pour_matching(match_result['officiel'])
            if not ocr_norm or not commune_norm:
                return match_result
            if ocr_norm[0] == commune_norm[0]:
                new_score = min(100, match_result['score'] + 12)
                return {**match_result, 'score': new_score, 'methode': match_result.get('methode','') + '+1ere_lettre'}
            # Malus de -8 points si la première lettre est différente
            new_score = max(0, match_result['score'] - 8)
            return {**match_result, 'score': new_score}

        def _apply_antirep_penalty(match_result: Dict, page_counts: Dict[str, int], n_cells: int) -> Dict:
            """Réduit le score si une commune domine trop (> 30% des cellules de la page)."""
            commune = match_result.get('officiel', '')
            if not commune or commune in ('Non identifiée', '[VIDE]'):
                return match_result
            count = page_counts.get(commune, 0)
            threshold = max(3, int(n_cells * 0.30))
            if count >= threshold:
                penalty = min(40, (count - threshold + 1) * 10)
                new_score = max(0, match_result['score'] - penalty)
                print(f"    [AntiRep] '{commune}' détectée {count}x → malus -{penalty}pts (score: {match_result['score']}→{new_score})")
                return {**match_result, 'score': new_score}
            return match_result

        n_total_cells = len(commune_cells)

        # PASSE 0 — Lecture brute de toutes les cellules (multi-hypothèses)
        for ci, cell in enumerate(commune_cells):
            # Passer les bornes X de la colonne pour ancrer le ROI (évite le débordement sur Date/Désignation)
            _col_x1 = int(cell['bbox'][0])
            _col_x2 = int(cell['bbox'][2])
            hypotheses, roi_brute = _read_cell_ocr(cell, img_cv, col_bound_x1=_col_x1, col_bound_x2=_col_x2)

            # Cellule vide → skip total
            if hypotheses and hypotheses[0][0] == '[VIDE]':
                cell_results[ci] = {
                    'cell': cell, 'roi': roi_brute, 'ocr_brut': '[VIDE]',
                    'match': {'officiel': 'Vide', 'score': 0, 'methode': 'vide', 'brut': ''},
                    'confirmation_status': 'Vide'
                }
                print(f"    [P0] Cellule {ci} → VIDE (ignorée)")
                continue

            txt_brut = hypotheses[0][0] if hypotheses else ''

            # Matching principal avec toutes les hypothèses (abréviations + fuzzy multi-beam)
            match = match_commune_multi_hypotheses(hypotheses, commune_db)
            match = _apply_first_letter_bonus(txt_brut, match)
            match = _apply_antirep_penalty(match, page_commune_counts, n_total_cells)

            # Si résultat insuffisant → resolve_unknown_commune (Module 3)
            if match['score'] < WriterStyleLearner.SEUIL_CERTITUDE:
                match_adv = resolve_unknown_commune(txt_brut, commune_db, writer_learner, roi_brute)
                match_adv = _apply_first_letter_bonus(txt_brut, match_adv)
                match_adv = _apply_antirep_penalty(match_adv, page_commune_counts, n_total_cells)
                if match_adv['score'] > match['score']:
                    match = match_adv

            # ── Qualité OCR : décision VLM ──
            # Le VLM est invoqué UNIQUEMENT si l'OCR porte suffisamment de signal alphabétique.
            # Si l'informativité est trop faible (symboles, 1 char, bruit), le VLM hallucinera
            # sur du vide → on l'empêche de forcer un score élevé sur une cellule non-lisible.
            _info_score = _ocr_informativeness(txt_brut)
            _vlm_eligible = _info_score >= 0.15  # Seuil abaissé : déclencher le VLM dès qu'il y a un signal minimal

            # VLM si toujours incertain ET OCR suffisamment informatif
            if match['score'] < WriterStyleLearner.SEUIL_CERTITUDE and llm_pipeline and _vlm_eligible:
                try:
                    match_vlm = vlm_first_commune_extraction(roi_brute, commune_db, llm_pipeline, txt_brut)
                    match_vlm = _apply_first_letter_bonus(txt_brut, match_vlm)
                    match_vlm = _apply_antirep_penalty(match_vlm, page_commune_counts, n_total_cells)
                    if match_vlm.get('score', 0) > match['score']:
                        match = match_vlm
                except Exception as e_v:
                    print(f"    [VLM erreur passe0] {e_v}")
            elif match['score'] < WriterStyleLearner.SEUIL_CERTITUDE and not _vlm_eligible:
                print(f"    [P0] VLM ignoré — OCR sans signal (info={_info_score:.3f} < 0.15) : '{txt_brut}'")

            cell_results[ci] = {
                'cell': cell,
                'roi': roi_brute,
                'ocr_brut': txt_brut,
                'match': match,
                'confirmation_status': '1\u00e8re passe'
            }

            # Enregistrer comme ancre si déjà certain
            if match['score'] >= WriterStyleLearner.SEUIL_CERTITUDE:
                writer_learner.record_confirmed(match['officiel'], roi_brute, txt_brut)
                cell_results[ci]['confirmation_status'] = 'Confirmé (P1)'
                page_commune_counts[match['officiel']] = page_commune_counts.get(match['officiel'], 0) + 1

            print(f"    [P0] Cellule {ci} (col={cell['col_id']}) ocr='{txt_brut}' "
                  f"\u2192 '{match['officiel']}' ({match['score']}%)")


        # BOUCLE ITÉRATIVE — jusqu'à convergence (max 5 iter)
        for iteration in range(5):
            ancres_avant = sum(1 for r in cell_results.values()
                               if r['match']['score'] >= WriterStyleLearner.SEUIL_CERTITUDE)
            print(f"\n  [Module4] Itération {iteration+1} — {ancres_avant} ancres connues")

            nouvelles = 0
            for ci, data in cell_results.items():
                if data['match']['score'] >= WriterStyleLearner.SEUIL_CERTITUDE:
                    continue  # Déjà résolu
                if data['ocr_brut'] == '[VIDE]':
                    continue  # Cellule vide → jamais résolue

                txt_brut = data['ocr_brut']
                roi_brute = data['roi']

                # resolve_unknown (sans boost visuel HOG)
                match_adv = resolve_unknown_commune(txt_brut, commune_db, writer_learner, roi_brute)
                match_adv = _apply_first_letter_bonus(txt_brut, match_adv)
                match_adv = _apply_antirep_penalty(match_adv, page_commune_counts, n_total_cells)

                # VLM si toujours incertain et OCR suffisamment informatif
                _info_iter = _ocr_informativeness(txt_brut)
                if match_adv['score'] < WriterStyleLearner.SEUIL_CERTITUDE and llm_pipeline and _info_iter >= 0.15:
                    try:
                        match_vlm = vlm_first_commune_extraction(roi_brute, commune_db, llm_pipeline, txt_brut)
                        match_vlm = _apply_first_letter_bonus(txt_brut, match_vlm)
                        match_vlm = _apply_antirep_penalty(match_vlm, page_commune_counts, n_total_cells)
                        if match_vlm.get('score', 0) > match_adv['score']:
                            match_adv = match_vlm
                    except Exception as e_v2:
                        print(f"    [VLM erreur iter{iteration+1}] {e_v2}")
                elif match_adv['score'] < WriterStyleLearner.SEUIL_CERTITUDE and _info_iter < 0.15:
                    print(f"    [Iter{iteration+1}] VLM ignoré — sans signal (info={_info_iter:.3f}) : '{txt_brut}'")

                if match_adv['score'] > data['match']['score']:
                    cell_results[ci]['match'] = match_adv

                if match_adv['score'] >= WriterStyleLearner.SEUIL_CERTITUDE:
                    writer_learner.record_confirmed(match_adv['officiel'], roi_brute, txt_brut)
                    cell_results[ci]['confirmation_status'] = f'Confirmé (Iter{iteration+1})'
                    page_commune_counts[match_adv['officiel']] = page_commune_counts.get(match_adv['officiel'], 0) + 1
                    nouvelles += 1
                    print(f"    [Iter{iteration+1}] Nouvelle ancre : '{match_adv['officiel']}' "
                          f"({match_adv['score']}%) pour cellule {ci}")

            if nouvelles == 0:
                print(f"  [Module4] Convergence atteinte à l'itération {iteration+1}.")
                break

        # ══════════════════════════════════════════════════════════════════
        #  MODULE 4 — SECOND PASSAGE : CONFIRMATION
        # ══════════════════════════════════════════════════════════════════
        print(f"\n  [Module4] === SECOND PASSAGE (confirmation) ===")
        writer_learner.load()  # Recharger pour inclure toutes les ancres du 1er passage

        for ci, data in cell_results.items():
            roi_brute = data['roi']
            match_p1 = data['match']
            txt_brut = data['ocr_brut']

            if roi_brute.size == 0:
                continue

            if txt_brut == '[VIDE]' or match_p1.get('officiel') in ('Vide', '[VIDE]'):
                print(f"    [P2] Cellule {ci} ignorée (VIDE)")
                continue

            # Construire la liste des 3 candidats visuellement les plus proches
            visual_boost_p2 = writer_learner.get_visual_boost(roi_brute, commune_db)
            top3_candidats = sorted(visual_boost_p2.items(), key=lambda x: x[1], reverse=True)[:3]
            top3_noms = [c[0] for c in top3_candidats]
            if not top3_noms:
                # Si le learning visuel est désactivé, on passe les meilleurs fuzzy matches (pour aider le VLM sur les abréviations !)
                txt_exp = normaliser_pour_matching(txt_brut)
                if txt_exp.startswith('ST '): txt_exp = 'SAINT ' + txt_exp[3:]
                elif txt_exp.startswith('STE '): txt_exp = 'SAINTE ' + txt_exp[4:]
                
                top5_fuzzy = sorted(commune_db, key=lambda e: max(
                    fuzz.token_set_ratio(txt_exp, e['normalise']),
                    fuzz.partial_ratio(txt_exp, e['normalise'])
                ), reverse=True)[:5]
                
                top3_noms = list(set([match_p1.get('officiel', '')] + [c['officiel'] for c in top5_fuzzy if c['officiel']]))

            # Re-lecture VLM avec HINT et liste restreinte de candidats
            # Gated : uniquement si l'OCR original est suffisamment informatif
            match_p2 = match_p1
            _info_p2 = _ocr_informativeness(txt_brut)
            if llm_pipeline and _info_p2 >= 0.15:
                try:
                    # On passe le résultat P1 comme OCR hint pour orienter le VLM
                    hint_p2 = match_p1.get('officiel', txt_brut)
                    # Construire une sous-db restreinte aux top3 + quelques voisins fuzzy
                    db_restreinte = [e for e in commune_db if e['officiel'] in top3_noms]
                    if not db_restreinte:
                        db_restreinte = commune_db[:20]
                    match_vlm_p2 = vlm_first_commune_extraction(
                        roi_brute, db_restreinte, llm_pipeline, hint_p2
                    )
                    if match_vlm_p2.get('score', 0) > 0:
                        match_p2 = match_vlm_p2
                except Exception as e_p2:
                    print(f"    [VLM P2 erreur] {e_p2}")
            elif _info_p2 < 0.25:
                print(f"    [P2] VLM ignoré — signal insuffisant (info={_info_p2:.3f}) : '{txt_brut}'")
            else:
                # Sans VLM : réappliquer resolve_unknown avec la base enrichie
                match_p2 = resolve_unknown_commune(txt_brut, commune_db, writer_learner, roi_brute)

            # Logique de confirmation
            officiel_p1 = match_p1.get('officiel', 'Inconnu')
            officiel_p2 = match_p2.get('officiel', 'Inconnu')
            score_p1 = match_p1.get('score', 0)
            score_p2 = match_p2.get('score', 0)

            if officiel_p2 == officiel_p1 and score_p2 > 0:
                # Confirmation : boost +10%
                score_final = min(100, score_p1 + 10)
                match_final = {**match_p1, 'score': score_final, 'methode': match_p1.get('methode','') + '+confirmé'}
                cell_results[ci]['confirmation_status'] = 'Confirmé'
                print(f"    [P2] Cellule {ci} CONFIRMÉE : '{officiel_p1}' ({score_final}%)")
            elif score_p2 > 85 and officiel_p2 != officiel_p1 and officiel_p2 != 'Inconnu':
                # Contradiction forte : P2 est plus sûr
                match_final = match_p2
                cell_results[ci]['confirmation_status'] = 'Corrigé (P2)'
                print(f"    [P2] Cellule {ci} CORRIGÉE : '{officiel_p1}' → '{officiel_p2}' ({score_p2}%)")
            elif score_p2 > score_p1:
                match_final = match_p2
                cell_results[ci]['confirmation_status'] = 'Amélioré (P2)'
                print(f"    [P2] Cellule {ci} améliorée : '{officiel_p2}' ({score_p2}%)")
            else:
                match_final = match_p1
                if officiel_p1 == 'Inconnu' or score_p1 == 0:
                    cell_results[ci]['confirmation_status'] = 'Non résolu'

            # --- SCORE DE CONFIANCE COMPOSITE ---
            ocr_brut_clean = re.sub(r'[^A-Za-z0-9]', '', txt_brut)
            longueur_mot = len(ocr_brut_clean)
            
            # 1. Pénalité sur les mots très courts (bruit lu comme un petit mot)
            if longueur_mot <= 3 and match_final.get('score', 0) > 60:
                # Plafonner artificiellement pour forcer la vérification
                match_final['score'] = min(60, match_final['score'])
                match_final['methode'] = match_final.get('methode', '') + '+penalite_mot_court'

            # 2. Contradiction Multimodale (P1 vs P2)
            if officiel_p1 != 'Inconnu' and officiel_p2 != 'Inconnu' and officiel_p1 != officiel_p2:
                # L'IA hésite fortement entre deux vraies communes, on baisse le score
                match_final['score'] = min(45, match_final['score'])
                match_final['methode'] = match_final.get('methode', '') + '+penalite_contradiction_ia'
                cell_results[ci]['confirmation_status'] = 'Contradiction IA'

            # 3. Cohérence Géométrique
            cx1, cy1, cx2, cy2 = data['cell']['bbox']
            largeur_box = cx2 - cx1
            # Si la boîte est très large (>120px) mais que le mot lu est très court (<=3 lettres)
            if largeur_box > 120 and longueur_mot <= 3 and match_final.get('score', 0) > 0:
                match_final['score'] = min(40, match_final['score'])
                match_final['methode'] = match_final.get('methode', '') + '+penalite_geometrique'

            cell_results[ci]['match'] = match_final

            # Enregistrer les nouvelles certitudes du second passage
            if match_final.get('score', 0) >= WriterStyleLearner.SEUIL_CERTITUDE:
                writer_learner.record_confirmed(match_final['officiel'], roi_brute, txt_brut)

        # ── ANNOTATION VISUELLE ──
        for ci, data in cell_results.items():
            cell = data['cell']
            match_final = data['match']
            conf_status = data['confirmation_status']
            cx1, cy1, cx2, cy2 = cell['bbox']
            officiel = match_final.get('officiel', 'Inconnu')
            c_score = match_final.get('score', 0)
            ocr_brut = data.get('ocr_brut', '')

            if officiel == 'Inconnu' or c_score == 0:
                box_color = (0, 0, 255)
                qualite_tag = "X"
            elif c_score >= WriterStyleLearner.SEUIL_CERTITUDE:
                if 'confirmé' in conf_status.lower() or conf_status == 'Confirmé':
                    box_color = (0, 200, 0)
                    qualite_tag = "✓✓"
                else:
                    box_color = (34, 180, 34)
                    qualite_tag = "✓"
            elif c_score >= 65:
                box_color = (0, 165, 255)
                qualite_tag = "?"
            else:
                box_color = (0, 60, 220)
                qualite_tag = "!!"

            cv2.rectangle(img_annotated, (cx1, cy1), (cx2, cy2), box_color, 3)

            label = f"{qualite_tag} {officiel} ({c_score}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            lbl_x = max(0, cx1 + 4)
            boite_h = cy2 - cy1

            overlay = img_annotated.copy()
            if boite_h >= th + 10:
                lbl_y = cy1 + th + 4
            else:
                lbl_y = cy2 + th + 4

            cv2.rectangle(overlay, (lbl_x, lbl_y - th - 4), (lbl_x + tw + 8, lbl_y + 4), box_color, -1)
            cv2.addWeighted(overlay, 0.4, img_annotated, 0.6, 0, img_annotated)
            cv2.putText(img_annotated, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
            cv2.putText(img_annotated, label, (lbl_x, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

            # OCR brut
            if ocr_brut:
                ocr_display = f"OCR: {ocr_brut[:25]}"
                (tw2, th2), _ = cv2.getTextSize(ocr_display, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                lbl_y2 = lbl_y + th2 + 4
                cv2.putText(img_annotated, ocr_display, (lbl_x, lbl_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
                cv2.putText(img_annotated, ocr_display, (lbl_x, lbl_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # ── CONSTRUCTION DES RÉSULTATS DE PAGE ──
        results_fusionnes: List[Dict[str, Any]] = []
        for ci, data in cell_results.items():
            cell = data['cell']
            match_final = data['match']
            cx1, cy1, cx2, cy2 = cell['bbox']
            officiel = match_final.get('officiel', 'Inconnu')
            c_score = match_final.get('score', 0)
            final_commune = officiel if c_score > 45 else 'Inconnu'

            results_fusionnes.append({
                "id_ligne": f"C{ci}_col{cell['col_id']}",
                "texte": f"[COMMUNE: {final_commune}] OCR='{data['ocr_brut']}'",
                "type_ocr": "TrOCR+VLM",
                "confiance": f"{c_score}%",
                "confiance_commune": c_score,
                "ocr_brut_commune": data['ocr_brut'],
                "hypotheses_ocr": match_final.get('hypotheses_ocr', ''),
                "bbox": [cx1, cy1, cx2, cy2],
                "commune_row": final_commune,
                "commune_match_info": match_final,
                "confirmation_status": data.get('confirmation_status', '1ère passe')
            })

        annotated_pages.append(img_annotated)
        results_data.append({
            "page": idx + 1,
            "detections": results_fusionnes,
            "raw_detections": page_results
        })

    # ── KPIs GLOBAUX (pipeline classique) ──
    kpis = extract_kpis_from_layout(results_data, villes_dict, commune_db)

    # ── ENRICHISSEMENT SÉMANTIQUE (moteur contextuel) ──
    # Le moteur sémantique comprend QUE CHERCHER selon le label de chaque champ.
    # Il valide les valeurs (commune dans la DB, échelle au bon format, etc.)
    # et intègre les corrections apprises par l'utilisateur dans l'interface.
    if _SEMANTIC_ENGINE_AVAILABLE:
        try:
            script_dir_main = os.path.dirname(os.path.abspath(__file__))

            # Agréger toutes les détections brutes de toutes les pages
            all_raw = []
            for p in results_data:
                all_raw.extend(p.get('raw_detections', []))

            # Appliquer les corrections apprises depuis l'interface de validation
            correction_learner = CorrectionLearner(geometre_id, base_dir='writer_styles')
            for det in all_raw:
                txt = det.get('texte', '')
                field_type = identify_field_type(txt)
                if field_type != 'inconnu':
                    # Chercher la valeur associée sur la même ligne
                    val_after_colon = txt.split(':', 1)[-1].strip() if ':' in txt else ''
                    if val_after_colon:
                        learned = correction_learner.lookup(field_type, val_after_colon)
                        if learned:
                            print(f"  [SemanticEngine] Correction apprise : [{field_type}] '{val_after_colon}' → '{learned}'")
                            det['texte'] = txt.split(':')[0] + ': ' + learned

            # Analyse sémantique globale
            kpis_sem = process_with_semantic_context(
                all_raw, commune_db, 'inconnu', script_dir_main
            )

            # Fusion : le moteur sémantique enrichit les KPIs existants
            # UNIQUEMENT si la confiance est suffisante et que le KPI classique est inconnu
            for field_key, field_result in kpis_sem.items():
                if field_key.startswith('_'):  # Ignorer _alerts, _suggestions
                    continue
                if not isinstance(field_result, dict):
                    continue
                confiance = field_result.get('confiance', 0.0)
                valeur = field_result.get('valeur', '')
                if not valeur or valeur in ('', 'nan'):
                    continue

                # Mapper les clés sémantiques vers les clés KPI existantes
                kpi_key_map = {
                    'commune': 'commune',
                    'echelle': 'echelle',
                    'section': 'cadastre_section',
                    'parcelle': 'cadastre_parcelle',
                    'geometre': 'geometre',
                    'dossier': 'n_dossier',
                    'proprietaire': 'proprietaire',
                    'date': 'date',
                }
                kpi_key = kpi_key_map.get(field_key)
                if not kpi_key:
                    continue

                current_val = kpis.get(kpi_key, 'Inconnu')
                if current_val in ('Inconnu', '', None) and confiance >= 0.65:
                    kpis[kpi_key] = valeur
                    print(f"  [SemanticEngine] KPI '{kpi_key}' enrichi : '{valeur}' (confiance={confiance:.0%})")
                elif field_key == 'commune':
                    # Pour la commune : comparer les scores et prendre le meilleur
                    sem_score = int(confiance * 100)
                    if sem_score > kpis.get('commune_score', 0) and confiance >= 0.70:
                        kpis['commune'] = valeur
                        kpis['commune_score'] = sem_score
                        if 'code' in field_result:
                            kpis['commune_code'] = field_result['code']
                        print(f"  [SemanticEngine] Commune mise à jour : '{valeur}' ({sem_score}%)")

            # Alertes de cohérence
            for alert in kpis_sem.get('_alerts', []):
                print(f"  [SemanticEngine] ⚠️  {alert}")

        except Exception as e_sem:
            print(f"  [SemanticEngine] Erreur non bloquante : {e_sem}")

    # Mettre à jour le géomètre depuis les KPIs extraits (si trouvé)
    if kpis.get('geometre', 'Inconnu') != 'Inconnu':
        geometre_reel = re.sub(r'[^a-zA-ZÀ-ÿ0-9_\-]', '_', kpis['geometre'])[:30]
        if geometre_reel != geometre_id:
            print(f"  [Module5] Renommage profil géomètre : '{geometre_id}' → '{geometre_reel}'")
            ancien_path = writer_learner.data_path
            writer_learner.geometre_id = geometre_reel
            writer_learner.data_path = os.path.join(writer_learner.base_dir, f"{geometre_reel}.json")
            writer_learner.save()
            if os.path.exists(ancien_path) and ancien_path != writer_learner.data_path:
                os.remove(ancien_path)

    type_doc = classify_document(results_data, kpis)

    # ── EXPORT JSON ──
    json_path = os.path.join('outputs', f"{base_name}_hybride_resultats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"kpis": kpis, "type": type_doc, "pages": results_data}, f, ensure_ascii=False, indent=4)

    print("\n  -> Analyse spatiale des KPIs et Classification...")
    print(f"  -> Type détecté : {type_doc}")
    print(f"  -> KPIs Extraits : {kpis}")

    # ── SAUVEGARDE IMAGES ANNOTÉES ──
    first_page_output_path = "Non généré"
    for idx_ann, img_ann in enumerate(annotated_pages):
        output_img_path = os.path.join('outputs', f"{base_name}_page_{idx_ann+1}_hybride_v6_annote.jpg")
        cv2.imwrite(output_img_path, img_ann)
        if idx_ann == 0:
            first_page_output_path = output_img_path

    # ── EXPORT CSV / EXCEL ──
    lignes_csv: List[Dict[str, Any]] = []
    for page_data in results_data:
        page_num = page_data["page"]
        for det in page_data["detections"]:
            texte = det["texte"].replace('\n', ' ').strip()
            commune_finale = det.get("commune_row", "Inconnu")
            
            # Correction : Ne jamais écraser une case vide ou "Non identifiée" par la commune globale
            # Cela faussait le travail de validation en pré-remplissant des hallucinations !
            if det.get("ocr_brut_commune") == "[VIDE]" or det.get("confirmation_status") == "Vide":
                commune_finale = "[VIDE]"
            elif commune_finale == "Inconnu":
                # On ne remplace par le KPI global QUE si on est sûr que la case n'est pas vide
                # (Par exemple pour gérer les guillemets " )
                if det.get("texte") == '"':
                    commune_finale = kpis.get("commune", "Inconnu")
            
            lignes_csv.append({
                "ID_Ligne": det.get("id_ligne", "N/A"),
                "Type_Document": type_doc,
                "Commune_Doc": kpis.get("commune", "Inconnu"),
                "Commune_Ligne": commune_finale,
                "Confiance_Commune_%": det.get("confiance_commune", 0),
                "Confirmation_Status": det.get("confirmation_status", ""),
                "Methode_Commune": det.get("commune_match_info", {}).get("methode", ""),
                "OCR_brut_commune": det.get("ocr_brut_commune", ""),
                "Hypotheses_OCR": det.get("hypotheses_ocr", ""),
                "Geometre": kpis.get("geometre", "Inconnu"),
                "N_Dossier": kpis.get("n_dossier", "Inconnu"),
                "Ordre": kpis.get("ordre_document", "Inconnu"),
                "Echelle": kpis.get("echelle", "Inconnu"),
                "Page": page_num,
                "Texte_Extrait": texte,
                "Confiance": det.get("confiance", "N/A"),
                "Outil_Utilise": det.get("type_ocr", "N/A"),
                "Coordonnees_Bbox_xyxy": str(det.get("bbox", []))
            })

    df = pd.DataFrame(lignes_csv)
    csv_path = os.path.join('outputs', f"{base_name}_resultats.csv")
    excel_path = os.path.join('outputs', f"{base_name}_resultats.xlsx")

    if not df.empty:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=';')
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer_xl:
            df.to_excel(writer_xl, index=False, sheet_name='Resultats_OCR')
            df_kpi = pd.DataFrame([kpis])
            df_kpi.to_excel(writer_xl, index=False, sheet_name='Synthese_KPI')
        print(f"-> Résultat Excel : {excel_path}")
        print(f"-> Résultat CSV   : {csv_path}")
    else:
        print("-> Aucun texte détecté, pas de fichier Excel généré.")

    print(f"\nExtraction terminée pour {file_path}.")
    print(f"-> Résultat visuel : {first_page_output_path}")
    print(f"-> Résultat texte  : {json_path}")

    t_fichier_s = time.time() - t0_fichier
    print(f"--- Temps de traitement : {int(t_fichier_s // 60)} min {int(t_fichier_s % 60)} s ---")


if __name__ == "__main__":
    import time as _time_main
    start_time_global = _time_main.time()

    setup_directories()
    print("Dossiers 'inputs' et 'outputs' prêts.")

    fichiers_entree = [f for f in os.listdir('inputs') if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]

    if fichiers_entree:
        models_charges = load_models()

        commune_db = load_commune_db()
        villes_dict = [e['normalise'] for e in commune_db]

        processor_obj = models_charges[2]
        _commune_logits_processor = build_commune_logits_processor(commune_db, processor_obj)
        _commune_abbrev_map = build_abbreviation_map(commune_db)

        for fichier in fichiers_entree:
            document_test = os.path.join('inputs', fichier)
            print(f"\n======================================")
            print(f"Traitement du fichier : {document_test}")

            # ── ROUTAGE PRIORITAIRE : Plan vs Livret ──────────────────────────
            # On teste d'abord le nouveau classifier (plus précis).
            # Si indisponible, on repasse sur l'ancien is_modern_plan().

            est_un_plan = False
            if _PLAN_CLASSIFIER_AVAILABLE:
                est_un_plan = is_plan_document(document_test)
            elif _MODERN_PLAN_AVAILABLE:
                est_un_plan = is_modern_plan(document_test)

            if est_un_plan:
                # ── Pipeline PLANS ────────────────────────────────────────────
                if _PLAN_CLASSIFIER_AVAILABLE:
                    # === NOUVEAU PIPELINE SPATIALISÉ ===
                    type_plan = classify_plan(document_test)
                    print(f">> Type détecté : PLAN [{type_plan}]")
                    print(f">> Lancement pipeline spatialisé plan_classifier")

                    result_plan = process_plan(document_test, models=models_charges, commune_db=commune_db)

                    if result_plan.get("skipped"):
                        print(f">> Plan ignoré : {result_plan.get('raison')}")
                        continue

                    csv_plan = export_plan_to_csv(result_plan, output_dir='outputs')

                    base_plan = os.path.splitext(fichier)[0]
                    json_plan_path = os.path.join('outputs', f"{base_plan}_plan_{type_plan}.json")
                    with open(json_plan_path, 'w', encoding='utf-8') as _f:
                        json.dump(result_plan, _f, ensure_ascii=False, indent=4, default=str)
                    print(f"  >> JSON sauvegardé : {json_plan_path}")
                    print(f"  >> CSV  sauvegardé : {csv_plan}")

                else:
                    # === FALLBACK : ancien pipeline modern_plan_extractor ===
                    print(f">> Type détecté : PLAN MODERNE (DA/DMPC) [fallback]")
                    commune_db_plan = list(commune_db)
                    nationale = load_commune_db_nationale('communes_france.json')
                    noms_existants = {e['normalise'] for e in commune_db_plan}
                    ajouts = [e for e in nationale if e['normalise'] not in noms_existants]
                    commune_db_plan.extend(ajouts)

                    geometre_id_plan = re.sub(r'[^a-zA-ZÀ-ÿ0-9_\-]', '_',
                        re.sub(r'[_\-]?\d+$', '', os.path.splitext(fichier)[0]).strip() or 'inconnu'
                    )[:30]

                    result_plan = process_modern_plan(
                        document_test,
                        commune_db=commune_db_plan,
                        geometre_id=geometre_id_plan,
                    )
                    export_modern_plan_to_csv(result_plan, output_dir='outputs')
                    base_plan = os.path.splitext(fichier)[0]
                    json_plan_path = os.path.join('outputs', f"{base_plan}_plan_moderne.json")
                    with open(json_plan_path, 'w', encoding='utf-8') as _f:
                        json.dump(result_plan, _f, ensure_ascii=False, indent=4, default=str)
                    print(f"  >> JSON sauvegardé : {json_plan_path}")

            else:
                # ── LIVRETS HISTORIQUES : ignorés dans ce mode ────────────────
                print(f">> Type détecté : LIVRET HISTORIQUE (manuscrit) — IGNORÉ")
                print(f"   (Pour traiter les livrets, utiliser le pipeline hybride séparément.)")
                continue

        print(f"\n============================================================")
        tot_s = _time_main.time() - start_time_global
        print(f"⏱ TEMPS TOTAL D'EXÉCUTION : {int(tot_s // 60)} min {int(tot_s % 60)} s")
        print(f"============================================================")

    else:
        print("Le dossier 'inputs' est vide.")
        print("-> Veuillez placer un document (.pdf, .jpg, .png) dans le dossier 'inputs' et relancer.")
