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
from PIL import Image, ImageDraw, ImageFont
import torch
import re
from kraken import blla, rpred
from kraken.lib import models as kraken_models
import unicodedata
from spellchecker import SpellChecker

try:
    from rapidfuzz import process, fuzz
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
    # Tirets et apostrophes → espace
    sans_accents = re.sub(r"[-''`]", ' ', sans_accents)
    # Majuscules
    majuscules = sans_accents.upper()
    # Suppression ponctuation (on garde lettres, chiffres, espaces)
    propre = re.sub(r'[^A-Z0-9 ]', ' ', majuscules)
    # Collapse espaces multiples
    return ' '.join(propre.split())


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


# Compatibilité : alias pour l'ancien code qui appelle load_villes_dictionary()
# Retourne une simple liste de noms normalisés (pour correct_ocr_with_dict)
def load_villes_dictionary(filepath='villes_07_26.txt'):
    """Compatibilité ascendante — retourne la liste des noms normalisés."""
    db = load_commune_db()
    return [e['normalise'] for e in db]

try:
    from transformers import LogitsProcessor, LogitsProcessorList
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
        self.trie = {}
        for seq in commune_token_seqs:
            node = self.trie
            for tok in seq:
                if tok not in node:
                    node[tok] = {}
                node = node[tok]
            # EOS valide dès que la séquence est complète
            node[eos_token_id] = {}

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

    def __call__(self, input_ids, scores):
        for i in range(input_ids.shape[0]):
            gen_part = [
                tok for tok in input_ids[i].tolist()
                if tok != self.decoder_start_token_id
            ]
            valid_tokens = self._get_valid_next_tokens(gen_part)
            # Mettre -inf pour tous les tokens non autorisés
            mask = scores[i].clone().fill_(float('-inf'))
            for tok_id in valid_tokens:
                if 0 <= tok_id < scores.shape[-1]:
                    mask[tok_id] = scores[i][tok_id]
            scores[i] = mask
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


# Variable globale : on le construit UNE SEULE FOIS en début de traitement
_commune_logits_processor = None


def match_commune_ardeche(texte_ocr, commune_db):
    """
    Compatibilité : appelle match_commune_multi_hypotheses avec une seule hypothèse.
    """
    return match_commune_multi_hypotheses([(texte_ocr, 1.0)], commune_db)


def match_commune_multi_hypotheses(hypotheses, commune_db):
    """
    Matching de commune sur plusieurs hypothèses OCR pondérées.
    
    `hypotheses` : liste de tuples (texte_ocr:str, beam_prob:float)
       - texte_ocr  : texte lu par le modèle
       - beam_prob  : probabilité log-normalisée du faisceau (0 à 1)
    
    Stratégie :
      • Pour chaque hypothèse, on calcule un score fuzzy multi-scorer (WRatio, token_set, partial).
      • Le score final = fuzzy_score * (0.5 + 0.5 * beam_prob)  <- poids équilibré OCR/probabilité
      • On retourne la commune avec le meilleur score agrégé.
      • La "confiance" retournée est un % (0-100) clair pour l'utilisateur.
    
    Retourne :
        {"officiel": str, "code": str, "score": int, "brut": str, "hypotheses_ocr": str}
    """
    if not commune_db or not process:
        return {'officiel': 'Non identifiée', 'code': '', 'score': 0,
                'brut': hypotheses[0][0] if hypotheses else '', 'hypotheses_ocr': ''}

    noms_normalises = [e['normalise'] for e in commune_db]
    
    best_score   = 0
    best_idx     = 0
    best_brut    = hypotheses[0][0] if hypotheses else ''
    hyp_log_parts = []  # pour affichage debug Excel

    for texte_ocr, beam_prob in hypotheses:
        texte_norm = normaliser_pour_matching(texte_ocr)
        if not texte_norm or len(texte_norm) < 2:
            continue

        # --- Scorers fuzzy ---
        r_w = process.extractOne(texte_norm, noms_normalises, scorer=fuzz.WRatio)
        r_t = process.extractOne(texte_norm, noms_normalises, scorer=fuzz.token_set_ratio)
        r_p = process.extractOne(texte_norm, noms_normalises, scorer=fuzz.partial_ratio)

        # On collecte le meilleur score fuzzy pour cette hypothèse
        fuzzy_best   = 0
        fuzzy_idx    = 0
        if r_w:
            _, sw, iw = r_w
            if sw > fuzzy_best: fuzzy_best, fuzzy_idx = sw, iw
        if r_t:
            _, st, it = r_t
            if st > fuzzy_best: fuzzy_best, fuzzy_idx = st, it
        if r_p:
            _, sp, ip = r_p
            # partial_ratio sur-évalue les courts mots → pondération légère
            sp_adj = sp * 0.90
            if sp_adj > fuzzy_best: fuzzy_best, fuzzy_idx = sp_adj, ip

        # Score final pondéré par la confiance du faisceau
        # beam_prob ∈ [0,1] : 0.5 garantit que le fuzzy score reste dominant même si beam=0
        score_final = fuzzy_best * (0.5 + 0.5 * float(beam_prob))

        hyp_log_parts.append(f"'{texte_ocr}'({beam_prob:.2f})→{noms_normalises[fuzzy_idx]}({fuzzy_best:.0f}%)")

        if score_final > best_score:
            best_score = score_final
            best_idx   = fuzzy_idx
            best_brut  = texte_ocr

    meilleure = commune_db[best_idx]
    hyp_log   = " | ".join(hyp_log_parts)
    
    print(f"      [match_multi] {hyp_log} => '{meilleure['officiel']}' "
          f"(score agrégé {best_score:.1f})")
    
    return {
        'officiel':        meilleure['officiel'],
        'code':            meilleure['code'],
        'score':           int(best_score),
        'brut':            best_brut,
        'hypotheses_ocr':  hyp_log
    }


def correct_ocr_with_dict(texte, dictionnaire, seuil=88):
    """Correction générique d'un texte OCR contre un dictionnaire de noms normalisés.
    Utilisé pour les corrections non-communes (noms de lieux Drôme, etc.)."""
    if not dictionnaire or not process or len(texte.strip()) < 3:
        return texte

    if len(texte) > 30:
        return texte

    t_clean = normaliser_pour_matching(texte)

    chiffres = sum(c.isdigit() for c in t_clean)
    if chiffres > 2:
        return texte

    if len(t_clean) < 3:
        return texte

    result = process.extractOne(t_clean, dictionnaire, scorer=fuzz.WRatio)
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

    return yolo_model, easyocr_reader, processor, trocr_model, device, spell, pylaia_model

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
    """Prétraitement OpenCV Avancé pour aider l'OCR.
    Modes :
    - 'hybrid' : Équilibre pour Tesseract + TrOCR
    - 'htr'    : Optimisé pour l'écriture manuscrite (contraste local fort)
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
    clahe = cv2.createCLAHE(clipLimit=3.0 if mode=='htr' else 2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(blur)
    
    if mode == 'htr':
        # Binarisation adaptative plus agressive pour le manuscrit
        # Permet de mieux séparer l'encre fine des taches du papier
        thresh = cv2.adaptiveThreshold(gray_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 21, 10)
    else:
        # OTSU standard pour l'imprimé
        _, thresh = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    roi_color_clean = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2RGB)
    final_roi = cv2.bitwise_not(thresh)
    
    # Padding
    final_roi = cv2.copyMakeBorder(final_roi, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    return final_roi, roi_color_clean

def process_document(file_path, models):
    """Pipeline sans YOLO : Détection EasyOCR + Prétraitement + Double OCR (Imprimé/Manuscrit)"""
    # Note : On ignore yolo_model puisqu'on utilise le détecteur d'EasyOCR
    _, easyocr_reader, processor, trocr_model, device, spell = models
    
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
    kpis = {
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
    pivots = {
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
    all_lines = []
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
                                            for okw in other_kws:
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
                        next_line = all_lines[i+1]["texte"]
                        is_pivot = False
                        for other_kws in pivots.values():
                            for okw in other_kws:
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
                            if kpis["commune"] == "Inconnu" or score_candidat > kpis.get("commune_score", 0):
                                kpis["commune"] = valeur
                                kpis["commune_score"] = score_candidat
                                kpis["commune_code"] = res['code']
                        else:
                            # On garde la valeur si elle est plus pertinente que l'actuelle
                            if kpis[key] == "Inconnu" or (len(valeur) > 3 and kpis[key] == "Inconnu"):
                                kpis[key] = valeur

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


def classify_document(results_data, kpis=None):
    """Fonction identification du type de document utilisant les KPIs et le texte global."""
    texte_global = ""
    for page in results_data:
        for det in page["detections"]:
            texte_global += " " + det["texte"].lower()
            
    mots_cles = {
        "Document d'Arpentage (DMPC)": ["arpentage", "dmpc", "chemise verte", "division", "document de modification", "arpentage"],
        "Plan de Bornage": ["bornage", "reconnaissance", "limites", "amiable", "pv", "procès-verbal", "borné ce jour", "reconnaissance de limites"],
        "Plan Topographique": ["topographique", "etat des lieux", "releve", "courbes de niveau", "levé topographique"],
        "Plan de Division": ["division", "lotissement", "création de lots"]
    }
    
    scores = {k: 0 for k in mots_cles}
    for type_doc, keywords in mots_cles.items():
        for kw in keywords:
            if kw in texte_global:
                scores[type_doc] += 1
    
    # Renforcement via KPIs
    if kpis:
        if kpis.get("ordre_document") != "Inconnu":
            scores["Document d'Arpentage (DMPC)"] += 2
        if "dmpc" in str(kpis.get("n_dossier", "")).lower():
            scores["Document d'Arpentage (DMPC)"] += 2

    meilleur_type = max(scores, key=scores.get)
    if scores[meilleur_type] == 0:
        return "Type Inconnu"
    return meilleur_type


def process_document_hybrid(file_path, models, villes_dict, commune_db):
    """Architecture Dual-OCR (Imprimé vs Hybride) :
    1. Tesseract Global : Scan de toute la page pour extraire 100% de l'imprimé.
    2. Fallback Kraken : Segmentation et lecture (PyLaia) uniquement sur les zones denses/manuscrites illisibles par Tesseract.
    """
    try:
        from kraken import blla
    except ImportError:
        print("Erreur : Kraken n'est pas installé. Lancez 'pip install kraken'.")
        return

    # Configuration du chemin Tesseract pour Windows via WSL
    # (On assume que l'utilisateur a configuré run_tesseract_windows correctement)
    
    _, _, processor, trocr_model, device, spell, pylaia_model = models
    
    images = read_document(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    results_data = []
    annotated_pages = []
    
    for idx, img_cv in enumerate(images):
        print(f"\nTraitement de la page/image {idx + 1} (Dual-Pipeline)...")
        img_annotated = img_cv.copy()
        page_results = []
        
        # --- DETECTION PREALABLE DE LA COLONNE COMMUNE ---
        commune_column_x = None # (x1, x2)
        candidate_headers = []
        
        # ETAPE 1 : TESSERACT GLOBAL
        gray_full = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        custom_tess_config = r'--oem 3 --psm 3 -l fra'
        tess_df = run_tesseract_windows(gray_full, custom_tess_config, output_type='tsv')
        
        mask_tesseract_lu = np.zeros(img_cv.shape[:2], dtype=np.uint8)
        
        if not tess_df.empty:
            tess_df = tess_df.dropna(subset=['text'])
            # Groupement par lignes tesseract
            for _, row in tess_df.iterrows():
                conf = int(row['conf']) if str(row['conf']).isdigit() else -1
                texte = str(row['text']).strip()
                if conf > 65 and texte != '':
                    x, y, w, h = row['left'], row['top'], row['width'], row['height']
                    cv2.rectangle(mask_tesseract_lu, (x-2, y-2), (x+w+2, y+h+2), 255, -1)
                    
                    # Correction et ajout
                    texte_corr = correct_cadastral_rules(texte)
                    page_results.append({
                        "bbox": [x, y, x+w, y+h],
                        "texte": texte_corr,
                        "confiance": f"{conf}%",
                        "type_ocr": "Tesseract (Imprimé)"
                    })
                    
                    # Détection de l'en-tête de colonne COMMUNE
                    # FILTRE Y STRICT : l'en-tête COMMUNE est TOUJOURS dans les 250 premiers pixels
                    # On exclut les cellules du corps ("JA Commune de...", "communale", etc.)
                    header_zone = y < 250 and y + h < 300
                    if "commune" in texte_corr.lower() and h < 100 and header_zone:
                        candidate_headers.append((x, x+w, y, y+h))
                        cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 3) # Highlight vert pour l'en-tête
                        print(f"  -> En-tête COMMUNE détecté (Tesseract) X=[{x},{x+w}] Y=[{y},{y+h}]")

        # ETAPE 2 : KRAKEN + TrOCR
        img_reste = img_cv.copy()
        img_reste[mask_tesseract_lu == 255] = (255, 255, 255)
        bounds = blla.segment(Image.fromarray(cv2.cvtColor(img_reste, cv2.COLOR_BGR2RGB)))
        
        lines_kraken = bounds.lines if hasattr(bounds, 'lines') else []
        for line in lines_kraken:
            poly = np.array(line.boundary, np.int32)
            x, y, w, h = cv2.boundingRect(poly)
            roi = img_cv[max(0, y-10):y+h+10, max(0, x-10):x+w+10]
            if roi.size == 0: continue
            
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
            
            # Détection de l'en-tête de colonne COMMUNE (version manuscrite)
            txt_norm_header = normaliser_pour_matching(txt_trocr)
            # FILTRE Y STRICT : l'en-tête est dans les 250 premiers pixels seulement
            header_zone_trocr = y < 250 and (y + h) < 300
            # Seuil 90 pour éviter les faux positifs dans le corps du tableau
            if process and (fuzz.WRatio(txt_norm_header, "COMMUNE") > 90 or "commune" == txt_trocr.lower().strip()) and h < 100 and header_zone_trocr:
                candidate_headers.append((x, x+w, y, y+h))
                cv2.rectangle(img_annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                print(f"  -> En-tête COMMUNE détecté (TrOCR) X=[{x},{x+w}] Y=[{y},{y+h}]")

        # --- DÉDUCTION DES COLONNES "COMMUNE" ---
        commune_columns_x = []
        
        # 1. Utiliser les en-têtes "Commune" détectés par l'OCR comme source de vérité ABSOLUE
        # L'utilisateur a confirmé que "le rectangle au niveau du mot commune est bon"
        for ref_x1, ref_x2, ref_y1, ref_y2 in candidate_headers:
            # On définit la colonne géométriquement sous le mot "Commune"
            commune_columns_x.append((ref_x1 - 80, ref_x2 + 150))
            
        # 2. Si un en-tête a été raté (ex: brouillé par l'écriture cursive), on complète via la Densité X
        is_double_page = img_cv.shape[1] > 2000
        
        # SÉPARATION DYNAMIQUE DE LA RELIURE (Double Page)
        # Ne pas utiliser betement img_cv.shape[1] / 2 car le scan n'est jamais parfaitement centré
        x_centers = [ (r['bbox'][0] + r['bbox'][2])/2.0 for r in page_results if len(r['texte']) > 1 ]
        
        midpoint = img_cv.shape[1] / 2
        if is_double_page and x_centers:
            # Trier les x_centers et trouver le plus grand "trou" (la marge centrale)
            x_sorted = sorted(x_centers)
            max_gap = 0
            for i in range(1, len(x_sorted)):
                gap = x_sorted[i] - x_sorted[i-1]
                # Le trou de la reliure doit logiquement être proche du centre (entre 30% et 70% de la largeur)
                if gap > max_gap and (img_cv.shape[1]*0.3) < x_sorted[i] < (img_cv.shape[1]*0.7):
                    max_gap = gap
                    midpoint = x_sorted[i-1] + (gap / 2.0)
                    
        # Vérifions si on a bien une colonne par "moitié" de page
        has_left_col = any((cx1+cx2)/2 < midpoint for cx1, cx2 in commune_columns_x)
        has_right_col = any((cx1+cx2)/2 >= midpoint for cx1, cx2 in commune_columns_x) if is_double_page else True
        
        if not has_left_col or not has_right_col:
            from collections import defaultdict
            bins = defaultdict(int)
            for cx in x_centers: bins[int(cx / 50) * 50] += 1
            sorted_bins = sorted(bins.items())
            keys, vals = [k for k, v in sorted_bins], [v for k, v in sorted_bins]
            
            peaks = []
            for i in range(len(keys)):
                if vals[i] < 10: continue
                is_max = True
                for j in range(len(keys)):
                    if i != j and abs(keys[i] - keys[j]) <= 80 and vals[j] > vals[i]:
                        is_max = False
                        break
                if is_max: peaks.append(keys[i])
                
            halves_peaks = []
            if is_double_page:
                if not has_left_col: halves_peaks.append([p for p in peaks if p < midpoint])
                if not has_right_col: halves_peaks.append([p for p in peaks if p >= midpoint])
            elif not has_left_col:
                halves_peaks.append(peaks)
                
            for half_peaks in halves_peaks:
                half_peaks.sort()
                # Validation intelligente du pic : lequel contient réellement des communes ?
                if len(half_peaks) >= 2:
                    best_peak_x = half_peaks[min(2, len(half_peaks)-1)] # Fallback 3ème si possible
                    max_commune_count = -1
                    
                    for p_x in half_peaks:
                        # On compte combien de mots dans cette bande verticale [p_x-80, p_x+120] 
                        # matchent un peu la base de communes.
                        count_matches = 0
                        for r in page_results:
                            rx = (r['bbox'][0] + r['bbox'][2]) / 2
                            if (p_x - 80) <= rx <= (p_x + 120):
                                # On ne checke que le texte un peu long pour éviter le bruit
                                if len(r['texte']) > 3:
                                    res_c = match_commune_ardeche(r['texte'], commune_db)
                                    if res_c['score'] > 60: count_matches += 1
                        
                        if count_matches > max_commune_count:
                            max_commune_count = count_matches
                            best_peak_x = p_x
                    
                    p_x = best_peak_x
                    commune_columns_x.append((p_x - 100, p_x + 150))
                    print(f"  -> Colonne Commune identifiée (MATCHES: {max_commune_count}) : X ~ {p_x}")

        # Dessiner les colonnes vertes pour debug visuel
        for cx1, cx2 in commune_columns_x:
            cv2.line(img_annotated, (int(cx1), 0), (int(cx1), img_cv.shape[0]), (0, 255, 0), 2)
            cv2.line(img_annotated, (int(cx2), 0), (int(cx2), img_cv.shape[0]), (0, 255, 0), 2)

        # --- REGROUPEMENT PAR LIGNES ET MATCHING PAR PAGE ---
        # On définit les "sous-pages" pour traiter séparément le côté gauche et droit (reliure)
        sub_pages_detections = []
        if is_double_page:
            left_p  = [r for r in page_results if (r['bbox'][0] + r['bbox'][2])/2.0 < midpoint]
            right_p = [r for r in page_results if (r['bbox'][0] + r['bbox'][2])/2.0 >= midpoint]
            sub_pages_detections = [("GAUCHE", left_p), ("DROITE", right_p)]
        else:
            sub_pages_detections = [("PAGE", page_results)]

        for side_name, sub_results in sub_pages_detections:
            if not sub_results: continue
            
            # REGROUPEMENT PAR LIGNES
            sub_results.sort(key=lambda x: (x["bbox"][1] + x["bbox"][3]) / 2.0)
            grouped_results = []
            ligne_courante = [sub_results[0]]
            for i in range(1, len(sub_results)):
                by = (sub_results[i]["bbox"][1] + sub_results[i]["bbox"][3]) / 2.0
                ys_courants = [(b["bbox"][1] + b["bbox"][3]) / 2.0 for b in ligne_courante]
                ly_moy = sum(ys_courants) / len(ys_courants)
                if abs(by - ly_moy) < 25: ligne_courante.append(sub_results[i])
                else:
                    grouped_results.append(ligne_courante)
                    ligne_courante = [sub_results[i]]
            grouped_results.append(ligne_courante)

            for line_idx, grp in enumerate(grouped_results):
                grp.sort(key=lambda x: x["bbox"][0])
                txt_line = " | ".join([b["texte"] for b in grp if b["texte"]])
                
                # --- MATCHING COMMUNE PAR LIGNE ---
                res_commune = {'officiel': 'Inconnu', 'score': 0}
                commune_bbox = None
                
                # 1. Vérifier si un segment de la ligne est dans la colonne COMMUNE
                for det in grp:
                    dx1, dy1, dx2, dy2 = det["bbox"]
                    dcx = (dx1 + dx2) / 2
                    
                    is_in_column = False
                    for cx1, cx2 in commune_columns_x:
                        margin = 25
                        if (cx1 - margin) <= dcx <= (cx2 + margin):
                            is_in_column = True
                            break
                    
                    if is_in_column and len(det["texte"].strip()) >= 2:
                        # === DÉCODAGE MULTI-HYPOTHÈSES (Beam Search Contraint) ===
                        hypotheses = [(det["texte"], 1.0)]
                        if trocr_model is not None and processor is not None:
                            try:
                                roi_commune = img_cv[max(0, dy1-5):min(img_cv.shape[0], dy2+5),
                                                     max(0, dx1-5):min(img_cv.shape[1], dx2+5)]
                                if roi_commune.size > 0:
                                    _, roi_clean_c = preprocess_roi_for_ocr(roi_commune, mode='htr')
                                    pv = processor(Image.fromarray(roi_clean_c), return_tensors="pt").pixel_values.to(device)
                                    
                                    lp_list = LogitsProcessorList([_commune_logits_processor]) if _commune_logits_processor else None
                                    out = trocr_model.generate(
                                        pv, max_new_tokens=20, num_beams=8, num_return_sequences=5,
                                        logits_processor=lp_list, output_scores=True, return_dict_in_generate=True
                                    )
                                    seqs  = processor.batch_decode(out.sequences, skip_special_tokens=True)
                                    if hasattr(out, 'sequences_scores'):
                                        import math
                                        raw_scores = out.sequences_scores.cpu().tolist()
                                        min_s = min(raw_scores)
                                        adjusted = [s - min_s for s in raw_scores]
                                        exp_s = [math.exp(min(a, 32)) for a in adjusted]
                                        total = sum(exp_s)
                                        norm_probs = [e / total for e in exp_s]
                                        hypotheses = [(correct_cadastral_rules(s), p) for s, p in zip(seqs, norm_probs)]
                            except Exception as e_beam:
                                hypotheses = [(det["texte"], 1.0)]
                        
                        match = match_commune_multi_hypotheses(hypotheses, commune_db)
                        if match['score'] > 10: 
                            res_commune = match
                            det["texte"] = f"[COL_COMMUNE] -> {match['officiel']}"
                            commune_bbox = [dx1, dy1, dx2, dy2]

                txt_line = " | ".join([b["texte"] for b in grp if b["texte"]])
                final_commune = res_commune['officiel'] if (commune_columns_x and res_commune['score'] > 10) else "Inconnu"
                if final_commune != "Inconnu":
                    txt_line = f"[COMMUNE: {final_commune}] " + txt_line
                    
                bbox_line = [min(b["bbox"][0] for b in grp), min(b["bbox"][1] for b in grp), 
                             max(b["bbox"][2] for b in grp), max(b["bbox"][3] for b in grp)]
                             
                line_id = f"{side_name[0]}{line_idx + 1}" # Ex: G1, D1
                x1, y1, x2, y2 = bbox_line
                cv2.line(img_annotated, (x1, y2), (x2, y2), (255, 100, 100), 1)
                
                # Détermination de l'abscisse d'ancrage
                anchor_x = x1
                if commune_columns_x:
                    mid_l = (x1 + x2) / 2
                    best_c = min(commune_columns_x, key=lambda c: abs((c[0]+c[1])/2 - mid_l))
                    anchor_x = int(best_c[0])

                if final_commune != "Inconnu" and commune_bbox:
                    cx1, cy1, cx2, cy2 = commune_bbox
                    cv2.rectangle(img_annotated, (cx1, cy1), (cx2, cy2), (0, 0, 255), 3)
                    label_display = f"{line_id}: {final_commune}"
                    (tw, th), _ = cv2.getTextSize(label_display, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_annotated, (anchor_x, cy1-th-10), (anchor_x+tw+10, cy1), (0, 0, 255), -1)
                    cv2.putText(img_annotated, label_display, (anchor_x+5, cy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    short_txt = txt_line[:40] + "..." if len(txt_line) > 40 else txt_line
                    label_display = f"{line_id}: {short_txt}"
                    (tw, th), _ = cv2.getTextSize(label_display, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(img_annotated, (anchor_x, y1-th-5), (anchor_x+tw+5, y1), (255, 120, 120), -1)
                    cv2.putText(img_annotated, label_display, (anchor_x+2, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                results_fusionnes.append({
                    "id_ligne": line_id,
                    "texte": txt_line,
                    "type_ocr": "+".join(list(set([b["type_ocr"] for b in grp]))),
                    "confiance": " | ".join([str(b.get("confiance", "N/A")) for b in grp]),
                    "confiance_commune": res_commune.get('score', 0),
                    "ocr_brut_commune": res_commune.get('brut', ''),
                    "hypotheses_ocr": res_commune.get('hypotheses_ocr', ''),
                    "bbox": bbox_line,
                    "commune_row": final_commune,
                    "commune_match_info": res_commune
                })

        annotated_pages.append(img_annotated)
        results_data.append({"page": idx+1, "detections": results_fusionnes, "raw_detections": page_results})

    # KPIs globaux
    kpis = extract_kpis_from_layout(results_data, villes_dict, commune_db)
    type_doc = classify_document(results_data, kpis)
    
    # Export final
    json_path = os.path.join('outputs', f"{base_name}_hybride_resultats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"kpis": kpis, "type": type_doc, "pages": results_data}, f, ensure_ascii=False, indent=4)
        
    print("\n  -> Analyse spatiale des KPIs et Classification...")
    print(f"  -> Type détecté : {type_doc}")
    print(f"  -> KPIs Extraits : {kpis}")
    
    # === AFFICHAGE DES KPIs VALIDES SUR LA PREMIERE PAGE (PIL pour les accents) ===
    # Désactivé à la demande de l'utilisateur (entête en haut à gauche en vert)
    if False and annotated_pages and kpis:
        try:
            # Travailler sur une copie explicite pour eviter les problemes de reference
            first_page_cv = np.array(annotated_pages[0], dtype=np.uint8)
            h_img, w_img = first_page_cv.shape[:2]
            
            kpis_valides = {k: v for k, v in kpis.items() if v and v != "Inconnu"}
            
            # Parametres du panel
            font_size_titre = 22
            font_size_commune = 36
            font_size_normal = 20
            lineh_titre = 32
            lineh = 45
            panel_h = lineh_titre + 10 + lineh * max(len(kpis_valides), 1) + 20
            panel_w = min(w_img - 10, 1300)
            
            # Dessiner le fond sombre directement (sans addWeighted qui peut poser probleme)
            first_page_cv[5:panel_h, 5:panel_w] = (first_page_cv[5:panel_h, 5:panel_w] * 0.2).astype(np.uint8)
            # Bordure verte
            cv2.rectangle(first_page_cv, (5, 5), (panel_w, panel_h), (0, 220, 0), 3)
            
            # Convertir en PIL RGB pour le texte Unicode
            img_pil = Image.fromarray(cv2.cvtColor(first_page_cv, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # Charger les polices
            try:
                font_titre = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size_titre)
                font_commune = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size_commune)
                font_normal = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size_normal)
            except Exception as fe:
                print(f"  [WARN] Police non trouvee: {fe}, utilisation police par défaut")
                font_titre = ImageFont.load_default()
                font_commune = font_titre
                font_normal = font_titre
            
            # Titre : type de document
            draw.text((15, 10), f"TYPE: {type_doc}", font=font_titre, fill=(200, 200, 200))
            
            y_offset = 10 + lineh_titre
            for key, value in kpis_valides.items():
                label = key.upper().replace('_', ' ')
                texte = f"{label}: {value}"
                if key == "commune":
                    draw.text((15, y_offset), texte, font=font_commune, fill=(0, 255, 80))
                    y_offset += lineh + 5
                else:
                    draw.text((15, y_offset), texte, font=font_normal, fill=(255, 255, 255))
                    y_offset += lineh
            
            # Reconvertir en OpenCV BGR et reassigner dans la liste
            annotated_pages[0] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            print(f"  -> Panel KPI dessiné sur la page 1 ({panel_w}x{panel_h}px)")
            
        except Exception as e:
            print(f"  [ERREUR] Impossible de dessiner le panel KPI: {e}")

    # Save all annotated images
    first_page_output_path = "Non généré"
    for idx, img_ann in enumerate(annotated_pages):
        output_img_path = os.path.join('outputs', f"{base_name}_page_{idx+1}_hybride_v5_annote.jpg")
        cv2.imwrite(output_img_path, img_ann)
        if idx == 0:
            first_page_output_path = output_img_path
    # Aplatir les résultats pour un tableau propre
    lignes_csv = []
    for page_data in results_data:
        page_num = page_data["page"]
        for det in page_data["detections"]:
            # Nettoyer les sauts de lignes pour Excel
            texte = det["texte"].replace('\n', ' ').strip()
            
            # On utilise la commune_row si elle existe, sinon le KPI global
            commune_finale = det.get("commune_row", "Inconnu")
            if commune_finale == "Inconnu":
                commune_finale = kpis.get("commune", "Inconnu")
                
            # Fusionner les métadonnées avec chaque ligne pour l'export
            ligne_data = {
                "ID_Ligne": det.get("id_ligne", "N/A"),
                "Type_Document": type_doc,
                "Commune_Doc": kpis["commune"],
                "Commune_Ligne": commune_finale,
                "Confiance_Commune_%": det.get("confiance_commune", 0),
                "OCR_brut_commune": det.get("ocr_brut_commune", ""),
                "Geometre": kpis["geometre"],
                "N_Dossier": kpis["n_dossier"],
                "Ordre": kpis["ordre_document"],
                "Echelle": kpis["echelle"],
                "Page": page_num,
                "Texte_Extrait": texte,
                "Confiance": det.get("confiance", "N/A"),
                "Outil_Utilise": det["type_ocr"],
                "Coordonnees_Bbox_xyxy": str(det["bbox"])
            }
            lignes_csv.append(ligne_data)
            
    df = pd.DataFrame(lignes_csv)
    
    csv_path = os.path.join('outputs', f"{base_name}_resultats.csv")
    excel_path = os.path.join('outputs', f"{base_name}_resultats.xlsx")
    
    if not df.empty:
        # Enregistrement
        df.to_csv(csv_path, index=False, encoding='utf-8-sig', sep=';')
        
        # Pour Excel, on peut faire un format plus sympa avec un onglet Summary ?
        # Pour rester simple, on garde un seul onglet mais avec les KPIs en premières colonnes
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Resultats_OCR')
            # Optionnel: on pourrait ajouter un onglet 'Synthese'
            df_kpi = pd.DataFrame([kpis])
            df_kpi.to_excel(writer, index=False, sheet_name='Synthese_KPI')
            
        print(f"-> Résultat Excel : {excel_path}")
        print(f"-> Résultat CSV   : {csv_path}")
    else:
        print("-> Aucun texte détecté, pas de fichier Excel généré.")
        
    print(f"\nExtraction terminée pour {file_path}.")
    print(f"-> Résultat visuel : {first_page_output_path}")
    print(f"-> Résultat texte  : {json_path}")

if __name__ == "__main__":
    setup_directories()
    print("Dossiers 'inputs' et 'outputs' prêts.")
    
    # === POUR TESTER ===
    fichiers_entree = [f for f in os.listdir('inputs') if f.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
    
    if fichiers_entree:
        # On charge les modèles
        models_charges = load_models()
        
        # Chargement de la base de données COMPLETE (avec codes INSEE)
        commune_db = load_commune_db() 
        # On garde villes_dict pour la compatibilité correction orthographe
        villes_dict = [e['normalise'] for e in commune_db]
        
        # Initialisation du processeur de décodage contraint (TRIE)
        processor = models_charges[2] # 3ème élément : TrOCRProcessor
        _commune_logits_processor = build_commune_logits_processor(commune_db, processor)
        
        for fichier in fichiers_entree:
            document_test = os.path.join('inputs', fichier)
            print(f"\n======================================")
            print(f"Traitement du fichier : {document_test}")
            
            # --- METHODE 3 : HYBRIDE KRAKEN + TESSERACT + TrOCR ---
            print(">> Lancement Pipeline Hybride (Segmentation Kraken + OCR Tesseract/TrOCR + Commune Matching)")
            process_document_hybrid(document_test, models_charges, villes_dict, commune_db)
            
    else:
        print("Le dossier 'inputs' est vide.")
        print("-> Veuillez placer un document (.pdf, .jpg, .png) dans le dossier 'inputs' et relancer le script.")

        