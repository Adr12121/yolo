"""
patch_non_dmpc.py — Applique les améliorations de détection non-DMPC sur plan_classifier.py
Usage : python patch_non_dmpc.py
"""
import re, sys, os

TARGET = os.path.join(os.path.dirname(__file__), "plan_classifier.py")

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

original_len = len(src)
print(f"[Patch] Fichier chargé : {original_len} caractères, {src.count(chr(10))} lignes")

# ────────────────────────────────────────────────────────────────────────────
# PATCH 1 : Ajouter constantes VLM adaptatives après GEOMETRES_CONNUS
# ────────────────────────────────────────────────────────────────────────────
ANCHOR_1 = 'GEOMETRES_CONNUS = ["DUPUY", "HARROIS", "RACAT", "SERRET", "CEYTE", "BARRIAL", "ROBERT"]'
INSERT_1 = '''
# ── Modèle VLM adaptatif selon le type de document ───────────────────
# llama3.2-vision (11B) est plus précis mais plus lent.
# On l'utilise pour les documents non-standards où llava (7B) hallucine.
_OLLAMA_MODEL_FOR_TYPE = {
    "DMPC":    "llava",             # Formulaire structuré → llava rapide suffit
    "PVa":     "llama3.2-vision",   # Texte libre tapé → meilleure compréhension sémantique
    "PLa":     "llava",             # Similaire au DMPC
    "CROQUIS": "llama3.2-vision",   # Documents anciens → llama3.2-vision obligatoire
    "GENERIC": "llama3.2-vision",   # Inconnu → prendre le meilleur
    "DEFAULT": "llava",
}

# ── Prompts VLM spécialisés par type et par champ ────────────────────
_VLM_PROMPTS_BY_TYPE = {
    "DMPC": {
        "commune":  "Quel est le nom de la commune écrit dans cette case ? Réponds uniquement par le nom de la commune, sans phrase.",
        "section":  "Quelle est la lettre de section cadastrale dans cette case ? Réponds uniquement par la lettre (ex: A, B, AB).",
        "feuille":  "Quel est le numéro de feuille dans cette case ? Réponds uniquement par le numéro.",
        "n_ordre":  "Quel est le numéro d'ordre ou numéro DA dans cette case ? Réponds uniquement par le numéro.",
        "geometre": "Quel est le nom du géomètre ou cabinet écrit dans cette case ? Réponds uniquement par le nom, sans phrase.",
        "date":     "Quelle est la date écrite dans cette case ? Réponds uniquement par la date (format JJ/MM/AAAA).",
        "_default": "Extrais UNIQUEMENT la valeur écrite dans cette case. NE FAIS AUCUNE PHRASE. Réponds juste par le texte lu. Si illisible, réponds \'vide\'.",
    },
    "PVa": {
        "commune":  "Ce document est un procès-verbal de bornage. Quel est le nom de la commune ? Il peut apparaître sous la forme \'commune de X\', \'sur le territoire de X\'. Réponds uniquement par le nom.",
        "date":     "Ce document est un acte de géomètre. Cherche la date à laquelle il a été dressé. Elle apparaît souvent sous \'Fait à [ville], le [date]\'. Réponds uniquement par la date (JJ/MM/AAAA ou JJ mois AAAA).",
        "geometre": "Ce document est un procès-verbal de géomètre-expert. Quel est le nom du géomètre qui a dressé ce document ? Il apparaît après \'le soussigné géomètre-expert\' ou \'dressé par\'. Réponds uniquement par le nom.",
        "section":  "Dans ce procès-verbal, quelle est la section cadastrale ? Elle apparaît sous la forme \'section [lettre]\'. Réponds uniquement par la lettre.",
        "proprietaires_anciens": "Qui est le demandeur ou propriétaire actuel ? Il apparaît après \'à la demande de\' ou \'requis par\'. Réponds uniquement par le nom.",
        "proprietaires_nouveaux": "Qui est le nouveau propriétaire ou bénéficiaire ? Il apparaît après \'au profit de\' ou \'acquis par\'. Réponds uniquement par le nom.",
        "_default": "Ce document est un acte géomètre. Extrais UNIQUEMENT la valeur de \'{field}\'. NE FAIS AUCUNE PHRASE. Réponds juste par le texte lu. Si illisible, réponds \'vide\'.",
    },
    "CROQUIS": {
        "commune":  "Ce document est un ancien plan cadastral ou croquis. Le nom de la commune peut être en tampon (avec des points entre les lettres, ex: S.A.I.N.T = SAINT). Quel est le nom de la commune ? Réponds uniquement par le nom.",
        "date":     "Ce document est un ancien plan. La date peut être manuscrite ou tamponnée. Cherche une date de dressé ou signé. Réponds uniquement par la date.",
        "geometre": "Ce document est un ancien plan cadastral. Le géomètre est identifié par son tampon ou signature. Son nom peut être en tampon avec des points entre les lettres — ignore ces points. Quel est le nom ? Réponds uniquement par le nom.",
        "section":  "Dans ce plan cadastral, quelle est la section cadastrale ? Souvent une lettre manuscrite ou tamponnée. Réponds uniquement par la lettre.",
        "echelle":  "Dans ce plan, quelle est l'échelle ? Format \'1/500\', \'1/1000\', \'1/2000\'. Réponds uniquement par l'échelle.",
        "_default": "Ce document est un ancien plan cadastral. Le texte peut être manuscrit ou en tampon. Si tu vois des points entre les lettres (ex: A.U.B.E.N.A.S), lis le mot sans les points. Extrais UNIQUEMENT la valeur de \'{field}\'. Si illisible, réponds \'vide\'.",
    },
    "GENERIC": {
        "_default": "Tu es un expert géomètre. Ce document est un plan ou acte cadastral. Extrais UNIQUEMENT la valeur de \'{field}\'. NE FAIS AUCUNE PHRASE. Réponds juste par la valeur brute. Si illisible, réponds \'vide\'.",
    },
    "DEFAULT": {
        "_default": "Extrais UNIQUEMENT la valeur de \'{field}\' sur l\'image. NE FAIS AUCUNE PHRASE. Réponds juste par le texte lu. Si illisible, réponds \'vide\'.",
    },
}

def _get_vlm_model(type_plan: str) -> str:
    """Retourne le modèle Ollama optimal pour un type de document donné."""
    return _OLLAMA_MODEL_FOR_TYPE.get(type_plan, _OLLAMA_MODEL_FOR_TYPE["DEFAULT"])

def _get_vlm_prompt(type_plan: str, field: str) -> str:
    """Retourne le prompt VLM spécialisé pour un type de document et un champ donnés."""
    prompts_for_type = _VLM_PROMPTS_BY_TYPE.get(type_plan, _VLM_PROMPTS_BY_TYPE["DEFAULT"])
    prompt = prompts_for_type.get(field, prompts_for_type.get("_default", _VLM_PROMPTS_BY_TYPE["DEFAULT"]["_default"]))
    return prompt.replace("{field}", field)
'''

if ANCHOR_1 in src:
    src = src.replace(ANCHOR_1, ANCHOR_1 + "\n" + INSERT_1, 1)
    print("[Patch 1] OK : constantes VLM adaptatives ajoutées")
else:
    print("[Patch 1] SKIP : ancre non trouvée")

# ────────────────────────────────────────────────────────────────────────────
# PATCH 2 : _clean_vlm_response helper + refactoriser _extract_with_vlm
#           Remplacer "model": "llava" par _get_vlm_model(type_plan)
#           et le prompt hardcodé par _get_vlm_prompt(type_plan, field)
# ────────────────────────────────────────────────────────────────────────────

# 2a : Ajouter _clean_vlm_response juste avant _extract_with_vlm
ANCHOR_2A = 'def _extract_with_vlm(img_bgr'
CLEAN_FN = '''def _clean_vlm_response(val: str) -> str:
    """Nettoyage commun des réponses VLM : supprime les phrases parasites."""
    import re as _re
    PREFIXES = [
        r"la photo montre.*?(?:sont|est|:)\\s*",
        r"l'image montre.*?(?:sont|est|:)\\s*",
        r"this image (?:shows|contains|depicts).*?(?::,)\\s*",
        r"il semble.*?:\\s*",
        r"les nombres.*?(?:sont|est|:)\\s*",
        r"le texte sur l'?image est", r"le texte dans l'?image est",
        r"le texte correspondant .*? est", r"il s'agit de",
        r"ce document", r"voici le texte", r"texte lu",
        r"le texte lu est", r"le texte est", r"texte:", r"valeur",
        r"je peux voir", r"je vois", r"i can see", r"the text (?:reads|says|is)",
        r"based on the image", r"in the image",
    ]
    pattern = r"(?i)^(?:.*?(?:(?:est|:|: |\\s)?" + r"|".join(PREFIXES) + r")\\s*:?\\s*)+"
    val = _re.sub(pattern, "", val).strip()
    val = _re.sub(r"(?i)^(?:texte|le texte|la valeur)?\\s*(?:dans|sur)?\\s*(?:l'?image|ce document)?\\s*(?:est|:)?\\s*", "", val).strip()
    val = val.replace('"', '').replace("'", "").strip(" '.:,")
    return val


'''

if ANCHOR_2A in src and '_clean_vlm_response' not in src:
    src = src.replace(ANCHOR_2A, CLEAN_FN + ANCHOR_2A, 1)
    print("[Patch 2a] OK : _clean_vlm_response ajouté")
elif '_clean_vlm_response' in src:
    print("[Patch 2a] SKIP : _clean_vlm_response déjà présent")
else:
    print("[Patch 2a] SKIP : ancre non trouvée")

# 2b : Remplacer les hardcodes "model": "llava" dans _extract_with_vlm (crop)
# On repère la section spécifique des crops et on remplace le bloc de prompt + model
OLD_CROP_PROMPT_BLOCK = '''            if field == "geometre":
                prompt = f"Extrais UNIQUEMENT le nom de la personne ou du cabinet écrit sur l'image. NE FAIS AUCUNE PHRASE. Ne dis pas 'L'image montre' ou 'Le texte est'. Réponds juste avec le nom brut. Si c'est illisible, réponds 'vide'."
            elif field in ["parcelles", "proprietaires_anciens", "proprietaires_nouveaux"]:
                prompt = f"Extrais UNIQUEMENT les mots ou numéros utiles liés à '{field}'. NE FAIS AUCUNE PHRASE DESCRIPTIVE. Ne dis pas 'La photo montre' ni 'Les nombres sont'. Donne la liste brute. Si cest illisible, réponds 'vide'."
            elif field == "commune":
                prompt = "Quel est le nom de la ville ou commune écrit sur cette image ? Réponds uniquement par le nom de la commune, sans aucune phrase."
            else:
                prompt = f"Extrais UNIQUEMENT la valeur de '{field}' sur l'image. NE FAIS AUCUNE PHRASE. Ne décris pas l'image. Réponds juste par le texte lu. Si illisible, réponds 'vide'."
                
            payload = {
                "model": "llava",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 64, "seed": 42}
            }'''

NEW_CROP_PROMPT_BLOCK = '''            # Prompt et modèle adaptatifs selon le type de document
            model_name = _get_vlm_model(type_plan)
            prompt = _get_vlm_prompt(type_plan, field)
            timeout_sec = 90 if model_name == "llama3.2-vision" else 60

            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 80, "seed": 42}
            }'''

if OLD_CROP_PROMPT_BLOCK in src:
    src = src.replace(OLD_CROP_PROMPT_BLOCK, NEW_CROP_PROMPT_BLOCK, 1)
    print("[Patch 2b] OK : prompt/model crops refactorisés")
else:
    print("[Patch 2b] SKIP : bloc crop introuvable (peut-être déjà patché)")

# 2c : Remplacer timeout=60 dans le subprocess crop par timeout=timeout_sec
OLD_TIMEOUT_CROP = 'res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)'
NEW_TIMEOUT_CROP = 'res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)'
if OLD_TIMEOUT_CROP in src:
    src = src.replace(OLD_TIMEOUT_CROP, NEW_TIMEOUT_CROP, 1)
    print("[Patch 2c] OK : timeout crop dynamique")
else:
    print("[Patch 2c] SKIP : pas de timeout=60 à remplacer")

# 2d : Remplacer le prompt + model dans _extract_with_vlm_full_page
OLD_FULL_PAGE_MODEL = '        "model": "llava",'
NEW_FULL_PAGE_MODEL = '        "model": _get_vlm_model(type_plan) if "type_plan" in dir() else "llava",'
# Plus ciblé : on cherche dans le contexte full_page
OLD_FULL_BLOCK = '''    prompt = f"Tu es un expert géomètre. Voici un document complet d'arpentage. Extrais les champs suivants de manière la plus concise possible : {', '.join(fields_to_extract)}. Attention : le texte peut être écrit avec un tampon où chaque lettre a un point en dessous (ex: M.O.T), ignore ces points pour lire le mot correctement. Ne devine pas les mots illisibles. Si c'est illisible, réponds 'illisible'. Réponds uniquement au format JSON avec les noms des champs en minuscules comme clés et les valeurs sous forme de chaîne."
    
    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [img_base64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 256, "seed": 42}
    }'''

NEW_FULL_BLOCK = '''    # Prompt et modèle adaptatifs
    model_name = _get_vlm_model(type_plan) if type_plan else "llava"
    max_dim = 2000 if model_name == "llama3.2-vision" else 1500

    _fields_desc = {
        "commune": "nom de la commune", "section": "lettre de section cadastrale",
        "feuille": "numéro de feuille", "n_ordre": "numéro d'ordre ou DA",
        "date": "date de dressé ou signature (JJ/MM/AAAA)",
        "echelle": "échelle (1/XXXX)", "geometre": "nom du géomètre-expert",
        "proprietaires_anciens": "nom du propriétaire cédant",
        "proprietaires_nouveaux": "nom du nouveau propriétaire",
        "indication": "objet du document",
    }
    _ctx = {
        "PVa": "C'est un procès-verbal de bornage tapé à la machine.",
        "CROQUIS": "C'est un ancien plan cadastral (texte manuscrit ou tampon — si tu vois des points entre lettres ex: S.A.I.N.T, lis SAINT).",
        "DMPC": "C'est un formulaire DMPC avec cartouche structuré.",
    }.get(type_plan, "C'est un document cadastral.")

    _fdesc = ", ".join(_fields_desc.get(f, f) for f in fields_to_extract)
    prompt = (
        f"Tu es un expert géomètre. {_ctx} "
        f"Extrais: {_fdesc}. "
        f"Ne devine pas les mots illisibles — mets 'vide'. "
        f"Réponds UNIQUEMENT en JSON avec les clés exactes : {', '.join(fields_to_extract)}. "
        f"Valeurs courtes et factuelles, sans phrases."
    )
    timeout_sec = 180 if model_name == "llama3.2-vision" else 120

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [img_base64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 512, "seed": 42}
    }'''

if OLD_FULL_BLOCK in src:
    src = src.replace(OLD_FULL_BLOCK, NEW_FULL_BLOCK, 1)
    print("[Patch 2d] OK : full_page prompt/model refactorisés")
else:
    print("[Patch 2d] SKIP : bloc full_page introuvable")

# 2e : Signature de _extract_with_vlm_full_page — ajouter type_plan param
OLD_SIG_FP = 'def _extract_with_vlm_full_page(img_bgr: np.ndarray, fields_to_extract: List[str], commune_db=None) -> Dict[str, Any]:'
NEW_SIG_FP = 'def _extract_with_vlm_full_page(img_bgr: np.ndarray, fields_to_extract: List[str], commune_db=None, type_plan: str = "GENERIC") -> Dict[str, Any]:'
if OLD_SIG_FP in src:
    src = src.replace(OLD_SIG_FP, NEW_SIG_FP, 1)
    print("[Patch 2e] OK : signature _extract_with_vlm_full_page mise à jour")
else:
    print("[Patch 2e] SKIP : signature déjà mise à jour ou introuvable")

# 2f : timeout full_page
OLD_TIMEOUT_FP = 'res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)'
NEW_TIMEOUT_FP = 'res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)'
if OLD_TIMEOUT_FP in src:
    src = src.replace(OLD_TIMEOUT_FP, NEW_TIMEOUT_FP, 1)
    print("[Patch 2f] OK : timeout full_page dynamique")
else:
    print("[Patch 2f] SKIP : pas de timeout=120 à remplacer")

# ────────────────────────────────────────────────────────────────────────────
# PATCH 3 : Améliorer _refine_type_plan_from_ocr
# ────────────────────────────────────────────────────────────────────────────

OLD_REFINE_BODY = '''    high_conf_blocks = [b for b in ocr_results if b[2] > 0.6]
    full_text = " ".join([b[1] for b in high_conf_blocks]).lower()
    full_text_all = " ".join([b[1] for b in ocr_results]).lower()
    
    # 0. Les extraits modernes (DGFIP) sont toujours des plans génériques/modernes
    if initial_type == "MODERN_DGFIP":
        return "GENERIC"
    if "finances publiques" in full_text_all or "extrait du plan cadastral" in full_text_all:
        return "GENERIC"
        
    # 1. Signaux textuels forts explicites
    if re.search(r'proc[eè]s\\s*[-]?\\s*verbal', full_text_all):
        return "PVa"
    if re.search(r'document\\s*modificatif|d\\.m\\.p\\.c|d[\\'\\'\\s]?arp[eoa]nt[ao]g[eo]', full_text_all):
        return "DMPC"
    if "lotissement" in full_text_all or "division" in full_text_all:
        return "PLa"
    if "croquis" in full_text_all or "conservation" in full_text_all:
        return "CROQUIS"  # Les anciens croquis sont traités via la grille spatiale globale
        
    # 2. Analyse de densité pour différencier moderne vs formulaire ancien
    labels = ["commune", "section", "feuille", "dossier", "ordre", "echelle"]
    filled_labels = 0
    empty_labels = 0
    
    for (bbox, text, prob) in ocr_results:
        if prob < 0.2: continue
        t_lower = text.lower()
        for lbl in labels:
            if lbl in t_lower:
                parts = re.split(r'[:\\-]', text)
                if len(parts) > 1 and len(parts[1].strip()) > 2:
                    filled_labels += 1
                elif len(text.lower().replace(lbl, "").strip()) > 2:
                    filled_labels += 1
                else:
                    empty_labels += 1
                    
    # S'il y a beaucoup de trous (labels trouvés seuls par l'OCR, la valeur manuscrite n'est pas lue)
    # c'est la signature d'un formulaire ancien type DMPC.
    if empty_labels > filled_labels and empty_labels >= 2:
        return "DMPC"
        
    # Si le texte est très peu dense (typique des vieux croquis ou manuscrits scannés),
    # on force le routage vers CROQUIS pour bénéficier du VLM spatial global.
    if initial_type == "GENERIC" and len(high_conf_blocks) < 80:
        return "CROQUIS"
        
    # Si le texte est très dense et que les labels sont remplis, c'est moderne (générique ou plan)
    if filled_labels >= 2 and len(high_conf_blocks) >= 80:
        return "GENERIC"
        
    return initial_type'''

NEW_REFINE_BODY = '''    high_conf_blocks = [b for b in ocr_results if b[2] > 0.6]
    all_blocks_any = [b for b in ocr_results if b[2] > 0.2]
    full_text_all = " ".join([b[1] for b in ocr_results]).lower()

    # 0. Documents modernes DGFIP
    if initial_type == "MODERN_DGFIP":
        return "GENERIC"
    if "finances publiques" in full_text_all or "extrait du plan cadastral" in full_text_all:
        return "GENERIC"

    # 1. Signaux textuels forts (ordre décroissant de spécificité)
    if re.search(r'proc[eè]s\\s*[-]?\\s*verbal', full_text_all):
        return "PVa"
    if re.search(r'bornage\\s+contradictoire|reconnaissance\\s+de\\s+limites|accord\\s+amiable\\s+de\\s+bornage', full_text_all):
        return "PVa"
    if re.search(r'document\\s*modificatif|d\\.m\\.p\\.c|d[\\'\\'\\s]?arp[eoa]nt[ao]g[eo]', full_text_all):
        return "DMPC"
    if "lotissement" in full_text_all or "division parcellaire" in full_text_all:
        return "PLa"
    if "croquis" in full_text_all or "conservation" in full_text_all:
        return "CROQUIS"

    # 2. Heuristique cumulative PVa texte libre (sans mot-clé fort unique)
    _pva_s = sum([
        1 if re.search(r'soussign[eé].*g[eé]om[eè]tre', full_text_all) else 0,
        1 if re.search(r'\\bbornage\\b|\\breconnaissance\\b|limites?\\s+de\\s+propri', full_text_all) else 0,
        1 if re.search(r'fait\\s+[aà]\\s+[A-Z][a-z]', full_text_all) else 0,
        1 if re.search(r'certifi[eé]\\s+exact|vu\\s+et\\s+approuv[eé]', full_text_all) else 0,
    ])
    if _pva_s >= 2:
        print(f"  [Classif] Signaux PVa libres ({_pva_s}/4) -> PVa")
        return "PVa"

    # 3. Labels DMPC vides vs remplis
    labels = ["commune", "section", "feuille", "dossier", "ordre", "echelle"]
    filled_labels = 0
    empty_labels = 0
    for (bbox, text, prob) in ocr_results:
        if prob < 0.2: continue
        t_lower = text.lower()
        for lbl in labels:
            if lbl in t_lower:
                parts = re.split(r'[:\\-]', text)
                if len(parts) > 1 and len(parts[1].strip()) > 2:
                    filled_labels += 1
                elif len(text.lower().replace(lbl, "").strip()) > 2:
                    filled_labels += 1
                else:
                    empty_labels += 1

    if empty_labels > filled_labels and empty_labels >= 2:
        print(f"  [Classif] Labels DMPC vides ({empty_labels}v/{filled_labels}r) -> DMPC")
        return "DMPC"

    # 4. Densité relative (remplace seuil fixe 80 blocs)
    n_high = len(high_conf_blocks)
    n_all = max(len(all_blocks_any), 1)
    ratio_hc = n_high / n_all

    if initial_type in ("GENERIC", "CROQUIS"):
        if filled_labels >= 2 and n_high >= 60:
            return "GENERIC"
        if n_high < 40 or (n_high < 70 and ratio_hc < 0.55):
            print(f"  [Classif] Document peu dense ({n_high} blocs HC, ratio={ratio_hc:.2f}) -> CROQUIS")
            return "CROQUIS"

    if filled_labels >= 2 and n_high >= 80:
        return "GENERIC"

    return initial_type'''

if OLD_REFINE_BODY in src:
    src = src.replace(OLD_REFINE_BODY, NEW_REFINE_BODY, 1)
    print("[Patch 3] OK : _refine_type_plan_from_ocr amélioré")
else:
    print("[Patch 3] SKIP : corps de _refine_type_plan_from_ocr introuvable (peut différer légèrement)")

# ────────────────────────────────────────────────────────────────────────────
# PATCH 4 : Passer type_plan à _extract_with_vlm_full_page dans process_plan
# ────────────────────────────────────────────────────────────────────────────

OLD_FP_CALL = 'vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db)'
NEW_FP_CALL = 'vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db, type_plan=type_plan)'
if OLD_FP_CALL in src:
    src = src.replace(OLD_FP_CALL, NEW_FP_CALL, 1)
    print("[Patch 4] OK : type_plan passé à _extract_with_vlm_full_page")
else:
    print("[Patch 4] SKIP : appel introuvable")

# ────────────────────────────────────────────────────────────────────────────
# ÉCRITURE
# ────────────────────────────────────────────────────────────────────────────
with open(TARGET, "w", encoding="utf-8") as f:
    f.write(src)
print(f"[Patch] Fichier écrit : {len(src)} caractères")
print("[Patch] Terminé.")
