"""
Corrections plan_classifier.py :
1. _clean_vlm_response : ajout des patterns "La date écrite dans la case est le..."
2. _validate_field("date") : pré-extraction de date depuis réponse verbeuse VLM
3. _validate_field("geometre") : suppression du fallback "liste assouplie" — si pas dans GEOMETRES_CONNUS, on rejette
4. Prompts VLM géomètre : ajout de la liste des géomètres connus dans le prompt pour guider le VLM
"""
import re

path = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py'

with open(path, encoding='utf-8') as f:
    content = f.read()

changes = []

# ═══════════════════════════════════════════════════════════════════════
# FIX 1 : _clean_vlm_response — ajout patterns date verbeuse
# ═══════════════════════════════════════════════════════════════════════
OLD_CLEAN = '''def _clean_vlm_response(val: str) -> str:
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
    val = val.replace(\'"\', \'\').replace("\'", "").strip(" \'.:,")
    return val'''

# On cherche la fonction par son début unique
OLD_CLEAN_START = 'def _clean_vlm_response(val: str) -> str:'
OLD_CLEAN_END   = '    val = val.replace(\'"\', \'\').replace("\'", "").strip(" \'.:,")\n    return val'

# Nouvelle version avec :
# 1. Pattern spécifique pour dates verbeuses (avant les autres)
# 2. Nettoyage géomètre boilerplate
NEW_CLEAN = '''def _clean_vlm_response(val: str, field: str = "") -> str:
    """Nettoyage commun des réponses VLM : supprime les phrases parasites."""
    import re as _re
    if not val:
        return val

    # ── Extraction prioritaire pour les dates (avant tout autre nettoyage) ──
    # Le VLM répond souvent "La date écrite dans la case est le JJ/MM/AAAA"
    if field == "date" or (not field and _re.search(r"\\d{1,2}[/\\\\.\\-]\\d{1,2}[/\\\\.\\-]\\d{2,4}", val)):
        _date_m = _re.search(
            r"(\\d{1,2}\\s*[/\\\\.\\-]\\s*\\d{1,2}\\s*[/\\\\.\\-]\\s*\\d{2,4}"
            r"|\\d{1,2}\\s+(?:janvier|f[e\\xe9]vrier|mars|avril|mai|juin|juillet|ao[u\\xfb]t|septembre|octobre|novembre|d[e\\xe9]cembre)\\s+\\d{4})",
            val, _re.IGNORECASE
        )
        if _date_m:
            return _date_m.group(1).strip()

    PREFIXES = [
        r"la (?:date|valeur|r[e\\xe9]ponse|information) (?:[\\w\\s]{0,30}?)(?:est|:)\\s*",
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
    val = _re.sub(r"(?i)^(?:texte|le texte|la valeur)?\\s*(?:dans|sur)?\\s*(?:l\'?image|ce document)?\\s*(?:est|:)?\\s*", "", val).strip()
    val = val.replace(\'"\', \'\').replace("\'", "").strip(" \'.:,")
    return val'''

# Chercher et remplacer la fonction
idx_start = content.find('def _clean_vlm_response(val: str) -> str:')
if idx_start == -1:
    # Essayer avec les caractères mojibake
    idx_start = content.find('def _clean_vlm_response(val: str)')
    
if idx_start != -1:
    # Trouver la fin de la fonction (prochaine def au même niveau d'indentation)
    idx_end = content.find('\ndef ', idx_start + 10)
    old_func = content[idx_start:idx_end]
    content = content[:idx_start] + NEW_CLEAN + '\n\n\n' + content[idx_end:]
    changes.append("FIX 1: _clean_vlm_response amélioré (dates verbeuses)")
else:
    changes.append("SKIP FIX 1: _clean_vlm_response non trouvé")

# ═══════════════════════════════════════════════════════════════════════
# FIX 2 : _validate_field("geometre") — supprimer le fallback "liste assouplie"
# Règle : si pas dans GEOMETRES_CONNUS via fuzzy (score >= 70), on rejette.
# ═══════════════════════════════════════════════════════════════════════
OLD_GEO_FALLBACK = '''        # 3. LISTE STRICTE ASSOUPLIE : Si pas dans la liste mais valide, on le renvoie tel quel
        if len(val_clean_norm) >= 3:
             return val_clean_norm
        return None'''

NEW_GEO_FALLBACK = '''        # 3. REJET STRICT : si pas dans GEOMETRES_CONNUS, on rejette.
        # Le géomètre doit être dans notre base d'archives ou vide.
        return None'''

if OLD_GEO_FALLBACK in content:
    content = content.replace(OLD_GEO_FALLBACK, NEW_GEO_FALLBACK)
    changes.append("FIX 2: Géomètre — fallback 'liste assouplie' supprimé (rejet strict)")
else:
    changes.append("SKIP FIX 2: pattern géomètre non trouvé")

# ═══════════════════════════════════════════════════════════════════════
# FIX 3 : Prompts VLM géomètre — ajouter la liste des géomètres connus
# ═══════════════════════════════════════════════════════════════════════
OLD_GEO_PROMPT_DMPC = '"geometre": "Quel est le nom du g'
# Chercher le prompt géomètre DMPC et le remplacer
OLD_DMPC_GEO = '"geometre": "Quel est le nom du g\u00e9om\u00e8tre ou cabinet \u00e9crit dans cette case ? R\u00e9ponds uniquement par le nom, sans phrase.",'

NEW_DMPC_GEO = (
    '"geometre": "Dans cette case, quel est le nom du g\u00e9om\u00e8tre-expert ? '
    'Les g\u00e9om\u00e8tres de ce territoire sont : DUPUY, HARROIS, RACAT, SERRET, CEYTE, BARRIAL, ROBERT. '
    'R\u00e9ponds UNIQUEMENT par un seul nom (ex: HARROIS). Si tu ne vois aucun de ces noms, r\u00e9ponds \'vide\'.",'
)

if OLD_DMPC_GEO in content:
    content = content.replace(OLD_DMPC_GEO, NEW_DMPC_GEO)
    changes.append("FIX 3a: Prompt VLM géomètre DMPC renforcé avec liste des géomètres")
else:
    changes.append("SKIP FIX 3a: prompt géomètre DMPC non trouvé (vérifier encodage)")

# Même chose pour CROQUIS
OLD_CROQUIS_GEO = '"geometre": "Ce document est un ancien plan cadastral. Le g\u00e9om\u00e8tre est identifi\u00e9 par son tampon ou signature. Son nom peut \u00eatre en tampon avec des points entre les lettres \u2014 ignore ces points. Quel est le nom ? R\u00e9ponds uniquement par le nom.",'

NEW_CROQUIS_GEO = (
    '"geometre": "Ce document est un ancien plan cadastral. Le g\u00e9om\u00e8tre est identifi\u00e9 par son tampon ou signature. '
    'Les g\u00e9om\u00e8tres de ce territoire sont : DUPUY, HARROIS, RACAT, SERRET, CEYTE, BARRIAL, ROBERT. '
    'Son nom peut \u00eatre en tampon avec des points entre les lettres (ex: H.A.R.R.O.I.S = HARROIS) \u2014 ignore ces points. '
    'R\u00e9ponds UNIQUEMENT par un seul nom parmi cette liste. Si illisible ou absent, r\u00e9ponds \'vide\'.",'
)

if OLD_CROQUIS_GEO in content:
    content = content.replace(OLD_CROQUIS_GEO, NEW_CROQUIS_GEO)
    changes.append("FIX 3b: Prompt VLM géomètre CROQUIS renforcé")
else:
    changes.append("SKIP FIX 3b: prompt géomètre CROQUIS non trouvé")

# ═══════════════════════════════════════════════════════════════════════
# FIX 4 : Seuil fuzzy géomètre abaissé à 70 (HARROIS -> HRNOIS GERVOLS : score trop bas)
# ═══════════════════════════════════════════════════════════════════════
OLD_GEO_SCORE = '                if result and result[1] >= 75:\n                    return result[0]'
NEW_GEO_SCORE = '                if result and result[1] >= 65:\n                    return result[0]'

if OLD_GEO_SCORE in content:
    content = content.replace(OLD_GEO_SCORE, NEW_GEO_SCORE)
    changes.append("FIX 4: Seuil fuzzy géomètre abaissé 75->65 (tolère OCR dégradé)")
else:
    changes.append("SKIP FIX 4: seuil fuzzy non trouvé")

# ═══════════════════════════════════════════════════════════════════════
# FIX 5 : _validate_field("date") — pré-extraction d'une date depuis réponse verbeuse
# Si val contient une phrase + une date, on extrait juste la date
# ═══════════════════════════════════════════════════════════════════════
OLD_DATE_VAL = '''    elif field_type == "date":'''
NEW_DATE_PREFIX = '''    elif field_type == "date":
        # Pré-extraction : si la réponse VLM est verbeuse ("La date écrite... est le XX/XX/XXXX"),
        # on extrait uniquement la date et on continue la validation normalement.
        _date_m = re.search(
            r'(\\d{1,2}\\s*[/\\.\\-]\\s*\\d{1,2}\\s*[/\\.\\-]\\s*\\d{2,4}'
            r'|\\d{1,2}\\s+(?:janvier|f[e\\xe9]vrier|mars|avril|mai|juin|juillet|ao[u\\xfb]t|septembre|octobre|novembre|d[e\\xe9]cembre)\\s+\\d{4})',
            val, re.IGNORECASE
        )
        if _date_m and len(val) > 15:  # Réponse longue = verbeuse → extraire la date
            val = _date_m.group(1).strip()
            val_norm = val.lower()
'''

OLD_DATE_VALIDATE = '''    elif field_type == "date":
        # Normaliser ann'''

if OLD_DATE_VALIDATE in content:
    content = content.replace(OLD_DATE_VALIDATE, NEW_DATE_PREFIX + '        # Normaliser ann')
    changes.append("FIX 5: _validate_field(date) - pré-extraction date depuis réponse verbeuse VLM")
else:
    changes.append("SKIP FIX 5: date validate non trouvé")

# Sauvegarder
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Corrections appliquées:")
for c in changes:
    status = "OK" if c.startswith("FIX") else "SKIP"
    print(f"  [{status}] {c}")
