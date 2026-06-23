"""
Script de réparation de plan_classifier.py
Corrige la corruption introduite et applique les 6 corrections de bugs.
"""
import re, shutil, os

SRC = "plan_classifier.py"
BACKUP = "plan_classifier.py.bak"

# Sauvegarder l'original
shutil.copy2(SRC, BACKUP)
print(f"Backup : {BACKUP}")

with open(SRC, "r", encoding="utf-8", errors="replace") as f:
    content = f.read()

# =============================================================================
# PATCH 1 — Restaurer la ligne 335 corrompue :
# "val_norm.startswith(interdit.lower()    elif field_type in ..."
# → "val_norm.startswith(interdit.lower() + \" \"):\n            return None\n"
# =============================================================================

OLD_CORRUPT_LINE = (
    '        if val_norm == interdit.lower() or val_norm.startswith(interdit.lower()'
    '    elif field_type in ("n_ordre", "n_dossier"):\n'
    '        # Doit contenir au moins un chiffre, pas de prose\n'
    '        if any(u in val_norm for u in [" ha", " a ", " ca", "m2", "nee", "demeurant", "epouse"]):\n'
    '            return None\n'
    '\n'
    "        # BUG 3 FIX — Rejeter les valeurs de type \"Mois + Année\" (ex: \"Sept1970\", \"Janvier2005\")\n"
    "        _MOIS = r'(?:jan(?:vier)?|f[eé]v(?:rier)?|mar(?:s)?|avr(?:il)?|mai|juin|juil(?:let)?|ao[uû]t?|sep(?:t(?:embre)?)?|oct(?:obre)?|nov(?:embre)?|d[eé]c(?:embre)?)'\n"
    r"        if re.match(r'(?i)^' + _MOIS + r'\\s*\\d{2,4}$', val.strip()):" + "\n"
    "            return None  # C'est un mois + année, pas un numéro d'ordre\n"
    "\n"
    "        # Rejeter si contient uniquement un mot alphabétique suivi d'une année (bruit OCR)\n"
    r"        if re.match(r'(?i)^[A-Za-z]{3,10}\\s*\\d{4}$', val.strip()) and not re.search(r'[A-Z]{1,4}\\d{2}', val.upper()):" + "\n"
    "            return None\n"
    "\n"
    "        # Format \"Récents\" (ex: A09032, A22-145, 2021-045) -> 1 lettre + année + incrément\n"
    r"        m_recent = re.search(r'\\b([A-Z]\\d{2}[\\.\\-]?\\d{2,5})\\b|\\b(\\d{4}[.\\-]\\d{2,5})\\b', val.upper())" + "\n"
    "        if m_recent:\n"
    "            return (m_recent.group(1) or m_recent.group(2)).replace(\" \", \"\")\n"
    "\n"
    "        # Format DA classique (ex: 3101B, 249, 60.8.A)\n"
    r"        m_da = re.search(r'(\\b\\d{1,5}\\s*[A-Z]\\b|\\b\\d{1,5}\\b)', val.upper())" + "\n"
    "        if m_da:\n"
    r"            cleaned = re.sub(r'\\s+', '', m_da.group(1))" + "\n"
    "            # Rejeter les purs chiffres de 4+ chiffres qui sont des années (1970, 2009, etc.)\n"
    r"            if re.match(r'^(19|20)\\d{2}$', cleaned):" + "\n"
    "                return None\n"
    "            return cleaned\n"
    "\n"
    "        # Fallback pour d'autres formats génériques (nettoyage)\n"
    '        clean = re.sub(r"[^A-Za-z0-9\\-\\/\\.]", "", val)\n'
    "        if 2 <= len(clean) <= 20 and re.search(r'\\d', clean):\n"
    "            return clean\n"
    "        return Noneier\"):\n"
    "        # Doit contenir au moins un chiffre, pas de prose\n"
    '        if any(u in val_norm for u in [" ha", " a ", " ca", "m2", "n\xc3\xa9e", "demeurant", "\xc3\xa9pouse"]):\n'
    "            return None\n"
    "            \n"
    "        # Format \"Récents\" (ex: A09032, A22-145, 2021-045) -> 1 lettre + année + incrément\n"
    r"        m_recent = re.search(r'\b([A-Z]\d{2}[.\-]?\d{2,5})\b|\b(\d{4}[.\-]\d{2,5})\b', val.upper())" + "\n"
)

NEW_CORRECT = (
    '        if val_norm == interdit.lower() or val_norm.startswith(interdit.lower() + " "):\n'
    "            return None\n"
    "\n"
    '    if field_type == "section":\n'
    r"        m = re.search(r'\b([A-Za-z]{1,2}|\d{1,3})\b', val)" + "\n"
    "        if m:\n"
    "            clean = m.group(1).upper()\n"
    r"            if re.match(r'^[018]$', clean) and len(val) <= 3:" + "\n"
    '                clean = clean.replace("0", "O").replace("1", "I").replace("8", "B")\n'
    "            return clean\n"
    "        return None\n"
    "\n"
    '    elif field_type == "feuille":\n'
    r"        m = re.search(r'\b(\d{1,4}[A-Za-z]?|[A-Za-z]\d{1,3})\b', val)" + "\n"
    "        if m:\n"
    "            return m.group(1).upper()\n"
    "        return None\n"
    "\n"
    '    elif field_type == "echelle":\n'
    '        if any(u in val_norm for u in ["ca", "ha", "m2", " a ", "m\u00b2"]):\n'
    "            return None\n"
    r"        m = re.search(r'(1\s*[/:]\s*\d{3,5}|\d{3,5})', val)" + "\n"
    "        if m:\n"
    '            return m.group(1).replace(" ", "")\n'
    "        return None\n"
    "\n"
    '    elif field_type in ("n_ordre", "n_dossier"):\n'
    '        if any(u in val_norm for u in [" ha", " a ", " ca", "m2", "nee", "demeurant", "epouse"]):\n'
    "            return None\n"
    "\n"
    "        # BUG 3 FIX \u2014 Rejeter les valeurs Mois+Ann\u00e9e (ex: Sept1970)\n"
    "        _MOIS = (r'(?:jan(?:vier)?|f[e\u00e9]v(?:rier)?|mar(?:s)?|avr(?:il)?|mai|juin'\n"
    "                 r'|juil(?:let)?|ao[u\u00fb]t?|sep(?:t(?:embre)?)?|oct(?:obre)?'\n"
    "                 r'|nov(?:embre)?|d[e\u00e9]c(?:embre)?)')\n"
    "        if re.match(r'(?i)^' + _MOIS + r'\\s*\\d{2,4}$', val.strip()):\n"
    "            return None\n"
    "        if re.match(r'(?i)^[A-Za-z]{3,10}\\s*\\d{4}$', val.strip()):\n"
    "            if not re.search(r'[A-Z]{1,4}\\d{2}', val.upper()):\n"
    "                return None\n"
    "\n"
    "        m_recent = re.search(r'\\b([A-Z]\\d{2}[.\\-]?\\d{2,5})\\b|\\b(\\d{4}[.\\-]\\d{2,5})\\b', val.upper())\n"
    "        if m_recent:\n"
    '            return (m_recent.group(1) or m_recent.group(2)).replace(" ", "")\n'
    "        m_da = re.search(r'(\\b\\d{1,5}\\s*[A-Z]\\b|\\b\\d{1,5}\\b)', val.upper())\n"
    "        if m_da:\n"
    "            cleaned = re.sub(r'\\s+', '', m_da.group(1))\n"
    "            if re.match(r'^(19|20)\\d{2}$', cleaned):\n"
    "                return None\n"
    "            return cleaned\n"
    "        clean = re.sub(r'[^A-Za-z0-9\\-\\/\\.]', '', val)\n"
    "        if 2 <= len(clean) <= 20 and re.search(r'\\d', clean):\n"
    "            return clean\n"
    "        return None\n"
    "\n"
    '    elif field_type == "date":\n'
)

if OLD_CORRUPT_LINE in content:
    content = content.replace(OLD_CORRUPT_LINE, NEW_CORRECT, 1)
    print("PATCH 1 appliqué : restauration de la structure if/elif")
else:
    print("PATCH 1 : pattern non trouvé — vérification manuelle nécessaire")
    # Chercher la ligne corrompue de façon plus souple
    idx = content.find('return Noneier\"):')
    if idx >= 0:
        print(f"  Corruption 'return Noneier' trouvée à l'index {idx}")

# =============================================================================
# PATCH 2 — Remplacer le bloc date (toujours en double après la corruption)
# =============================================================================

OLD_DATE_BLOCK = (
    '    elif field_type == "date":\n'
    '        # Doit \xc3\xaatre une vraie date (pas une date de naissance sortie de contexte)\n'
    '        # Les patterns CONTEXTUAL_PATTERNS ont d\xc3\xa9j\xc3\xa0 filtr\xc3\xa9 le contexte (\xc3\xa9tabli le / dress\xc3\xa9 le)\n'
    '        m = re.search(\n'
    r"            r'(\d{1,2}\s*[/\-\.]\s*\d{1,2}\s*[/\-\.]\s*\d{2,4}'" + "\n"
    r"            r'|\d{1,2}\s+(?:janvier|f[eÃ©]vrier|mars|avril|mai|juin|juillet|ao[uÃ»]t|septembre|octobre|novembre|d[eÃ©]cembre)\s+\d{4})'," + "\n"
    '            val_norm\n'
    '        )\n'
    '        if m:\n'
    '            return val.strip()\n'
    '        return None\n'
)

NEW_DATE_BLOCK = (
    '    elif field_type == "date":\n'
    "        # BUG 6 FIX \u2014 Normaliser les ann\u00e9es \u00e0 2 chiffres (14/10/97 -> 14/10/1997)\n"
    "        def _norm_year(date_str: str) -> str:\n"
    "            def _expand_yr(m_y):\n"
    "                sep, yy_str = m_y.group(1), m_y.group(2)\n"
    "                yy = int(yy_str)\n"
    "                yyyy = 2000 + yy if yy <= 30 else 1900 + yy\n"
    "                return sep + str(yyyy)\n"
    "            return re.sub(r'([/\\.\\-])(\\d{2})$', _expand_yr, date_str.strip())\n"
    "\n"
    "        m = re.search(\n"
    "            r'(\\d{1,2}\\s*[/\\-\\.]\\s*\\d{1,2}\\s*[/\\-\\.]\\s*\\d{2,4}'\n"
    "            r'|\\d{1,2}\\s+(?:janvier|f[e\u00e9]vrier|mars|avril|mai|juin'\n"
    "            r'|juillet|ao[u\u00fb]t|septembre|octobre|novembre|d[e\u00e9]cembre)\\s+\\d{4})',\n"
    "            val_norm\n"
    "        )\n"
    "        if m:\n"
    "            return _norm_year(val.strip())\n"
    "        return None\n"
)

if OLD_DATE_BLOCK in content:
    content = content.replace(OLD_DATE_BLOCK, NEW_DATE_BLOCK, 1)
    print("PATCH 2 appliqué : bloc date normalisé (années 2 chiffres)")
else:
    print("PATCH 2 : bloc date non trouvé avec ce pattern exact — passage au remplacement partiel")
    # Essai avec un pattern plus court
    OLD_DATE_SHORT = (
        "        if m:\n"
        "            return val.strip()\n"
        "        return None\n"
        "\n"
        "    elif field_type == \"commune\":"
    )
    NEW_DATE_SHORT = (
        "        if m:\n"
        "            return _norm_year(val.strip())\n"
        "        return None\n"
        "\n"
        "    elif field_type == \"commune\":"
    )
    if OLD_DATE_SHORT in content:
        content = content.replace(OLD_DATE_SHORT, NEW_DATE_SHORT, 1)
        # Ajouter la fonction _norm_year juste avant "m = re.search"
        content = content.replace(
            "    elif field_type == \"date\":\n        # Doit",
            "    elif field_type == \"date\":\n"
            "        # BUG 6 FIX - normaliser ann\u00e9es 2 chiffres\n"
            "        def _norm_year(s):\n"
            "            def _e(m):\n"
            "                yy=int(m.group(2)); return m.group(1)+(str(2000+yy) if yy<=30 else str(1900+yy))\n"
            "            return re.sub(r'([/\\.\\-])(\\d{2})$',_e,s.strip())\n"
            "        # Doit",
            1
        )
        print("PATCH 2b appliqué")

# =============================================================================
# PATCH 3 — Remplacer le bloc commune (BUG 1 FIX)
# =============================================================================

OLD_COMMUNE = (
    '    elif field_type == "commune":\n'
    '        # Pas de chiffres, pas trop court, pas trop long (une commune n\'est pas une phrase)\n'
    '        COMMUNES_INTERDITES = {"commune", "territoire", "section", "echelle", "plan", "bornage"}\n'
    '        if val_norm.lower() in COMMUNES_INTERDITES:\n'
    '            return None\n'
    '        # Min 4 chars (pas "Mme", pas "CH"), max 50 chars (pas une phrase comme "Les operations ont ete...")\n'
    '        if len(val) < 4 or len(val) > 50:\n'
    '            return None\n'
    '        # Pas de chiffres sauf rares cas (Saint-1er...)\n'
    '        if sum(c.isdigit() for c in val) > 2:\n'
    '            return None\n'
    "        # Pas de phrase (pas de verbe/conjonction du d\xc3\xa9but)\n"
    "        if re.match(r'(?i)^(les?|des?|de|du|un|une|ce|cette|il|elle|on|nous|vous|ils)', val):\n"
    '            return None\n'
    '        return val.strip()\n'
)

NEW_COMMUNE = (
    '    elif field_type == "commune":\n'
    '        COMMUNES_INTERDITES = {"commune", "territoire", "section", "echelle", "plan", "bornage"}\n'
    '        if val_norm.lower() in COMMUNES_INTERDITES:\n'
    '            return None\n'
    '        if len(val) < 4 or len(val) > 50:\n'
    '            return None\n'
    '        if sum(c.isdigit() for c in val) > 2:\n'
    '            return None\n'
    "        if re.match(r'(?i)^(les?|des?|de|du|un|une|ce|cette|il|elle|on|nous|vous|ils)', val):\n"
    '            return None\n'
    "        # BUG 1 FIX \u2014 Rejeter les en-t\u00eates institutionnels\n"
    "        _INST = [\n"
    '            "direction generale des finances publiques",\n'
    '            "direction g\u00e9n\u00e9rale des finances publiques", "dgfip",\n'
    '            "services fiscaux", "conservation des hypotheques",\n'
    '            "conservation des hypoth\u00e8ques", "bureau de la conservation",\n'
    '            "ministere", "minist\u00e8re", "prefecture", "pr\u00e9fecture",\n'
    '            "sous-prefecture", "mairie", "conseil general", "conseil g\u00e9n\u00e9ral",\n'
    "        ]\n"
    "        val_low = val.lower().strip()\n"
    "        if any(inst in val_low for inst in _INST):\n"
    "            return None\n"
    "        if len(val.split()) > 6:\n"
    "            return None\n"
    '        return val.strip()\n'
)

if OLD_COMMUNE in content:
    content = content.replace(OLD_COMMUNE, NEW_COMMUNE, 1)
    print("PATCH 3 appliqué : filtre institutions dans commune")
else:
    print("PATCH 3 : bloc commune non trouvé")

# =============================================================================
# PATCH 4 — Bloc géomètre (BUG 2 FIX)
# =============================================================================

OLD_GEO_START = '    elif field_type == "geometre":\n        # Rejeter si le texte commence par un code postal\n        if re.match(r\'^\d{4,5}\', val.strip()):\n'
NEW_GEO_START = (
    '    elif field_type == "geometre":\n'
    "        # BUG 2 FIX \u2014 Rejeter si commence par chiffre/guillemet ou description de r\u00f4le\n"
    "        if re.match(r'^[\\d\"\\'.]+', val.strip()):\n"
)

# Approche plus sûre : chercher par fragment unique
OLD_GEO_FRAG = "    elif field_type == \"geometre\":\n        # Rejeter si le texte commence par un code postal\n"
NEW_GEO_FRAG = (
    "    elif field_type == \"geometre\":\n"
    "        # Rejeter si commence par chiffre/guillemet (BUG 2 FIX)\n"
    "        if re.match(r'^[\\d\"\\'.]+', val.strip()):\n"
    "            return None\n"
    "        # BUG 2 FIX \u2014 Rejeter les descriptions de r\u00f4le (>5 mots = formulaire)\n"
    "        if len(val.strip().split()) > 5:\n"
    "            return None\n"
    "        _TERMES_ROLE = [\n"
    '            "inspecteur", "technicien", "qualite", "qualit\u00e9",\n'
    '            "personne agreee", "personne agr\u00e9\u00e9e", "fonctionnaire",\n'
    '            "certifie par", "certifi\u00e9 par", "soussignes", "soussign\u00e9s",\n'
    '            "present document", "pr\u00e9sent document",\n'
    "        ]\n"
    "        if any(t in val.lower() for t in _TERMES_ROLE):\n"
    "            return None\n"
    "        # (suite des vérifications existantes)\n"
    "        if re.match(r'^\\d{4,5}', val.strip()):\n"
)

if OLD_GEO_FRAG in content:
    content = content.replace(OLD_GEO_FRAG, NEW_GEO_FRAG, 1)
    # Retirer le "return None" en doublon qui suit immédiatement
    content = content.replace(
        "        if re.match(r'^\\d{4,5}', val.strip()):\n"
        "            return None\n"
        "        # Rejeter les faux g",
        "        if re.match(r'^\\d{4,5}', val.strip()):\n"
        "            return None\n"
        "        # Faux g",
        1
    )
    print("PATCH 4 appliqué : filtre description de rôle dans géomètre")
    # Ajouter aussi "gedmetry" et "geometry" dans FAUX_GEOMETRES
    content = content.replace(
        '            "soussign", "expert", "experts bp", "le geometre", "le g\u00e9om\u00e8tre",\n'
        '        }\n',
        '            "soussign", "expert", "experts bp", "le geometre", "le g\u00e9om\u00e8tre",\n'
        '            "metre-expert foncier", "gedmetry", "geometry",\n'
        '        }\n',
        1
    )
else:
    print("PATCH 4 : bloc géomètre non trouvé avec ce fragment")

# =============================================================================
# PATCH 5 — Ajouter _fix_encoding() après _norm()
# =============================================================================
FIX_ENC_FUNC = '''
def _fix_encoding(text: str) -> str:
    """Corrige le double encodage latin-1/UTF-8 (ex: 'Ã©' -> 'é')."""
    if not text:
        return text
    try:
        fixed = text.encode('latin-1').decode('utf-8')
        if any(c in fixed for c in 'éèêëàâùûîïôç'):
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text

'''

if "_fix_encoding" not in content:
    # Insérer après la fonction _norm
    content = content.replace(
        "\n\n# \u2500\u2500 Validation m",
        FIX_ENC_FUNC + "\n# \u2500\u2500 Validation m",
        1
    )
    print("PATCH 5 appliqué : ajout de _fix_encoding()")
else:
    print("PATCH 5 : _fix_encoding déjà présente")

# =============================================================================
# Écrire le fichier corrigé
# =============================================================================
with open(SRC, "w", encoding="utf-8") as f:
    f.write(content)
print(f"\n✅ plan_classifier.py réparé et sauvegardé.")
print(f"   Backup disponible dans : {BACKUP}")

# Vérification syntaxe
import py_compile, sys
try:
    py_compile.compile(SRC, doraise=True)
    print("✅ Syntaxe Python OK")
except py_compile.PyCompileError as e:
    print(f"❌ ERREUR de syntaxe : {e}")
    print("   Restauration du backup...")
    shutil.copy2(BACKUP, SRC)
    sys.exit(1)
