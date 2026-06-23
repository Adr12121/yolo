"""
fix_plan_classifier.py
Répare la corruption de plan_classifier.py introduite par un patch raté.
Lancez : python fix_plan_classifier.py
"""
import shutil, py_compile, sys, os

SRC = os.path.join(os.path.dirname(__file__), "plan_classifier.py")
BACKUP = SRC + ".bak"

shutil.copy2(SRC, BACKUP)
print(f"[1/4] Backup sauvegardé : {BACKUP}")

with open(SRC, "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()

print(f"[2/4] Fichier lu : {len(lines)} lignes")

# Trouver les lignes corrompues
idx_corrupt1 = None  # ligne "startswith(interdit.lower()    elif..."
idx_corrupt2 = None  # ligne "return Noneier"):"

for i, l in enumerate(lines):
    if 'val_norm.startswith(interdit.lower()    elif' in l:
        idx_corrupt1 = i
    if 'return Noneier' in l:
        idx_corrupt2 = i

print(f"    Corruption 1 (startswith) à la ligne {idx_corrupt1 + 1 if idx_corrupt1 is not None else 'non trouvée'}")
print(f"    Corruption 2 (Noneier)    à la ligne {idx_corrupt2 + 1 if idx_corrupt2 is not None else 'non trouvée'}")

if idx_corrupt1 is None or idx_corrupt2 is None:
    print("ERREUR : corruptions non trouvées. Le fichier a peut-être déjà été réparé.")
    sys.exit(1)

# ── BLOC DE REMPLACEMENT PROPRE ──────────────────────────────────────────────
# Remplace tout entre idx_corrupt1 et idx_corrupt2 (inclus)
# par la version corrigée de la structure if/elif de _validate_field

NOUVEAU_BLOC = """\
    for interdit in LABEL_INTERDITS:
        if val_norm == interdit.lower() or val_norm.startswith(interdit.lower() + " "):
            return None

    if field_type == "section":
        m = re.search(r'\\b([A-Za-z]{1,2}|\\d{1,3})\\b', val)
        if m:
            clean = m.group(1).upper()
            if re.match(r'^[018]$', clean) and len(val) <= 3:
                clean = clean.replace("0", "O").replace("1", "I").replace("8", "B")
            return clean
        return None

    elif field_type == "feuille":
        m = re.search(r'\\b(\\d{1,4}[A-Za-z]?|[A-Za-z]\\d{1,3})\\b', val)
        if m:
            return m.group(1).upper()
        return None

    elif field_type == "echelle":
        if any(u in val_norm for u in ["ca", "ha", "m2", " a ", "m\\u00b2"]):
            return None
        m = re.search(r'(1\\s*[/:]\\s*\\d{3,5}|\\d{3,5})', val)
        if m:
            return m.group(1).replace(" ", "")
        return None

    elif field_type in ("n_ordre", "n_dossier"):
        # Doit contenir au moins un chiffre, pas de prose
        if any(u in val_norm for u in [" ha", " a ", " ca", "m2", "nee", "demeurant", "epouse"]):
            return None
        # FIX BUG 3 — Rejeter "Mois + Année" (ex: Sept1970, Janvier2005)
        _MOIS = (r'(?:jan(?:vier)?|f[e\\u00e9]v(?:rier)?|mar(?:s)?|avr(?:il)?|mai|juin'
                 r'|juil(?:let)?|ao[u\\u00fb]t?|sep(?:t(?:embre)?)?|oct(?:obre)?'
                 r'|nov(?:embre)?|d[e\\u00e9]c(?:embre)?)')
        if re.match(r'(?i)^' + _MOIS + r'\\s*\\d{2,4}$', val.strip()):
            return None
        if re.match(r'(?i)^[A-Za-z]{3,10}\\s*\\d{4}$', val.strip()):
            if not re.search(r'[A-Z]{1,4}\\d{2}', val.upper()):
                return None
        # Format récent (ex: A09032, A22-145, 2021-045)
        m_recent = re.search(r'\\b([A-Z]\\d{2}[.\\-]?\\d{2,5})\\b|\\b(\\d{4}[.\\-]\\d{2,5})\\b', val.upper())
        if m_recent:
            return (m_recent.group(1) or m_recent.group(2)).replace(" ", "")
        # Format DA classique (ex: 3101B, 249)
        m_da = re.search(r'(\\b\\d{1,5}\\s*[A-Z]\\b|\\b\\d{1,5}\\b)', val.upper())
        if m_da:
            cleaned = re.sub(r'\\s+', '', m_da.group(1))
            if re.match(r'^(19|20)\\d{2}$', cleaned):  # année pure -> rejet
                return None
            return cleaned
        clean = re.sub(r'[^A-Za-z0-9\\-\\/\\.]', '', val)
        if 2 <= len(clean) <= 20 and re.search(r'\\d', clean):
            return clean
        return None

"""

# Reconstruire le fichier
#  - Garder tout avant idx_corrupt1
#  - Insérer le nouveau bloc
#  - Sauter les lignes jusqu'à idx_corrupt2 (inclus)
#  - Reprendre à partir de idx_corrupt2 + 1

new_lines = (
    lines[:idx_corrupt1] +
    [NOUVEAU_BLOC] +
    lines[idx_corrupt2 + 1:]
)

with open(SRC, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"[3/4] Fichier réparé écrit ({len(new_lines)} lignes)")

# Vérifier la syntaxe
print("[4/4] Vérification de la syntaxe Python...")
try:
    py_compile.compile(SRC, doraise=True)
    print("✅ Syntaxe OK — plan_classifier.py est réparé !")
    print("   Vous pouvez relancer le pipeline OCR.")
except py_compile.PyCompileError as e:
    print(f"❌ Erreur de syntaxe après réparation : {e}")
    print("   Restauration du backup...")
    shutil.copy2(BACKUP, SRC)
    print("   Backup restauré. Contactez le support.")
    sys.exit(1)
