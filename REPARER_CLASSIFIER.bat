@echo off
echo Reparation de plan_classifier.py...
cd /d "%~dp0"

python -c "
import shutil, py_compile, sys

src = 'plan_classifier.py'
bak = 'plan_classifier.py.bak'
shutil.copy2(src, bak)

with open(src, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

i1, i2 = None, None
for i, l in enumerate(lines):
    if 'val_norm.startswith(interdit.lower()    elif' in l:
        i1 = i
    if 'return Noneier' in l:
        i2 = i

if i1 is None or i2 is None:
    print('Corruptions non trouvees. Fichier deja repare ?')
    sys.exit(0)

bloc = '''    for interdit in LABEL_INTERDITS:
        if val_norm == interdit.lower() or val_norm.startswith(interdit.lower() + \" \"):
            return None

    if field_type == \"section\":
        m = re.search(r\"\\\\b([A-Za-z]{1,2}|\\\\d{1,3})\\\\b\", val)
        if m:
            clean = m.group(1).upper()
            if re.match(r\"^[018]$\", clean) and len(val) <= 3:
                clean = clean.replace(\"0\", \"O\").replace(\"1\", \"I\").replace(\"8\", \"B\")
            return clean
        return None

    elif field_type == \"feuille\":
        m = re.search(r\"\\\\b(\\\\d{1,4}[A-Za-z]?|[A-Za-z]\\\\d{1,3})\\\\b\", val)
        if m:
            return m.group(1).upper()
        return None

    elif field_type == \"echelle\":
        if any(u in val_norm for u in [\"ca\", \"ha\", \"m2\", \" a \", \"m\\u00b2\"]):
            return None
        m = re.search(r\"(1\\\\s*[/:]\\\\s*\\\\d{3,5}|\\\\d{3,5})\", val)
        if m:
            return m.group(1).replace(\" \", \"\")
        return None

    elif field_type in (\"n_ordre\", \"n_dossier\"):
        if any(u in val_norm for u in [\" ha\", \" a \", \" ca\", \"m2\", \"nee\", \"demeurant\", \"epouse\"]):
            return None
        m_recent = re.search(r\"\\\\b([A-Z]\\\\d{2}[.\\\\-]?\\\\d{2,5})\\\\b|\\\\b(\\\\d{4}[.\\\\-]\\\\d{2,5})\\\\b\", val.upper())
        if m_recent:
            return (m_recent.group(1) or m_recent.group(2)).replace(\" \", \"\")
        m_da = re.search(r\"(\\\\b\\\\d{1,5}\\\\s*[A-Z]\\\\b|\\\\b\\\\d{1,5}\\\\b)\", val.upper())
        if m_da:
            cleaned = re.sub(r\"\\\\s+\", \"\", m_da.group(1))
            if re.match(r\"^(19|20)\\\\d{2}$\", cleaned):
                return None
            return cleaned
        clean = re.sub(r\"[^A-Za-z0-9\\\\-\\\\/\\\\.]\", \"\", val)
        if 2 <= len(clean) <= 20 and re.search(r\"\\\\d\", clean):
            return clean
        return None

'''

new_lines = lines[:i1] + [bloc] + lines[i2 + 1:]

with open(src, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

try:
    py_compile.compile(src, doraise=True)
    print('SUCCES : plan_classifier.py repare, syntaxe OK')
except py_compile.PyCompileError as e:
    print('ERREUR syntaxe : ' + str(e))
    shutil.copy2(bak, src)
    print('Backup restaure.')
    sys.exit(1)
"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================
    echo   plan_classifier.py repare avec succes !
    echo   Vous pouvez relancer le pipeline OCR.
    echo ============================================
) else (
    echo ERREUR lors de la reparation.
)
pause
