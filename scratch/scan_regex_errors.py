"""
Scanne toutes les raw strings (r"...") dans plan_classifier.py
et tente de les compiler comme regex pour détecter les erreurs.
"""
import re

path = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py'

with open(path, encoding='utf-8') as f:
    lines = f.readlines()

errors = []
for i, line in enumerate(lines, 1):
    # Chercher toutes les raw strings
    for m in re.finditer(r'r"([^"\\]|\\.)*"', line):
        raw_str = m.group(0)[2:-1]  # enlever r" et "
        try:
            re.compile(raw_str)
        except re.error as e:
            errors.append((i, raw_str[:80], str(e)))

if errors:
    print(f"ERREURS TROUVEES ({len(errors)}):")
    for lineno, pat, err in errors:
        print(f"  Ligne {lineno}: {err}")
        print(f"    Pattern: {repr(pat)}")
else:
    print("OK: Aucune regex invalide trouvée dans plan_classifier.py")
