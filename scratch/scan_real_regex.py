"""
Cherche uniquement les regex dans des appels re.sub/re.search/re.compile/re.match/re.findall
et teste uniquement celles-là.
"""
import re

path = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py'

with open(path, encoding='utf-8') as f:
    content = f.read()
    lines = content.splitlines()

# Pattern pour les appels re.XXX(r"...", ...) 
RE_CALL = re.compile(r're\.\w+\(\s*r"((?:[^"\\]|\\.)*)"\s*[,\)]')

errors = []
for i, line in enumerate(lines, 1):
    for m in RE_CALL.finditer(line):
        pat = m.group(1)
        try:
            re.compile(pat)
        except re.error as e:
            errors.append((i, pat[:100], str(e)))
            
if errors:
    print(f"VRAIES ERREURS REGEX ({len(errors)}):")
    for lineno, pat, err in errors:
        print(f"  Ligne {lineno}: {err}")
        print(f"    Pattern: {repr(pat)}")
else:
    print("OK: Aucune erreur dans les vrais appels re.xxx() de plan_classifier.py")
    
# Test spécifique de la ligne 1514 (index 1513)
line_1514 = lines[1513]
print(f"\nLigne 1514: {repr(line_1514.strip())}")
m = RE_CALL.search(line_1514)
if m:
    try:
        re.compile(m.group(1))
        print("Ligne 1514: OK")
    except re.error as e:
        print(f"Ligne 1514: ERREUR {e}")
