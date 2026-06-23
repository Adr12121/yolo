"""
Corrige les regex corrompues par le mojibake dans plan_classifier.py.
Le pattern [^A-Za-zÀ-ÿ\s\-] a été stocké comme [^A-Za-zÃ€-Ã¿\s\-]
ce qui crée un "bad character range" fatal sous Python 3.12.
"""
import re

path = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py'

with open(path, encoding='utf-8') as f:
    content = f.read()

# Remplacement de toutes les variantes corrompues de la plage de caractères accentués
replacements = [
    # Forme mojibake la plus fréquente : À-ÿ corrompu
    (r'[^A-Za-z\xc3\x80-\xc3\xbf\\s\\-]', r'[^A-Za-z\xc0-\xff\\s\\-]'),
    # Dans les chaînes littérales du fichier (telles que lues en UTF-8)
    ('Ã€-Ã¿', 'À-ÿ'),
    ('Ã€-Ã½', 'À-ý'),
    # Autres variantes possibles
    ('\\xc3\\x80-\\xc3\\xbf', '\\xc0-\\xff'),
]

original = content

# Remplacer directement les bytes corrompus dans la string lue en utf-8
# La chaîne r"[^A-Za-zÃ€-Ã¿\s\-]" en UTF-8 contient les bytes de À et ÿ encodés deux fois
# On remplace la séquence visuelle corrompue par la séquence Unicode correcte
CORRUPTED = '[^A-Za-z\u00c3\u0080-\u00c3\u00bf\\s\\-]'  # Comment ça apparaît en UTF-8 mal lu
CORRECT =   '[^A-Za-z\u00c0-\u00ff\\s\\-]'               # La vraie plage Unicode

count = content.count(CORRUPTED)
if count > 0:
    content = content.replace(CORRUPTED, CORRECT)
    print(f"Remplacé {count} occurrence(s) de la plage corrompue (méthode 1)")

# Méthode 2 : remplacer la représentation string brute
CORRUPTED2 = 'Ã€-Ã¿'  # tel qu'on le voit dans le viewer
CORRECT2 = 'À-ÿ'
count2 = content.count(CORRUPTED2)
if count2 > 0:
    content = content.replace(CORRUPTED2, CORRECT2)
    print(f"Remplacé {count2} occurrence(s) supplémentaire(s) (méthode 2)")

if content == original:
    # Dernière tentative : lire le fichier en latin-1 et chercher
    with open(path, encoding='latin-1') as f:
        raw = f.read()
    # Chercher le pattern avec les vraies bytes
    hits = [(m.start(), m.group()) for m in re.finditer(r'\[.*?\\s\\\\-\]', raw)]
    print(f"Méthode 3 - Occurrences trouvées en latin-1: {hits[:5]}")
    print("AUCUN remplacement effectué - vérifier le fichier manuellement")
else:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fichier sauvegardé avec succès")
    
    # Vérification rapide: tenter de compiler le regex
    try:
        re.compile(CORRECT)
        print(f"✓ Regex '{CORRECT}' valide")
    except Exception as e:
        print(f"✗ Regex '{CORRECT}' invalide: {e}")
