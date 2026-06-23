import re

with open('plan_classifier.py', encoding='utf-8') as f:
    lines = f.readlines()

line = lines[1513]  # ligne 1514 (0-indexed)
print('Ligne brute:', repr(line))

# Extraire le pattern regex entre guillemets
m = re.search(r'r"(.+?)"', line)
if m:
    pattern = m.group(1)
    print('Pattern:', repr(pattern))
    try:
        re.compile(pattern)
        print('OK: regex valide')
    except Exception as e:
        print('ERREUR:', e)
        # Correction: remplacer par \xc0-\xff
        fixed_pattern = re.sub(r'[^\x00-\x7f]+-[^\x00-\x7f]+', r'\\xc0-\\xff', pattern)
        print('Pattern fixé:', repr(fixed_pattern))
        # Appliquer dans le fichier
        new_line = line.replace(pattern, fixed_pattern)
        lines[1513] = new_line
        with open('plan_classifier.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print('Fichier corrigé')
else:
    print('Pattern non trouvé dans la ligne')
