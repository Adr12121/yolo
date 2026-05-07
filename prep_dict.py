import json
import re
import os

def clean_name(name):
    # Enlever les accents, majuscules et caractères spéciaux pour la comparaison
    import unicodedata
    name = str(name).strip().upper()
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = re.sub(r'[^A-Z0-9 ]', ' ', name)
    return ' '.join(name.split())

try:
    with open('drome.json', 'r', encoding='utf-8') as f:
        drome = json.load(f)
except Exception as e:
    print(f"Erreur drome.json: {e}")
    drome = []

try:
    with open('ardeche.json', 'r', encoding='utf-8') as f:
        ardeche = json.load(f)
except Exception as e:
    print(f"Erreur ardeche.json: {e}")
    ardeche = []

communes = set()
for c in drome + ardeche:
    nom = c.get('nom', '')
    if nom:
        communes.add(clean_name(nom))

print(f"Noms uniques de communes : {len(communes)}")
with open('villes_07_26.txt', 'w', encoding='utf-8') as f:
    for c in sorted(list(communes)):
        f.write(c + '\n')
print("Dictionnaire enregistré dans villes_07_26.txt")
