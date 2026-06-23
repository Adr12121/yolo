import re

file_path = 'plan_classifier.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

new_dmpc_prompts = '''"DMPC": {
        "commune":  "Quel est le nom de la commune écrit sur l'image ? Réponds uniquement par le nom de la commune, sans phrase.",
        "section":  "Quelle est la lettre de section cadastrale ? Réponds uniquement par la lettre (ex: A, B, AB).",
        "feuille":  "Quel est le numéro de feuille ? Réponds uniquement par le numéro.",
        "n_ordre":  "Quel est le numéro d'ordre ou numéro DA ? Réponds uniquement par le numéro.",
        "geometre": "Quel est le nom du géomètre ou cabinet ? Réponds uniquement par le nom, sans phrase.",
        "date":     "Quelle est la date ? Réponds uniquement par la date (format JJ/MM/AAAA).",
        "indication": "Extrais l'objet ou l'indication de ce document. Si tu vois 3 choix pré-imprimés (A, B, C) avec deux rayés, réponds UNIQUEMENT par la lettre non rayée (A, B ou C). Sinon, extrais l'indication sous forme de texte court.",
        "proprietaires_anciens": "Extrais UNIQUEMENT les noms des propriétaires (anciens propriétaires, cédants, ou propriétaires actuels). NE FAIS AUCUNE PHRASE DESCRIPTIVE. Donne juste les noms. Si illisible, réponds 'vide'.",
        "proprietaires_nouveaux": "Extrais UNIQUEMENT les noms des nouveaux propriétaires ou acquéreurs. NE FAIS AUCUNE PHRASE DESCRIPTIVE. Donne juste les noms. Si illisible, réponds 'vide'.",
        "_default": "Extrais UNIQUEMENT la valeur utile sur l'image. NE FAIS AUCUNE PHRASE. Réponds juste par le texte lu. Si illisible, réponds 'vide'.",
    },'''

# We need to replace the existing "DMPC": { ... }, block in _VLM_PROMPTS_BY_TYPE
pattern = re.compile(r'"DMPC"\s*:\s*\{.*?"_default"\s*:[^\}]+\},', re.DOTALL)
if pattern.search(content):
    content = pattern.sub(new_dmpc_prompts, content)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("DMPC prompts replaced successfully!")
else:
    print("Could not find DMPC prompts block.")
