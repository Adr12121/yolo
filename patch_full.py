import os
import re

file_path = 'plan_classifier.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Regex n_ordre
old_ordre = r'''        r"(?:document\s+modifi[e\xe9]|document\s+d['’]arpentage)[^\d]{0,20}(\d{1,6})",'''
new_ordre = r'''        r"(?:document\s+modifi[e\xe9]|document\s+d['’]arpentage)\s*(?:n[o\xb0°'’])\s*(\d{1,6})",'''
if old_ordre in content:
    content = content.replace(old_ordre, new_ordre)
else:
    print("Warning: old_ordre not found")

# 2. Zones DMPC
old_dmpc = r'''    "DMPC": {   # DMPC (jaunâtre, tapé + manuscrit)
        "commune":    [0.0, 0.0, 0.55, 0.25],  # en premier à gauche
        "section":    [0.0, 0.0, 0.55, 0.35],  # ensuite à gauche
        "feuille":    [0.0, 0.0, 0.55, 0.35],  # ensuite à gauche
        "echelle":    [0.0, 0.0, 0.55, 0.40],  # ensuite à gauche
        "n_ordre":    [0.55, 0.0, 1.0, 0.30],  # en haut à droite (DA)
        "n_dossier":  [0.55, 0.0, 1.0, 0.30],
        "date":       [0.0, 0.65, 1.0, 1.0],   # bas de page
        "geometre":   [0.45, 0.65, 1.0, 1.0],  # bas à droite (dressé par)
        "signataires":[0.0, 0.65, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.65, 1.0, 1.0], # bas de page (noms des prop)
        "proprietaires_nouveaux": [0.0, 0.65, 1.0, 1.0],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.65, 1.0, 1.0],   # propositions certifié par
    },'''
# Fallback in case of weird encodings (Ã¢ instead of â)
old_dmpc_alt = r'''    "DMPC": {   # DMPC (jaunÃ¢tre, tapÃ© + manuscrit)
        "commune":    [0.0, 0.0, 0.55, 0.25],  # en premier Ã  gauche
        "section":    [0.0, 0.0, 0.55, 0.35],  # ensuite Ã  gauche
        "feuille":    [0.0, 0.0, 0.55, 0.35],  # ensuite Ã  gauche
        "echelle":    [0.0, 0.0, 0.55, 0.40],  # ensuite Ã  gauche
        "n_ordre":    [0.55, 0.0, 1.0, 0.30],  # en haut Ã  droite (DA)
        "n_dossier":  [0.55, 0.0, 1.0, 0.30],
        "date":       [0.0, 0.65, 1.0, 1.0],   # bas de page
        "geometre":   [0.45, 0.65, 1.0, 1.0],  # bas Ã  droite (dressÃ© par)
        "signataires":[0.0, 0.65, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.65, 1.0, 1.0], # bas de page (noms des prop)
        "proprietaires_nouveaux": [0.0, 0.65, 1.0, 1.0],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.65, 1.0, 1.0],   # propositions certifiÃ© par
    },'''

new_dmpc = r'''    "DMPC": {   # DMPC (jaunâtre, tapé + manuscrit)
        "commune":    [0.0, 0.0, 1.0, 0.40],
        "section":    [0.0, 0.0, 1.0, 0.40],
        "feuille":    [0.0, 0.0, 1.0, 0.40],
        "echelle":    [0.0, 0.0, 1.0, 0.40],
        "n_ordre":    [0.0, 0.0, 1.0, 0.40],
        "n_dossier":  [0.0, 0.0, 1.0, 0.40],
        "date":       [0.0, 0.65, 1.0, 1.0],
        "geometre":   [0.0, 0.65, 1.0, 1.0],
        "signataires":[0.0, 0.65, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.25, 1.0, 1.0], 
        "proprietaires_nouveaux": [0.0, 0.25, 1.0, 1.0],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.65, 1.0, 1.0],
    },'''
if old_dmpc in content: content = content.replace(old_dmpc, new_dmpc)
elif old_dmpc_alt in content: content = content.replace(old_dmpc_alt, new_dmpc)
else: print("Warning: old_dmpc not found")

# 3. Zones PLa
old_pla = r'''    "PLa": {   # Document d'arpentage (DA) — Cartouche 1/3 haut
        "commune":    [0.0, 0.0, 0.50, 0.25],  # 1/3 haut, partie gauche
        "n_ordre":    [0.0, 0.0, 0.50, 0.35],  # 1/3 haut, partie gauche (DA)
        "section":    [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "feuille":    [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "date":       [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "echelle":    [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "geometre":   [0.0, 0.65, 1.0, 1.0],   # bas de page / signatures
        "signataires":[0.0, 0.65, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.35, 1.0, 0.70],
        "proprietaires_nouveaux": [0.0, 0.35, 1.0, 0.70],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 0.50, 0.35],
    },'''
old_pla_alt = r'''    "PLa": {   # Document d'arpentage (DA) â€” Cartouche 1/3 haut
        "commune":    [0.0, 0.0, 0.50, 0.25],  # 1/3 haut, partie gauche
        "n_ordre":    [0.0, 0.0, 0.50, 0.35],  # 1/3 haut, partie gauche (DA)
        "section":    [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "feuille":    [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "date":       [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "echelle":    [0.50, 0.0, 1.0, 0.35],  # 1/3 haut, partie droite
        "geometre":   [0.0, 0.65, 1.0, 1.0],   # bas de page / signatures
        "signataires":[0.0, 0.65, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.35, 1.0, 0.70],
        "proprietaires_nouveaux": [0.0, 0.35, 1.0, 0.70],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 0.50, 0.35],
    },'''
new_pla = r'''    "PLa": {   # Document d'arpentage (DA) — Cartouche 1/3 haut
        "commune":    [0.0, 0.0, 0.50, 0.35],
        "n_ordre":    [0.0, 0.0, 1.0, 0.40],
        "section":    [0.0, 0.0, 1.0, 0.35],
        "feuille":    [0.0, 0.0, 1.0, 0.35],
        "date":       [0.0, 0.0, 1.0, 0.35],
        "echelle":    [0.0, 0.0, 1.0, 0.35],
        "geometre":   [0.0, 0.65, 1.0, 1.0],
        "signataires":[0.0, 0.65, 1.0, 1.0],
        "proprietaires_anciens":  [0.0, 0.25, 1.0, 0.75],
        "proprietaires_nouveaux": [0.0, 0.25, 1.0, 0.75],
        "parcelles":  [0.0, 0.0, 1.0, 1.0],
        "indication": [0.0, 0.0, 1.0, 0.45],
    },'''
if old_pla in content: content = content.replace(old_pla, new_pla)
elif old_pla_alt in content: content = content.replace(old_pla_alt, new_pla)
else: print("Warning: old_pla not found")


# 4. Multi-page fusion
old_merge = '''        # ── Fusion dans doc_champs ──
        for k, v in champs.items():
            if k not in doc_champs or k == "date":
                # On ajoute le numéro de page pour l'interface de validation
                if isinstance(v, dict):
                    v["page"] = page_num
                doc_champs[k] = v'''

new_merge = '''        # ── Fusion dans doc_champs ──
        for k, v in champs.items():
            if isinstance(v, dict):
                v["page"] = page_num
            if k not in doc_champs:
                doc_champs[k] = v
            else:
                val_exist = doc_champs[k].get("valeur", "") if isinstance(doc_champs[k], dict) else doc_champs[k]
                val_new = v.get("valeur", "") if isinstance(v, dict) else v
                conf_exist = doc_champs[k].get("confidence", 0.0) if isinstance(doc_champs[k], dict) else 0.0
                conf_new = v.get("confidence", 0.0) if isinstance(v, dict) else 0.0
                
                if k == "parcelles":
                    try:
                        import ast
                        l1 = ast.literal_eval(val_exist) if isinstance(val_exist, str) else val_exist
                        l2 = ast.literal_eval(val_new) if isinstance(val_new, str) else val_new
                        if isinstance(l1, list) and isinstance(l2, list):
                            v["valeur"] = str(list(set(l1 + l2)))
                            doc_champs[k] = v
                    except:
                        if not val_exist or (val_new and conf_new > conf_exist): doc_champs[k] = v
                elif not val_exist or str(val_exist).strip() == "[]":
                    doc_champs[k] = v
                elif val_new and str(val_new).strip() != "[]" and conf_new > conf_exist:
                    doc_champs[k] = v'''

if old_merge in content: content = content.replace(old_merge, new_merge)
else: print("Warning: old_merge not found")


# 5. Date validation
date_hook = '''    elif field_type == "date":
        # Normaliser années à 2 chiffres (14/10/97 -> 14/10/1997)'''
date_add = '''    elif field_type == "date":
        if re.search(r'(?i)(?:n[eé]e?|décédé|d[eé]c[eé]d[eé]|arr[eê]t[eé]|loi|d[eé]cret|naissance|n[aâ]quit)', text):
            return None
        # Normaliser années à 2 chiffres (14/10/97 -> 14/10/1997)'''
if date_hook in content: content = content.replace(date_hook, date_add)
else: print("Warning: date_hook not found")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch OK")
