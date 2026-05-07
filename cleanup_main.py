"""
Script de nettoyage : tronque main.py à la ligne 2957.
Toutes les lignes après sont du vieux code dupliqué.
"""
import sys

filepath = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\main.py'
keep_lines = 2957  # Garder lignes 1..2957 (index 0..2956)

with open(filepath, encoding='utf-8') as f:
    lines = f.readlines()

print(f"Fichier actuel : {len(lines)} lignes")

# Vérification de sécurité
line_2957 = lines[2956].strip() if len(lines) >= 2957 else ''
print(f"Ligne 2957 : {line_2957!r}")

if input("Confirmer la troncature ? (oui/non) : ").strip().lower() == 'oui':
    kept = lines[:keep_lines]
    # Nettoyer les lignes vides à la fin
    while kept and kept[-1].strip() == '':
        kept.pop()
    kept.append('\n')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(kept)
    print(f"Done. Fichier tronqué à {len(kept)} lignes.")
else:
    print("Annulé.")
