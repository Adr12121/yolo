import re

def remove_emojis_and_fix_caps(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove specific emojis
    emojis = [
        "✨", "💡", "🏛️", "🔍", "✅", "⚠️", "📋", "🚀", "🧪", "❌", "🔄", "ℹ️", "🔎", "↩️"
    ]
    for e in emojis:
        content = content.replace(e + " ", "")
        content = content.replace(e, "")

    # Fix all caps strings, specifically targeting headers and buttons
    replacements = {
        '"EXPORT DMPC / GÉOFONCIER (Générer le fichier normé)"': '"Export DMPC / Géofoncier (générer le fichier normé)"',
        '"EXPORT CSV NORMÉ (Géofoncier)"': '"Export CSV normé (Géofoncier)"',
        'STATUT :': 'Statut :',
        '"CONFORME"': '"Conforme"',
        '"ALERTE"': '"Alerte"',
        '"REJET"': '"Rejet"',
        'CONFORME': 'Conforme',
        '>ALERTE<': '>Alerte<',
        '>REJET<': '>Rejet<',
        'À VÉRIFIER': 'À vérifier',
        'REJETÉ': 'Rejeté',
        'MATCH_UNIQUE': 'Match unique', # Wait, this might break logic if status == "MATCH_UNIQUE"
        'NO_MATCH': 'Aucun match',
        'CANDIDATS': 'Candidats',
    }

    # Be careful not to replace variable names like status == "MATCH_UNIQUE"
    # We only want to replace what's displayed to the user
    display_replacements = [
        ('EXPORT CSV NORMÉ (Géofoncier)', 'Export CSV normé (Géofoncier)'),
        ('STATUT :', 'Statut :'),
        ('À VÉRIFIER', 'À vérifier'),
        ('REJETÉ', 'Rejeté'),
        ('SECTION 1 — Résolution du Répertoire', 'Section 1 — Résolution du répertoire'),
        ('SECTION 2 — Versement Géofoncier', 'Section 2 — Versement Géofoncier'),
    ]

    for old, new in display_replacements:
        content = content.replace(old, new)
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

remove_emojis_and_fix_caps("app_validation.py")
