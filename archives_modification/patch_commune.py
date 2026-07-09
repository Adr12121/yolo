import os
import sys

with open('plan_classifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"plan de",', '"plan de", "feuille", "n°", "numero", "date", "d",')
content = content.replace('{"commune", "territoire", "section", "echelle", "plan", "bornage"}', '{"commune", "territoire", "section", "echelle", "plan", "bornage", "feuille", "date", "d"}')
content = content.replace(r'(?i)^commune\s*(?:de\s*)?(?::|\-)?\s*(.+)$', r'(?i)^commu[ni]e?\s*(?:de\s*)?(?::|\-)?\s*(.+)$')
content = content.replace('val.lower() in ("commune", "commune de"):', 'val.lower() in ("commune", "commune de") or __import__("re").match(r"(?i)^commu[ni]e?$", val):')

with open('plan_classifier.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch applied successfully.")
