"""
Script de reparation complète du fichier plan_classifier.py
Corrige:
1. Le triple-quote manquant au debut du fichier
2. Ecrit le fichier en UTF-8 sans BOM
"""

with open('plan_classifier.py', 'rb') as f:
    raw = f.read()

# Corriger le debut : manque un " au debut (triple quotes cassé par PowerShell)
if raw.startswith(b'""'):
    raw = b'"' + raw
    print("Fixed: added missing triple-quote opening")

# Ecrire en UTF-8 sans BOM
with open('plan_classifier.py', 'wb') as f:
    f.write(raw)

print("File written successfully.")

# Verifier la syntaxe
import ast
content = raw.decode('utf-8', errors='replace')
try:
    ast.parse(content)
    print("SYNTAXE OK")
except SyntaxError as e:
    print(f"SyntaxError at line {e.lineno}: {e.msg}")
    # Afficher le contexte
    lines = content.split('\n')
    start = max(0, e.lineno - 3)
    end = min(len(lines), e.lineno + 3)
    for i, l in enumerate(lines[start:end], start=start+1):
        marker = ">>>" if i == e.lineno else "   "
        print(f"{marker} {i}: {repr(l[:100])}")
