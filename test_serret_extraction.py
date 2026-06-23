import sys
import os
import json

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from plan_classifier import process_plan

# Load commune DB
print("Loading commune DB...")
commune_db = []
try:
    with open("ardeche.json", "r", encoding="utf-8") as f:
        commune_db = json.load(f)
except Exception as e:
    print(f"Error loading commune DB: {e}")

print("Processing EXEMPLE_SERRET.pdf...")
pdf_path = r"inputs\EXEMPLE_SERRET.pdf"

if not os.path.exists(pdf_path):
    print("Test file not found.")
    sys.exit(1)

res = process_plan(pdf_path, models=None, commune_db=commune_db)

print("\n--- RESULTS ---")
print("Commune: ", res.get("champs", {}).get("commune", {}))
print("N_Ordre: ", res.get("champs", {}).get("n_ordre", {}))

# Sauvegarder dans output/ pour vérifier
out_path = os.path.join("outputs", "EXEMPLE_SERRET_TEST.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(res.get("champs", {}), f, indent=4, ensure_ascii=False)
print(f"Resultats sauvegardes dans {out_path}")
