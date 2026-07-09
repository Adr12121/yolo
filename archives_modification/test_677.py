import os
import json
from main import process_modern_plan

# Load commune DB
try:
    with open('communes_france.json', 'r', encoding='utf-8') as f:
        commune_db = json.load(f)
except:
    with open('ardeche.json', 'r', encoding='utf-8') as f:
        commune_db = json.load(f)

# Run process_modern_plan
res = process_modern_plan(
    "inputs/geofoncier_dmpc_07289_000_677.pdf",
    commune_db=commune_db,
    geometre_id="geofoncier_dmpc_07289_000_677"
)

# Print specific results
print("RESULTATS EXTRAITS:")
for k, v in res.items():
    if isinstance(v, dict) and "valeur" in v:
        print(f"  {k}: {v['valeur']} (brut: {v.get('brut', '')})")
    else:
        print(f"  {k}: {v}")
