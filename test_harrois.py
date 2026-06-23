from main import load_commune_db, load_models
from plan_classifier import process_plan
import json

if __name__ == "__main__":
    db = load_commune_db('ardeche.json', 'villes_07_26.txt')
    models = load_models()
    res = process_plan("inputs/EXEMPLE_HARROIS.pdf", models=models, commune_db=db)
    print("FINISHED")
    print(json.dumps(res, indent=2))
