import os
import sys
import fitz
import json
from plan_classifier import process_plan
import easyocr

if __name__ == "__main__":
    reader = easyocr.Reader(['fr'], gpu=False)
    # mock models
    models_charges = {"ocr": reader}
    
    # Load commune DB
    from main import load_commune_db
    commune_db = load_commune_db('ardeche.json', 'villes_07_26.txt')

    file_path = "inputs/1992C100001_a094147_1PVa.pdf"
    
    print(f"Testing on {file_path}")
    result = process_plan(file_path, models=models_charges, commune_db=commune_db)
    
    print("--------------------------------------------------")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
