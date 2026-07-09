import os
import re

file_path = 'plan_classifier.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

classify_pattern = re.compile(r'def classify_plan\(filepath: str\) -> str:.*?(?=\ndef is_plan_document)', re.DOTALL)

new_classify_func = '''def classify_plan(filepath: str) -> str:
    import fitz
    import os
    import re
    
    # 1. Analyse physique du PDF (le juge de paix)
    try:
        doc = fitz.open(filepath)
        txt = doc[0].get_text("text").strip()
        doc.close()
        is_vector = len(txt) > 50
    except Exception:
        is_vector = False
        txt = ""
        
    name = os.path.basename(filepath).lower()
    
    # 2. Routage intelligent basé sur la nature physique
    if is_vector:
        print(f"  [Classifier] Document VECTORIEL propre détecté ({len(txt)} chars). Force type DMPC (Recherche par zones).")
        return "DMPC"
    else:
        print(f"  [Classifier] Document SCANNÉ (image) détecté. Mode contextuel/manuscrit.")
        if re.search(r'pva|1pva|pvb|pvc|proc[eè]s.?verbal', name):
            return "PVa"
        if re.search(r'2pla|1pla|pla|lotissement|division', name):
            return "PLa"
        return "GENERIC"
'''

content = classify_pattern.sub(new_classify_func, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Patch Classify OK")
