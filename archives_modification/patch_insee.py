import sys
import re

def patch_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return

    # 1. Patch _match_commune
    old_match = '''    try:
        from rapidfuzz import process as rfp, fuzz
    except ImportError:
        return text'''
        
    new_match = '''    try:
        from rapidfuzz import process as rfp, fuzz
    except ImportError:
        return text

    # Handle INSEE code (e.g., '07289' or '289')
    import re
    text_clean = text.strip()
    if re.match(r'^(?:07)?\d{3}$', text_clean):
        code_insee = text_clean if len(text_clean) == 5 else f"07{text_clean}"
        for c in commune_db:
            if str(c.get("code", "")) == code_insee:
                return c["officiel"]
        return text'''
        
    if old_match in content:
        content = content.replace(old_match, new_match)
        print(f"Patched _match_commune in {filepath}")

    # 2. Patch validate_field in plan_classifier.py
    if 'plan_classifier.py' in filepath:
        old_val = '''        COMMUNES_INTERDITES = {"commune", "territoire", "section", "echelle", "plan", "bornage", "feuille", "date", "d"}
        val_norm = val.lower()
        if val_norm in COMMUNES_INTERDITES:
            return None
        if len(val) < 4 or len(val) > 50:
            return None
        if sum(c.isdigit() for c in val) > 2:
            return None'''
            
        new_val = '''        COMMUNES_INTERDITES = {"commune", "territoire", "section", "echelle", "plan", "bornage", "feuille", "date", "d"}
        val_norm = val.lower()
        if val_norm in COMMUNES_INTERDITES:
            return None
            
        import re
        if re.match(r'^(?:07)?\d{3}$', val.strip()):
            return val.strip()
            
        if len(val) < 4 or len(val) > 50:
            return None
        if sum(c.isdigit() for c in val) > 2:
            return None'''
            
        if old_val in content:
            content = content.replace(old_val, new_val)
            print(f"Patched validate_field in {filepath}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

patch_file('plan_classifier.py')
patch_file('modern_plan_extractor.py')
patch_file('app_validation.py')
