import codecs
import re

file_path = 'plan_classifier.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

# 1. Update _match_commune for INSEE
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
    print("Patched _match_commune")

# 2. Update _validate_field for INSEE
old_val = '''        COMMUNES_INTERDITES = {"commune", "territoire", "section", "echelle", "plan", "bornage", "feuille", "date", "d"}
        val_norm = val.lower()
        if val_norm in COMMUNES_INTERDITES:
            return None'''
new_val = '''        COMMUNES_INTERDITES = {"commune", "territoire", "section", "echelle", "plan", "bornage", "feuille", "date", "d"}
        val_norm = val.lower()
        if val_norm in COMMUNES_INTERDITES:
            return None
            
        import re
        if re.match(r'^(?:07)?\d{3}$', val.strip()):
            return val.strip()'''
if old_val in content:
    content = content.replace(old_val, new_val)
    print("Patched _validate_field for INSEE")
elif 'if re.match(r"^(?:07)?\d{3}$", val.strip()):' not in content:
    # Try another way to patch it
    old_val2 = '''        val_norm = val.lower()
        if val_norm in COMMUNES_INTERDITES:
            return None'''
    new_val2 = '''        val_norm = val.lower()
        if val_norm in COMMUNES_INTERDITES:
            return None
        if re.match(r'^(?:07)?\d{3}$', val.strip()):
            return val.strip()'''
    if old_val2 in content:
        content = content.replace(old_val2, new_val2)
        print("Patched _validate_field for INSEE (fallback)")

# 3. Update communes rules
content = content.replace('"plan de",', '"plan de", "feuille", "n°", "numero", "date", "d",')
content = content.replace('{"commune", "territoire", "section", "echelle", "plan", "bornage"}', '{"commune", "territoire", "section", "echelle", "plan", "bornage", "feuille", "date", "d"}')

# 4. Update indication logic in _clean_vlm_response
if 'def _clean_vlm_response(val: str, field: str = "") -> str:' not in content:
    content = content.replace('def _clean_vlm_response(val: str) -> str:', 'def _clean_vlm_response(val: str, field: str = "") -> str:')

clean_insert = '''    if not val:
        return val'''
clean_replacement = '''    if not val:
        return val
        
    if field == "indication":
        val_clean = val.strip().upper()
        if "A" in val_clean and "B" not in val_clean and "C" not in val_clean:
            return "d'après les indications qu'ils ont fournies au bureau"
        elif "B" in val_clean and "A" not in val_clean and "C" not in val_clean:
            return "en conformité d'un piquetage qu'ils ont effectué sur le terrain"
        elif "C" in val_clean and "A" not in val_clean and "B" not in val_clean:
            return "d'après un plan d'arpentage ou de bornage, dont copie ci-jointe"'''

if 'field == "indication"' not in content:
    content = content.replace(clean_insert, clean_replacement)
    print("Patched _clean_vlm_response for indication")

# Update calls to _clean_vlm_response
content = content.replace('raw = _clean_vlm_response(raw)', 'raw = _clean_vlm_response(raw, field)')
content = content.replace('val = _clean_vlm_response(val)', 'val = _clean_vlm_response(val, field)')

# 5. Add indication to DMPC prompt
dmpc_prompt_marker = '"date":     "Quelle est la date'
dmpc_prompt_insert = '''        "indication": "Ce document contient 3 choix pré-imprimés concernant l'arpentage/bornage. Deux choix sont rayés au stylo. Trouve le seul choix qui N'EST PAS rayé et réponds uniquement par la lettre correspondante (A, B ou C) :\\n A - d'après les indications qu'ils ont fournies au bureau\\n B - en conformité d'un piquetage qu'ils ont effectué sur le terrain\\n C - d'après un plan d'arpentage ou de bornage, dont copie ci-jointe",'''

if dmpc_prompt_marker in content and 'indication": "Ce document' not in content:
    content = content.replace(dmpc_prompt_marker, dmpc_prompt_insert + '\n' + '        ' + dmpc_prompt_marker)
    print("Patched DMPC prompt for indication")

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(content)
print("All patches injected into plan_classifier.py.")
