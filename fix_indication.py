import codecs

with codecs.open("plan_classifier.py", "r", "utf-8", errors="ignore") as f:
    text = f.read()

# 1. Fix Mojibake
text = text.replace("Ã€-Ã¿", "À-ÿ")
text = text.replace("Ã€", "À")
text = text.replace("Ã©", "é")
text = text.replace("Ã¨", "è")
text = text.replace("Ãª", "ê")
text = text.replace("Ã¢", "â")
text = text.replace("Ã®", "î")
text = text.replace("Ã¯", "ï")
text = text.replace("Ã´", "ô")
text = text.replace("Ã¹", "ù")
text = text.replace("Ã»", "û")
text = text.replace("Ã§", "ç")
text = text.replace("Ã", "à") # sometimes Ã alone is à
text = text.replace("Â", "")  # remove trailing bytes if left alone

# 2. Add indication to DMPC prompt
target_dmpc = '"date":     "Quelle est la date'
if target_dmpc in text and 'indication' not in text[text.find(target_dmpc):text.find(target_dmpc)+500]:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if target_dmpc in line:
            lines.insert(i + 1, '        "indication": "Ce document contient 3 choix pré-imprimés concernant l\'arpentage/bornage. Deux choix sont rayés au stylo. Trouve le seul choix qui N\'EST PAS rayé et réponds uniquement par la lettre correspondante (A, B ou C) :\\n A - d\'après les indications qu\'ils ont fournies au bureau\\n B - en conformité d\'un piquetage qu\'ils ont effectué sur le terrain\\n C - d\'après un plan d\'arpentage ou de bornage, dont copie ci-jointe",')
            break
    text = '\n'.join(lines)

# 3. Add indication logic to _clean_vlm_response
target_clean = 'def _clean_vlm_response(val: str, field: str = "") -> str:'
if target_clean in text and 'field == "indication"' not in text:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if target_clean in line:
            for j in range(i, i+15):
                if 'return val' in lines[j] and 'if not val:' in lines[j-1]:
                    insert_idx = j + 1
                    lines.insert(insert_idx, '    if field == "indication":')
                    lines.insert(insert_idx + 1, '        val_clean = val.strip().upper()')
                    lines.insert(insert_idx + 2, '        if "A" in val_clean and "B" not in val_clean and "C" not in val_clean:')
                    lines.insert(insert_idx + 3, '            return "d\'après les indications qu\'ils ont fournies au bureau"')
                    lines.insert(insert_idx + 4, '        elif "B" in val_clean and "A" not in val_clean and "C" not in val_clean:')
                    lines.insert(insert_idx + 5, '            return "en conformité d\'un piquetage qu\'ils ont effectué sur le terrain"')
                    lines.insert(insert_idx + 6, '        elif "C" in val_clean and "A" not in val_clean and "B" not in val_clean:')
                    lines.insert(insert_idx + 7, '            return "d\'après un plan d\'arpentage ou de bornage, dont copie ci-jointe"')
                    break
            break
    text = '\n'.join(lines)

# 4. Update calls to _clean_vlm_response
text = text.replace('raw = _clean_vlm_response(raw)', 'raw = _clean_vlm_response(raw, field)')
text = text.replace('val = _clean_vlm_response(val)', 'val = _clean_vlm_response(val, f)')

with codecs.open("plan_classifier.py", "w", "utf-8") as f:
    f.write(text)
