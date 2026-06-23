import codecs
import re

file_path = 'plan_classifier.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

target_pattern = r"(val = val\.replace\('\"', ''\)\.replace\(\"'\", \"\"\)\.strip\(\" '\.:,\"\)\s*\n\s*return val)"

replacement = '''val = val.replace('"', '').replace("'", "").strip(" '.:,")
    if field == "indication":
        val_clean = val.strip().upper()
        if "A" in val_clean and "B" not in val_clean and "C" not in val_clean:
            return "d'après les indications qu'ils ont fournies au bureau"
        elif "B" in val_clean and "A" not in val_clean and "C" not in val_clean:
            return "en conformité d'un piquetage qu'ils ont effectué sur le terrain"
        elif "C" in val_clean and "A" not in val_clean and "B" not in val_clean:
            return "d'après un plan d'arpentage ou de bornage, dont copie ci-jointe"
    return val'''

if re.search(target_pattern, content):
    content = re.sub(target_pattern, replacement, content)
    print("Injected indication into _clean_vlm_response")
else:
    print("Failed to find target in _clean_vlm_response")

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(content)
