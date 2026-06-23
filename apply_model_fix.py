import codecs

file_path = 'plan_classifier.py'
with codecs.open(file_path, 'r', 'utf-8') as f:
    content = f.read()

# Fix 1: Change DMPC model to llama3.2-vision
old_model = '"DMPC":    "llava",'
new_model = '"DMPC":    "llama3.2-vision",'
if old_model in content:
    content = content.replace(old_model, new_model)
    print("Changed DMPC to llama3.2-vision")

# Fix 2: Add DMPC to full page fallback
old_fallback = 'if all_missing and type_plan in ["CROQUIS", "PVa", "GENERIC"]:'
new_fallback = 'if all_missing and type_plan in ["DMPC", "CROQUIS", "PVa", "GENERIC"]:'
if old_fallback in content:
    content = content.replace(old_fallback, new_fallback)
    print("Added DMPC to full page fallback")

# Fix 3: Fix "? valider" mojibake just in case
content = content.replace('"\xef\xbf\xbd? valider"', '"À valider"')
content = content.replace('"? valider"', '"À valider"')

with codecs.open(file_path, 'w', 'utf-8') as f:
    f.write(content)
