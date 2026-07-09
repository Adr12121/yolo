with open('plan_classifier.py', 'rb') as f:
    raw = f.read()

# Fix the fallback line
old_line = b'if all_missing and type_plan in ["CROQUIS", "PVa", "GENERIC"]:'
new_line = b'if all_missing and type_plan in ["DMPC", "CROQUIS", "PVa", "GENERIC"]:'
raw = raw.replace(old_line, new_line)

# Fix '? valider' mojibake
old_valider = b'"\xef\xbf\xbd? valider"'
new_valider = b'"\xc3\x80 valider"'
raw = raw.replace(old_valider, new_valider)
raw = raw.replace(b'"? valider"', b'"\xc3\x80 valider"')

with open('plan_classifier.py', 'wb') as f:
    f.write(raw)

print('Patched DMPC fallback and A valider.')
