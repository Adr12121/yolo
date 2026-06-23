import re

path = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py'
with open(path, encoding='utf-8') as f:
    content = f.read()

# Find the malformed line from the PowerShell replace
old_bad = "crops_data = _generate_crops(`n                all_ocr_page, list(champs_attendus), (h, w),`n                zone_constraints=zones_def,`n            )"

# Also try original (in case the bad one didn't apply)
old_orig = "crops_data = _generate_crops(all_ocr_page, list(champs_attendus), (h, w))"

new = (
    "crops_data = _generate_crops(\n"
    "                all_ocr_page, list(champs_attendus), (h, w),\n"
    "                zone_constraints=zones_def,\n"
    "            )"
)

if old_bad in content:
    content = content.replace(old_bad, new)
    print("Patched: replaced bad PowerShell version")
elif old_orig in content:
    content = content.replace(old_orig, new)
    print("Patched: replaced original version")
else:
    # Search for any variant
    idx = content.find("_generate_crops")
    while idx != -1:
        print(f"Found at {idx}: {repr(content[idx:idx+120])}")
        idx = content.find("_generate_crops", idx + 1)
    print("NOT PATCHED - inspect output above")
    exit(1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("File saved.")
