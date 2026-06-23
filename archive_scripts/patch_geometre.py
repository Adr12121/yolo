import re, sys, shutil
from pathlib import Path

TARGET = Path(__file__).parent / "plan_classifier.py"

with open(TARGET, encoding="utf-8") as f:
    content = f.read()

lines = content.splitlines(keepends=True)
n = len(lines)

target_start = None
target_end   = None

for i, line in enumerate(lines):
    if "Fallback" in line and "ometre" in line and "PVa" in line and "not in champs" not in line:
        j = i
        while j < n and not (lines[j].strip() == "" and j > i + 3):
            j += 1
        target_start = i
        target_end   = j
        break

if target_start is None:
    for i, line in enumerate(lines):
        if '"geometre" not in champs' in line and i > 100:
            context = "".join(lines[i:i+5])
            if "full_text_page_low" in context:
                j = i - 1
                while j >= 0 and lines[j].strip().startswith("#"):
                    j -= 1
                target_start = j + 1
                k = i
                while k < n:
                    if lines[k].strip() == "" and k > i + 8:
                        break
                    k += 1
                target_end = k
                break

if target_start is None:
    sys.exit(1)

NEW_BLOCK = '''            # -- Fallback geometre pour PVa --
            # Chercher "geometre-expert" ou "dresse par" dans le texte de la page.
            # GARDE : ne pas activer sur [signatures] (boilerplate juridique uniquement).
            if "geometre" not in champs and role != "signatures":
                m_geo = re.search(
                    r\'(?:g[e\\xe9]om[e\\xe8]tre[s\\-]?(?:\\s*expert)?|dress[e\\xe9]\\s+par|le\\s+soussign[e\\xe9]|op[e\\xe9]rateur)\\s*[:\\-\\s]*\'
                    r\'([A-Z\\xc0-\\xdd][A-Za-z\\xc0-\\xff\\s\\-\\.]{2,40})\',
                    full_text_page, re.IGNORECASE
                )
                if m_geo:
                    geo_val = m_geo.group(1).strip().rstrip(".,;:")
                    geo_val = re.sub(r\'\\s+\', \' \', geo_val)
                    # Blacklist etendue : phrases des pages de signatures
                    _GEO_BOILERPLATE_FB = [
                        "soussign", "accepte", "reserve", "certifi", "signature",
                        "approuve", "mentions", "parties", "inscription",
                        "conservation", "commune", "section", "date et",
                    ]
                    _est_boilerplate = (
                        any(frag in geo_val.lower() for frag in _GEO_BOILERPLATE_FB)
                        or bool(re.search(r\'(?i)expert$\', geo_val))
                    )
                    if len(geo_val) > 3 and not re.match(r\'^\\d\', geo_val) and not _est_boilerplate:
                        champs["geometre"] = {
                            "valeur": geo_val,
                            "zone": [0.0, 0.55, 1.0, 1.0],
                            "brut": m_geo.group(0).strip()
                        }
                        print(f"    [geometre] -> \'{geo_val}\' (Fallback PVa role={role})")'''

backup = TARGET.with_suffix(".py.bak_geo")
shutil.copy2(TARGET, backup)

new_lines = lines[:target_start] + [NEW_BLOCK + "\n\n"] + lines[target_end:]
new_content = "".join(new_lines)

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Patch applique avec succes !")
