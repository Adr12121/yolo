import sys
import re

with open('c:\\Users\\Topo_4\\Documents\\AT_PFE\\Anti\\yolo\\plan_classifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern_sec_feu = r'(# [^\n]*Fallback Global Infaillible pour Section et Feuille[^\n]*\n(?:[ \t]+.*\n){1,20}\s*print\(f"    \[feuille\][^\n]*\))'

replacement_sec_feu = r"""            # ── Fallback Global Infaillible pour Section et Feuille ──
            if "section" not in champs:
                m_sec = re.search(r'(?i)\bsection\s+([A-Z]{1,2})\b(?!\s*°)', full_text_page)
                if m_sec:
                    champs["section"] = {
                        "valeur": m_sec.group(1).upper(),
                        "zone": [0.0, 0.0, 1.0, 1.0],
                        "brut": m_sec.group(0)
                    }
                    print(f"    [section] -> '{m_sec.group(1).upper()}' (Fallback Global)")
            if "feuille" not in champs:
                m_feu = re.search(r'(?i)\bfeuille\s+(?:n[o°])?\s*(\d{1,3})\b', full_text_page)
                if m_feu:
                    champs["feuille"] = {
                        "valeur": m_feu.group(1),
                        "zone": [0.0, 0.0, 1.0, 1.0],
                        "brut": m_feu.group(0)
                    }
                    print(f"    [feuille] -> '{m_feu.group(1)}' (Fallback Global)")

            # ── Fallback Global Infaillible pour Indication (Objet) ──
            if "indication" not in champs or not champs.get("indication", {}).get("valeur") or len(champs.get("indication", {}).get("valeur", "")) > 40:
                m_obj = re.search(r'(?i)(DIVISION|LOTISSEMENT|ARPENTAGE|REMEMBREMENT|MODIFICATIF PARCELLAIRE|ALIGNEMENT)', full_text_page)
                if m_obj:
                    champs["indication"] = {
                        "valeur": m_obj.group(1).upper(),
                        "zone": [0.0, 0.0, 1.0, 1.0],
                        "brut": f"Scanner global objet: {m_obj.group(0)}"
                    }
                    print(f"    [indication] -> '{m_obj.group(1).upper()}' (Fallback Global)")

            # ── Fallback Global Infaillible pour Date ──
            if "date" not in champs or not champs.get("date", {}).get("valeur"):
                dates_trouvees = re.findall(
                    r'\b(\d{1,2}\s*[/\-\.]\s*\d{1,2}\s*[/\-\.]\s*(?:19|20)\d{2})\b'
                    r'|\b(\d{1,2}\s+(?:janvier|f[eé]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[eé]cembre)\s+(?:19|20)\d{2})\b',
                    full_text_page, re.IGNORECASE
                )
                if dates_trouvees:
                    best_date = dates_trouvees[-1][0] or dates_trouvees[-1][1]
                    champs["date"] = {
                        "valeur": best_date,
                        "zone": [0.0, 0.0, 1.0, 1.0],
                        "brut": f"Scanner global date: {best_date}"
                    }
                    print(f"    [date] -> '{best_date}' (Fallback Global)")"""

# Find the matched string and replace it manually
match = re.search(pattern_sec_feu, content)
if match:
    content = content[:match.start()] + replacement_sec_feu + content[match.end():]
else:
    print("Pattern not found!")

with open('c:\\Users\\Topo_4\\Documents\\AT_PFE\\Anti\\yolo\\plan_classifier.py', 'w', encoding='utf-8') as f:
    f.write(content)
