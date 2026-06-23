"""
patch_pva.py - Implement _extract_pva_textuel and hook it into process_plan
"""
import re, os

TARGET = os.path.join(os.path.dirname(__file__), "plan_classifier.py")

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

PVA_FUNC = '''
def _extract_pva_textuel(ocr_results, w: int, h: int, validate_fn) -> dict:
    """
    Extraction dédiée pour les Procès-Verbaux de bornage (PVa).
    Analyse séquentielle du texte libre et des mots-clés typiques.
    """
    import re
    res = {}
    full_text = " ".join([b[1] for b in ocr_results])
    
    blocks = [{"text": b[1], "bbox": b[0], "prob": b[2], 
               "cy": sum(p[1] for p in b[0])/4.0, "cx": sum(p[0] for p in b[0])/4.0,
               "y0": min(p[1] for p in b[0]), "y1": max(p[1] for p in b[0]),
               "x0": min(p[0] for p in b[0]), "x1": max(p[0] for p in b[0])} 
              for b in ocr_results]

    # 1. Commune : "commune de [X]" ou "territoire de la commune de [X]"
    m_com = re.search(r'(?i)commune\s+(?:de\s+|d\')([A-Za-z\xc0-\xff][A-Za-z\xc0-\xff\s\-]{2,30}?)(?:\s+section|\s*,|\s*\n|$)', full_text)
    if m_com:
        val = validate_fn("commune", m_com.group(1))
        if val:
            res["commune"] = {"valeur": val, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_com.group(0), "methode": "pva_regex_commune", "confidence": 0.95}

    # 2. Date et Lieu : "Fait à [Lieu], le [Date]"
    # Le lieu Fait à X correspond très souvent à la commune ou à une ville reconnue.
    m_fait = re.search(r'(?i)fait\s+[aà]\s+([A-Za-z\xc0-\xff\s\-]{2,30}?)\s*,\s*le\s+([0-9]{1,2}(?:er)?\s+[a-zéû]+\s+[0-9]{4}|[0-9]{1,2}\s*[/\-\.]\s*[0-9]{1,2}\s*[/\-\.]\s*[0-9]{2,4})', full_text)
    if m_fait:
        lieu = m_fait.group(1).strip()
        date_str = m_fait.group(2).strip()
        if "commune" not in res:
            val_lieu = validate_fn("commune", lieu)
            if val_lieu:
                res["commune"] = {"valeur": val_lieu, "zone": [0.0, 0.0, 1.0, 1.0], "brut": f"Fait à {lieu}", "methode": "pva_regex_fait_a", "confidence": 0.90}
        val_date = validate_fn("date", date_str)
        if val_date:
            res["date"] = {"valeur": val_date, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_fait.group(0), "methode": "pva_regex_date", "confidence": 0.95}

    # Fallback date si pas de "Fait à"
    if "date" not in res:
        m_date = re.search(r'(?i)\ble\s+([0-9]{1,2}(?:er)?\s+[a-zéû]+\s+[0-9]{4}|[0-9]{1,2}\s*[/\-\.]\s*[0-9]{1,2}\s*[/\-\.]\s*[0-9]{2,4})\b', full_text)
        if m_date:
            val = validate_fn("date", m_date.group(1))
            if val:
                res["date"] = {"valeur": val, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_date.group(0), "methode": "pva_regex_date", "confidence": 0.85}

    # 3. Section
    m_sect = re.search(r'(?i)section\s+([A-Z]{1,2})', full_text)
    if m_sect:
        val = validate_fn("section", m_sect.group(1))
        if val:
            res["section"] = {"valeur": val, "zone": [0.0, 0.0, 1.0, 1.0], "brut": m_sect.group(0), "methode": "pva_regex_section", "confidence": 0.90}

    # 4. Géomètre : "Le Géomètre", "Le soussigné", etc.
    cands_geo = []
    for b in blocks:
        if b["cy"] > h * 0.4:  # Scanner la moitié inférieure
            val = validate_fn("geometre", b["text"])
            if val:
                cands_geo.append((b, val))
    if cands_geo:
        # Prendre le dernier (le plus en bas)
        best = cands_geo[-1]
        res["geometre"] = {
            "valeur": best[1], 
            "zone": [best[0]["x0"]/w, best[0]["y0"]/h, best[0]["x1"]/w, best[0]["y1"]/h], 
            "brut": best[0]["text"], 
            "methode": "pva_scan_geometre", 
            "confidence": 0.85
        }

    return res
'''

# Insertion de la fonction avant process_plan si elle n'y est pas
if '_extract_pva_textuel' not in src:
    anchor = 'def process_plan('
    idx = src.find(anchor)
    if idx != -1:
        src = src[:idx] + PVA_FUNC + '\n' + src[idx:]
        print('[Patch PVa] Injected _extract_pva_textuel')

# Ajout de l'appel dans process_plan
call_anchor = 'if _dmpc_res:\n                _graph_page_results.update(_dmpc_res)'
new_call = '''if _dmpc_res:
                _graph_page_results.update(_dmpc_res)

        # 1b. Stratégie Experte PVa (Texte libre tapé à la machine)
        if type_plan == "PVa":
            print(f"  [PlanClassifier] Extraction PVa textuelle prioritaire...")
            _pva_res = _extract_pva_textuel(
                all_ocr_page, w, h, lambda f, v: _validate_field(f, v, commune_db)
            )
            if _pva_res:
                _graph_page_results.update(_pva_res)'''

if 'Stratégie Experte PVa' not in src:
    src = src.replace(call_anchor, new_call)
    print('[Patch PVa] Injected PVa hook into process_plan')

# Universal fallback pour 'Fait à / le' dans process_plan après l'extraction par zone
fallback_anchor = '# ── 3. Merge dans doc_champs ──'
fallback_code = '''# ── 2b. Fallback Universel : "Fait à / le" ──
        # Si la date ou la commune manquent encore, on tente de les récupérer via la signature "Fait à ... le ..."
        if "date" not in champs or "commune" not in champs:
            full_text_page = " ".join([b[1] for b in all_ocr_page])
            m_fait = re.search(r'(?i)fait\\s+[aà]\\s+([A-Za-z\\xc0-\\xff\\s\\-]{2,30}?)\\s*,\\s*le\\s+([0-9]{1,2}(?:er)?\\s+[a-zéû]+\\s+[0-9]{4}|[0-9]{1,2}\\s*[\\/\\-\\.]\\s*[0-9]{1,2}\\s*[\\/\\-\\.]\\s*[0-9]{2,4})', full_text_page)
            if m_fait:
                lieu = m_fait.group(1).strip()
                date_str = m_fait.group(2).strip()
                if "commune" not in champs and "commune" in champs_attendus:
                    val_lieu = _validate_field("commune", lieu, commune_db)
                    if val_lieu:
                        champs["commune"] = {"valeur": val_lieu, "zone": [0,0,1,1], "brut": f"Fait à {lieu}", "methode": "fallback_fait_a", "confidence": 0.85}
                if "date" not in champs and "date" in champs_attendus:
                    val_date = _validate_field("date", date_str, commune_db)
                    if val_date:
                        champs["date"] = {"valeur": val_date, "zone": [0,0,1,1], "brut": m_fait.group(0), "methode": "fallback_fait_date", "confidence": 0.90}

        '''

if 'Fallback Universel' not in src:
    idx = src.find(fallback_anchor)
    if idx != -1:
        src = src[:idx] + fallback_code + '\n        ' + src[idx:]
        print('[Patch PVa] Injected Universal Fallback for "Fait à/le"')

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(src)
print('[Patch PVa] Done.')
