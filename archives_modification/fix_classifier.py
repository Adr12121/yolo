import re

with open("plan_classifier.py", "r", encoding="utf-8") as f:
    content = f.read()

# Etape 1: Restaurer la validation commune qui a ete ecrasee
bad_commune_block = """                            if commune_db:
                                    break

    # -- Niveau 2 : Scan global par Regex pour n_ordre (utile si l'OCR écrit "n dordro") --
    if "n_ordre" not in res:
        full_text_local = " ".join(b["text"] for b in blocks)
        m_da_regex = re.search(r"(?i)(?:n[\\\"°oO]?\\s*d['’]?o[ri]?dr[eo]|d\\.?a\\.?\\s*n[\\\"°oO]?|dossier\\s*d['’]?arpentage)\\s*[:\\-\\s]?\\s*([0-9a-zA-Z]{2,6})", full_text_local)
        if m_da_regex:
            val_valid = validate_fn("n_ordre", m_da_regex.group(1))
            if val_valid:
                res["n_ordre"] = {
                    "valeur": val_valid, "zone": [0.0, 0.0, 1.0, 1.0],
                    "brut": m_da_regex.group(0), "methode": "dmpc_fulltext_regex", "confidence": 0.85,
                }"""

good_commune_block = """                            if commune_db:
                                matched = _match_commune(val, commune_db)
                                if matched != val or any(c["officiel"].upper() == val.upper() for c in commune_db):
                                    val = matched
                                else:
                                    continue # Rejeter ce candidat qui ne matche pas la DB
                            res["commune"] = {
                                "valeur": val, "zone": [cand["x0"]/w, cand["y0"]/h, cand["x1"]/w, cand["y1"]/h],
                                "brut": f"{commune_label['text']} -> {cand['text']}", "methode": "dmpc_label_fuzzy", "confidence": 0.95,
                            }
                            break"""

content = content.replace(bad_commune_block, good_commune_block)

# Etape 2: Inserer le regex N_Ordre apres le Niveau 1 (break de la boucle for)
target_break = """                                "confidence": 0.98,
                            }
                            break"""

insert_n_ordre = """                                "confidence": 0.98,
                            }
                            break

    # -- Niveau 2 : Scan global par Regex pour n_ordre (utile si l'OCR écrit "n dordro") --
    if "n_ordre" not in res:
        full_text_local = " ".join(b["text"] for b in blocks)
        m_da_regex = re.search(r"(?i)(?:n[\\\"°oO]?\\s*d['’]?o[ri]?dr[eo]|d\\.?a\\.?\\s*n[\\\"°oO]?|dossier\\s*d['’]?arpentage)\\s*[:\\-\\s]?\\s*([0-9a-zA-Z]{2,6})", full_text_local)
        if m_da_regex:
            val_valid = validate_fn("n_ordre", m_da_regex.group(1))
            if val_valid:
                res["n_ordre"] = {
                    "valeur": val_valid, "zone": [0.0, 0.0, 1.0, 1.0],
                    "brut": m_da_regex.group(0), "methode": "dmpc_fulltext_regex", "confidence": 0.85,
                }"""

# Seulement inserer s'il n'est pas deja la
if "# -- Niveau 2 : Scan global par Regex pour n_ordre" not in content:
    parts = content.split("# 3. Geometre : Quadrant bas-droit")
    if len(parts) > 1:
        idx = parts[0].rfind(target_break)
        if idx != -1:
            parts[0] = parts[0][:idx] + insert_n_ordre + parts[0][idx+len(target_break):]
            content = "# 3. Geometre : Quadrant bas-droit".join(parts)

with open("plan_classifier.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fix applied successfully.")
