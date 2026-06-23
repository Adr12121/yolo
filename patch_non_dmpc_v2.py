"""
patch_non_dmpc_v2.py — Applique les patches restants sur plan_classifier.py (v2)
Patches à appliquer :
  P3 : Améliorer _refine_type_plan_from_ocr (corps)
  P4 : Ajouter _extract_with_vlm et _extract_with_vlm_full_page avant process_plan
  P5 : Mettre à jour process_plan pour passer type_plan aux VLM
"""
import re, os

TARGET = os.path.join(os.path.dirname(__file__), "plan_classifier.py")

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

print(f"[Patch v2] Fichier chargé : {src.count(chr(10))} lignes")

# ─────────────────────────────────────────────────────────────────
# PATCH 3 : améliorer _refine_type_plan_from_ocr
# ─────────────────────────────────────────────────────────────────
OLD3 = '''    # 1. Signaux textuels forts explicites
    if re.search(r'proc[eè]s\\s*[-]?\\s*verbal', full_text_all):
        return "PVa"
    if re.search(r'document\\s*modificatif|d\\.m\\.p\\.c|d[\\\'\'\\s]?arp[eoa]nt[ao]g[eo]', full_text_all):
        return "DMPC"
    if "lotissement" in full_text_all or "division" in full_text_all:
        return "PLa"
    if "croquis" in full_text_all or "conservation" in full_text_all:
        return "CROQUIS"  # Les anciens croquis sont traités via la grille spatiale globale'''

NEW3 = '''    # 1. Signaux textuels forts (ordre décroissant de spécificité)
    if re.search(r'proc[eè]s\\s*[-]?\\s*verbal', full_text_all):
        return "PVa"
    if re.search(r'bornage\\s+contradictoire|reconnaissance\\s+de\\s+limites|accord\\s+amiable\\s+de\\s+bornage', full_text_all):
        return "PVa"
    if re.search(r'document\\s*modificatif|d\\.m\\.p\\.c|d[\\\'\'\\s]?arp[eoa]nt[ao]g[eo]', full_text_all):
        return "DMPC"
    if "lotissement" in full_text_all or "division parcellaire" in full_text_all:
        return "PLa"
    if "croquis" in full_text_all or "conservation" in full_text_all:
        return "CROQUIS"

    # 1b. Heuristique cumulative PVa texte libre (sans mot-clé fort unique)
    _pva_s = sum([
        1 if re.search(r'soussign[eé].*g[eé]om[eè]tre', full_text_all) else 0,
        1 if re.search(r'\\bbornage\\b|\\breconnaissance\\b|limites?\\s+de\\s+propri', full_text_all) else 0,
        1 if re.search(r'fait\\s+[aà]\\s+[A-Z][a-z]', full_text_all) else 0,
        1 if re.search(r'certifi[eé]\\s+exact|vu\\s+et\\s+approuv[eé]', full_text_all) else 0,
    ])
    if _pva_s >= 2:
        print(f"  [Classif] Signaux PVa libres ({_pva_s}/4) -> PVa")
        return "PVa"'''

if OLD3 in src:
    src = src.replace(OLD3, NEW3, 1)
    print("[Patch 3] OK : signaux PVa ajoutés dans _refine_type_plan_from_ocr")
else:
    print("[Patch 3] SKIP : bloc signaux introuvable")

# Remplacer aussi le seuil fixe 80 par une densité relative
OLD3b = '''    # Si le texte est très peu dense (typique des vieux croquis ou manuscrits scannés),
    # on force le routage vers CROQUIS pour bénéficier du VLM spatial global.
    if initial_type == "GENERIC" and len(high_conf_blocks) < 80:
        return "CROQUIS"
        
    # Si le texte est très dense et que les labels sont remplis, c'est moderne (générique ou plan)
    if filled_labels >= 2 and len(high_conf_blocks) >= 80:
        return "GENERIC"
        
    return initial_type'''

NEW3b = '''    # Densité relative (remplace seuil fixe 80 blocs)
    n_high = len(high_conf_blocks)
    n_all = max(len([b for b in ocr_results if b[2] > 0.2]), 1)
    ratio_hc = n_high / n_all

    if initial_type in ("GENERIC", "CROQUIS"):
        if filled_labels >= 2 and n_high >= 60:
            return "GENERIC"
        if n_high < 40 or (n_high < 70 and ratio_hc < 0.55):
            print(f"  [Classif] Document peu dense ({n_high} blocs HC, ratio={ratio_hc:.2f}) -> CROQUIS")
            return "CROQUIS"

    if filled_labels >= 2 and n_high >= 80:
        return "GENERIC"

    return initial_type'''

if OLD3b in src:
    src = src.replace(OLD3b, NEW3b, 1)
    print("[Patch 3b] OK : seuil densité relatif")
else:
    print("[Patch 3b] SKIP : bloc densité introuvable")

# ─────────────────────────────────────────────────────────────────
# PATCH 4 : Insérer _extract_with_vlm et _extract_with_vlm_full_page
#           juste avant "# ── Pipeline principal"
# ─────────────────────────────────────────────────────────────────
VLM_FUNCTIONS = '''
def _extract_with_vlm(img_bgr, type_plan: str, validate_fn, crops_data=None):
    """Extraction VLM ciblée sur des crops d'ancrage avec modèle adaptatif."""
    import base64, json, subprocess, os, re
    import cv2
    res_dict = {}
    if not crops_data:
        return res_dict

    model_name = _get_vlm_model(type_plan)
    print(f"  [VLM] Crops sur {len(crops_data)} champs (modele: {model_name})...")
    h_img, w_img = img_bgr.shape[:2]

    for field, crop_info in crops_data.items():
        z = crop_info["zone"]
        x0, y0 = int(z[0]*w_img), int(z[1]*h_img)
        x1, y1 = int(z[2]*w_img), int(z[3]*h_img)
        crop_img = img_bgr[y0:y1, x0:x1]
        if crop_img.size == 0:
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buffer = cv2.imencode('.jpg', crop_img, encode_param)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        prompt = _get_vlm_prompt(type_plan, field)
        timeout_sec = 90 if model_name == "llama3.2-vision" else 60

        payload = {
            "model": model_name, "prompt": prompt,
            "images": [img_base64], "stream": False,
            "options": {"temperature": 0.0, "num_predict": 80, "seed": 42}
        }
        os.makedirs("outputs", exist_ok=True)
        payload_path = os.path.join(os.getcwd(), "outputs", f"vlm_payload_{field}.json")
        with open(payload_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp)
        cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate",
               "-H", "Content-Type: application/json", "-d", f"@{payload_path}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if res.returncode == 0 and res.stdout.strip():
                ollama_json = json.loads(res.stdout)
                raw = _reparer_encodage(ollama_json.get("response", "").strip())
                if raw and raw.lower() not in ["none", "null", "", "inconnu", "vide", "illisible"]:
                    if any(kw in raw.lower() for kw in ["désolé", "sorry", "cannot", "i cannot"]):
                        raw = ""
                    if raw:
                        raw = _clean_vlm_response(raw)
                    if raw:
                        final_val = validate_fn(field, raw) if validate_fn else raw
                        if final_val:
                            res_dict[field] = {
                                "valeur": final_val,
                                "zone": z,
                                "brut": crop_info.get("brut", "") + " -> " + raw,
                                "methode": f"vlm_crop_{model_name.split(':')[0]}",
                                "confidence": 0.92 if model_name == "llama3.2-vision" else 0.90,
                            }
                            print(f"    [VLM/{model_name.split(':')[0]}] {field} -> '{final_val}'")
        except Exception as e:
            print(f"    [VLM Crop] Erreur {field}: {e}")
        finally:
            if os.path.exists(payload_path):
                os.remove(payload_path)
    return res_dict


def _extract_with_vlm_full_page(img_bgr, fields_to_extract, commune_db=None, type_plan: str = "GENERIC"):
    """Extraction VLM pleine page avec modèle et prompt adaptatifs."""
    import base64, json, subprocess, os, re
    import cv2
    res_dict = {}
    model_name = _get_vlm_model(type_plan)
    print(f"  [VLM Pleine Page] Champs: {fields_to_extract} (modele: {model_name})")

    h, w = img_bgr.shape[:2]
    max_dim = 2000 if model_name == "llama3.2-vision" else 1500
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_r = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
    else:
        img_r = img_bgr

    _, buffer = cv2.imencode('.jpg', img_r, [int(cv2.IMWRITE_JPEG_QUALITY), 87])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    _fdesc = {
        "commune": "nom de la commune", "section": "lettre de section cadastrale",
        "feuille": "numéro de feuille", "n_ordre": "numéro d'ordre ou DA",
        "date": "date de dressé (JJ/MM/AAAA)", "echelle": "échelle (1/XXXX)",
        "geometre": "nom du géomètre-expert", "indication": "objet du document",
        "proprietaires_anciens": "propriétaire cédant",
        "proprietaires_nouveaux": "nouveau propriétaire",
    }
    _ctx = {
        "PVa": "C'est un procès-verbal de bornage tapé à la machine.",
        "CROQUIS": "C'est un ancien plan cadastral (texte manuscrit/tampon).",
        "DMPC": "C'est un formulaire DMPC avec cartouche.",
    }.get(type_plan, "C'est un document cadastral.")

    fields_desc = ", ".join(_fdesc.get(f, f) for f in fields_to_extract)
    prompt = (
        f"Tu es un expert géomètre. {_ctx} "
        f"Extrais: {fields_desc}. "
        f"Ne devine pas — si illisible, mets 'vide'. "
        f"Réponds UNIQUEMENT en JSON avec les clés: {', '.join(fields_to_extract)}. "
        f"Valeurs courtes et factuelles."
    )
    timeout_sec = 180 if model_name == "llama3.2-vision" else 120
    payload = {
        "model": model_name, "prompt": prompt, "images": [img_base64],
        "stream": False, "format": "json",
        "options": {"temperature": 0.0, "num_predict": 512, "seed": 42}
    }
    os.makedirs("outputs", exist_ok=True)
    payload_path = os.path.join(os.getcwd(), "outputs", "vlm_payload_full_page.json")
    with open(payload_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate",
           "-H", "Content-Type: application/json", "-d", f"@{payload_path}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if res.returncode == 0 and res.stdout.strip():
            ollama_json = json.loads(res.stdout)
            raw = ollama_json.get("response", "").strip()
            try:
                data = json.loads(raw)
                for f, val in data.items():
                    if f not in fields_to_extract or not isinstance(val, str):
                        continue
                    if val.lower() in ["none", "null", "", "inconnu", "vide", "illisible"]:
                        continue
                    val = _reparer_encodage(val)
                    if any(kw in val.lower() for kw in ["désolé", "peux pas", "illisible", "cannot"]):
                        continue
                    val = _clean_vlm_response(val)
                    if not val:
                        continue
                    final_val = _validate_field(f, val, commune_db)
                    if final_val:
                        conf = 0.78 if model_name == "llama3.2-vision" else 0.70
                        res_dict[f] = {
                            "valeur": final_val,
                            "zone": [0.0, 0.0, 1.0, 1.0],
                            "brut": f"VLM FullPage ({model_name.split(':')[0]}): {val}",
                            "methode": f"vlm_fullpage_{model_name.split(':')[0]}",
                            "confidence": conf,
                        }
                        print(f"    [VLM FP/{model_name.split(':')[0]}] {f} -> '{final_val}'")
            except json.JSONDecodeError as e:
                print(f"    [VLM FullPage JSON Error] {e} - Raw: {raw[:200]}")
    except Exception as e:
        print(f"    [VLM FullPage Error] ({model_name}): {e}")
    finally:
        if os.path.exists(payload_path):
            os.remove(payload_path)
    return res_dict

'''

ANCHOR_PIPELINE = '# ── Pipeline principal'
if ANCHOR_PIPELINE in src and '_extract_with_vlm_full_page' not in src:
    src = src.replace(ANCHOR_PIPELINE, VLM_FUNCTIONS + ANCHOR_PIPELINE, 1)
    print("[Patch 4] OK : fonctions VLM ajoutées")
elif '_extract_with_vlm_full_page' in src:
    print("[Patch 4] SKIP : fonctions VLM déjà présentes")
else:
    print("[Patch 4] SKIP : ancre pipeline introuvable")

# ─────────────────────────────────────────────────────────────────
# PATCH 5 : Dans process_plan, appeler _extract_with_vlm et
#           passer type_plan à _extract_with_vlm_full_page
# ─────────────────────────────────────────────────────────────────
OLD5 = 'vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db)'
NEW5 = 'vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db, type_plan=type_plan)'
if OLD5 in src:
    src = src.replace(OLD5, NEW5, 1)
    print("[Patch 5] OK : type_plan passé à _extract_with_vlm_full_page")
else:
    print("[Patch 5] SKIP : appel introuvable")

# ─────────────────────────────────────────────────────────────────
# ÉCRITURE
# ─────────────────────────────────────────────────────────────────
with open(TARGET, "w", encoding="utf-8") as f:
    f.write(src)
print(f"[Patch v2] Fichier écrit : {src.count(chr(10))} lignes")
print("[Patch v2] Terminé.")
