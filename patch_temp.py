import sys
import os

with open('plan_classifier.py', 'r', encoding='utf-8') as f:
    src = f.read()

VLM_FUNCS = '''
def _extract_with_vlm(img_bgr, type_plan: str, validate_fn, crops_data=None):
    """Extraction VLM ciblée sur des crops d'ancrage avec modèle adaptatif."""
    import base64, json, subprocess, os
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
        _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        prompt = _get_vlm_prompt(type_plan, field)
        timeout_sec = 90 if model_name == "llama3.2-vision" else 60
        payload = {"model": model_name, "prompt": prompt, "images": [img_base64],
                   "stream": False, "options": {"temperature": 0.0, "num_predict": 80, "seed": 42}}
        os.makedirs("outputs", exist_ok=True)
        payload_path = os.path.join(os.getcwd(), "outputs", f"vlm_payload_{field}.json")
        with open(payload_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp)
        cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate",
               "-H", "Content-Type: application/json", "-d", f"@{payload_path}"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if res.returncode == 0 and res.stdout.strip():
                ojson = json.loads(res.stdout)
                raw = _reparer_encodage(ojson.get("response", "").strip())
                if raw and raw.lower() not in ["none", "null", "", "inconnu", "vide", "illisible"]:
                    if any(kw in raw.lower() for kw in ["désolé", "sorry", "cannot", "i cannot"]):
                        raw = ""
                    if raw:
                        raw = _clean_vlm_response(raw)
                    if raw:
                        final_val = validate_fn(field, raw) if validate_fn else raw
                        if final_val:
                            res_dict[field] = {
                                "valeur": final_val, "zone": z,
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
    import base64, json, subprocess, os
    import cv2
    res_dict = {}
    model_name = _get_vlm_model(type_plan)
    print(f"  [VLM Pleine Page] {fields_to_extract} (modele: {model_name})")
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
        "CROQUIS": "C'est un ancien plan cadastral (texte manuscrit ou tampon).",
        "DMPC": "C'est un formulaire DMPC avec cartouche structuré.",
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
    payload = {"model": model_name, "prompt": prompt, "images": [img_base64],
               "stream": False, "format": "json",
               "options": {"temperature": 0.0, "num_predict": 512, "seed": 42}}
    os.makedirs("outputs", exist_ok=True)
    payload_path = os.path.join(os.getcwd(), "outputs", "vlm_payload_full_page.json")
    with open(payload_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate",
           "-H", "Content-Type: application/json", "-d", f"@{payload_path}"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if res.returncode == 0 and res.stdout.strip():
            ojson = json.loads(res.stdout)
            raw = ojson.get("response", "").strip()
            try:
                data = json.loads(raw)
                for f, val in data.items():
                    if f not in fields_to_extract or not isinstance(val, str):
                        continue
                    if val.lower() in ["none", "null", "", "inconnu", "vide", "illisible"]:
                        continue
                    val = _reparer_encodage(val)
                    if any(kw in val.lower() for kw in ["désolé", "illisible", "cannot"]):
                        continue
                    val = _clean_vlm_response(val)
                    if not val:
                        continue
                    final_val = _validate_field(f, val, commune_db)
                    if final_val:
                        conf = 0.78 if model_name == "llama3.2-vision" else 0.70
                        res_dict[f] = {
                            "valeur": final_val, "zone": [0.0, 0.0, 1.0, 1.0],
                            "brut": f"VLM FP ({model_name.split(':')[0]}): {val}",
                            "methode": f"vlm_fullpage_{model_name.split(':')[0]}",
                            "confidence": conf,
                        }
                        print(f"    [VLM FP/{model_name.split(':')[0]}] {f} -> '{final_val}'")
            except json.JSONDecodeError as e:
                print(f"    [VLM FullPage JSON Error] {e}")
    except Exception as e:
        print(f"    [VLM FullPage Error] ({model_name}): {e}")
    finally:
        if os.path.exists(payload_path):
            os.remove(payload_path)
    return res_dict

'''

if '_extract_with_vlm_full_page' not in src:
    anchor = '# ── Pipeline principal'
    idx = src.find(anchor)
    if idx == -1:
        print('Pipeline principal anchor NOT FOUND. Will try alternative.')
        anchor = 'def process_plan('
        idx = src.find(anchor)
    
    if idx != -1:
        src = src[:idx] + VLM_FUNCS + '\n' + src[idx:]
        print('Injected VLM functions.')
    else:
        print('Failed to find injection point for VLM functions.')
else:
    print('VLM functions already present.')

src = src.replace(
    'vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db)',
    'vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db, type_plan=type_plan)'
)
print('Replaced _extract_with_vlm_full_page call if it existed.')

with open('plan_classifier.py', 'w', encoding='utf-8') as f:
    f.write(src)
print('Done writing.')
