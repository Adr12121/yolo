import os

TARGET = os.path.join(os.path.dirname(__file__), "plan_classifier.py")

with open(TARGET, "r", encoding="utf-8") as f:
    src = f.read()

HOOK_CODE = '''        # ── FULL PAGE VLM FALLBACK (Phase 4) ──
        all_missing = [f for f in champs_attendus if f not in champs or not champs.get(f, {}).get("valeur")]
        if all_missing and type_plan in ["CROQUIS", "PVa", "GENERIC", "DMPC"]:
            print(f"  [PlanClassifier] Champs manquants {all_missing}. Appel VLM Pleine Page...")
            vlm_full = _extract_with_vlm_full_page(img_bgr, all_missing, commune_db, type_plan=type_plan)
            for k, v in vlm_full.items():
                if k not in champs or not champs.get(k, {}).get("valeur"):
                    champs[k] = v

'''

anchor = '# ── Cycle 1 : Vérification Autonome ──'

if 'FULL PAGE VLM FALLBACK' not in src:
    idx = src.find(anchor)
    if idx != -1:
        src = src[:idx] + HOOK_CODE + "        " + src[idx:]
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
        print("[Patch VLM Full] Injecté avec succès.")
    else:
        print("[Patch VLM Full] Ancre introuvable !")
else:
    print("[Patch VLM Full] Déjà injecté.")
