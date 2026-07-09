"""
bootstrap_kb.py — Pré-remplit knowledge_base.json avec TOUTES les communes
d'ardeche.json, en conservant les entrées manuelles existantes.

Usage : python bootstrap_kb.py
         python bootstrap_kb.py --dry-run   (affiche sans écrire)
"""
import json, os, sys

KB_PATH      = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
ARDECHE_PATH = os.path.join(os.path.dirname(__file__), "ardeche.json")
DRY_RUN      = "--dry-run" in sys.argv

def main():
    # ── Charger la KB existante ──────────────────────────────────────────────
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    communes_kb = kb.setdefault("communes", {})

    # ── Charger ardeche.json ─────────────────────────────────────────────────
    if not os.path.exists(ARDECHE_PATH):
        print(f"[ERREUR] {ARDECHE_PATH} introuvable — abandono.")
        sys.exit(1)

    with open(ARDECHE_PATH, "r", encoding="utf-8") as f:
        ardeche = json.load(f)

    added = 0
    skipped = 0

    for entry in ardeche:
        nom    = entry.get("nom", "").strip()
        code   = entry.get("code", "").strip()
        dept   = code[:2] if len(code) >= 2 else ""

        if not nom:
            continue

        if nom in communes_kb:
            # Entrée manuelle existante : ne pas écraser, juste compléter le code si manquant
            existing = communes_kb[nom]
            if not existing.get("code_insee") and code:
                existing["code_insee"] = code
            skipped += 1
            continue

        # Nouvelle commune → entrée vide (sera enrichie par validation humaine)
        communes_kb[nom] = {
            "code_insee":             code,
            "departement":            dept,
            "sections_connues":       [],   # à enrichir au fil des validations
            "prefixes_dossier":       [],   # idem
            "types_plans_frequents":  [],   # idem
        }
        added += 1

    print(f"[bootstrap_kb] {added} communes ajoutées, {skipped} déjà présentes.")

    if DRY_RUN:
        print("[bootstrap_kb] Mode --dry-run : aucune écriture.")
        return

    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"[bootstrap_kb] knowledge_base.json mis à jour ({len(communes_kb)} communes au total).")


if __name__ == "__main__":
    main()
