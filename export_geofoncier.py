"""
export_geofoncier.py
====================
Service d'export continu vers Géofoncier.

- Tourne en permanence (boucle infinie).
- Scanne le dossier 'outputs/' à intervalles réguliers (SCAN_INTERVAL_SEC).
- Traite chaque fichier *_plan_moderne.json non encore publié.
- Déplace les fichiers traités dans 'outputs/archives_publiees/' pour ne pas
  les retraiter à la prochaine itération.
- Rafraîchit le token Géofoncier automatiquement via geofoncier_api.get_valid_token()
  (aucune intervention manuelle requise si GEOFONCIER_LOGIN / GEOFONCIER_PASSWORD
  sont définis dans le .env).

Usage :
    python export_geofoncier.py

Arrêt : Ctrl+C
"""

import os
import re
import glob
import json
import csv
import time
import shutil
import datetime

from geofoncier_api import (
    create_geofoncier_dossier,
    upload_document_to_dossier,
    get_valid_token,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Mode Simulation : True = aucun envoi réel (test uniquement)
DRY_RUN = False

# Dossiers de travail
OUTPUTS_DIR  = "outputs"
INPUTS_DIR   = "inputs"
ARCHIVES_DIR = os.path.join(OUTPUTS_DIR, "archives_publiees")

# Intervalle entre deux scans (en secondes)
SCAN_INTERVAL_SEC = 30

# Extensions de documents source acceptées
SOURCE_EXTENSIONS = ['.pdf', '.jpg', '.png', '.tif', '.tiff']

# Fichier de suivi des publications (dans le dossier racine du projet)
SUIVI_DIR  = "suivi_geofoncier"
SUIVI_JSON = os.path.join(SUIVI_DIR, "suivi_publications.json")
SUIVI_CSV  = os.path.join(SUIVI_DIR, "suivi_publications.csv")

# Colonnes du CSV de suivi
SUIVI_COLONNES = [
    "date_publication",
    "commune",
    "code_insee",
    "geometre",
    "ref_dossier",
    "type_plan",
    "id_geofoncier",
    "document_source",
    "statut",
    "fichier_json",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    """Cree les dossiers necessaires s'ils n'existent pas."""
    os.makedirs(OUTPUTS_DIR,  exist_ok=True)
    os.makedirs(ARCHIVES_DIR, exist_ok=True)
    os.makedirs(INPUTS_DIR,   exist_ok=True)
    os.makedirs(SUIVI_DIR,    exist_ok=True)


def _find_source_doc(base_name: str) -> str | None:
    """Cherche le fichier source (PDF/image) associé à un nom de base."""
    for ext in SOURCE_EXTENSIONS:
        path = os.path.join(INPUTS_DIR, f"{base_name}{ext}")
        if os.path.exists(path):
            return path
    return None


def _archive_file(path: str):
    """Déplace un fichier dans le dossier d'archives."""
    dest = os.path.join(ARCHIVES_DIR, os.path.basename(path))
    # Si un fichier du même nom existe déjà dans les archives, on le suffixe
    if os.path.exists(dest):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(os.path.basename(path))
        dest = os.path.join(ARCHIVES_DIR, f"{name}_{ts}{ext}")
    shutil.move(path, dest)
    print(f"   📦 Archivé → {dest}")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _register_publication(entry: dict):
    """
    Enregistre une publication dans le registre de suivi.

    Mise a jour de deux fichiers :
      - suivi_geofoncier/suivi_publications.json  : historique complet (JSON)
      - suivi_geofoncier/suivi_publications.csv   : trie par commune (Excel-friendly)

    'entry' doit contenir les cles definies dans SUIVI_COLONNES.
    """
    # ── 1. Mise a jour du JSON ─────────────────────────────────────────────
    historique = []
    if os.path.exists(SUIVI_JSON):
        try:
            with open(SUIVI_JSON, encoding="utf-8") as f:
                historique = json.load(f)
        except Exception:
            historique = []

    historique.append(entry)

    with open(SUIVI_JSON, "w", encoding="utf-8") as f:
        json.dump(historique, f, ensure_ascii=False, indent=2)

    # ── 2. Regen du CSV trie par commune puis date ─────────────────────────
    trie = sorted(
        historique,
        key=lambda r: (
            r.get("commune", "").upper(),
            r.get("date_publication", ""),
        )
    )
    with open(SUIVI_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUIVI_COLONNES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trie)

    print(f"   [Suivi] Registre mis a jour : {len(historique)} publication(s) enregistree(s).")


# ─────────────────────────────────────────────────────────────────────────────
# Traitement d'un seul fichier JSON
# ─────────────────────────────────────────────────────────────────────────────

def process_json(json_path: str) -> bool:
    """
    Traite un fichier JSON : crée le dossier sur Géofoncier, uploade le document.
    Retourne True si le traitement est terminé (succès ou échec définitif),
    False si on doit réessayer ultérieurement.
    """
    fname = os.path.basename(json_path)
    print(f"\n[{_now()}] --- Traitement : {fname} ---")

    # Lire le JSON pour extraire les metadonnees de suivi
    try:
        with open(json_path, encoding="utf-8") as _f:
            _meta = json.load(_f)
    except Exception:
        _meta = {}

    # Extraire les champs utiles pour le registre
    _commune   = _meta.get("commune", _meta.get("champs", {}).get("commune", {}).get("valeur", ""))
    _code_insee= _meta.get("code_insee", _meta.get("enr_code_insee", ""))
    _geometre  = _meta.get("geometre", _meta.get("champs", {}).get("geometre", {}).get("valeur", ""))
    _ref       = _meta.get("ref_dossier", _meta.get("n_dossier", ""))
    _type_plan = re.search(r'_plan_([^.]+)\.json$', fname)
    _type_plan = _type_plan.group(1) if _type_plan else "inconnu"

    # 1. Creation de la pastille / dossier
    result_dossier = create_geofoncier_dossier(json_path)

    if not result_dossier.get("success"):
        err = result_dossier.get("error_msg", result_dossier.get("error", "inconnu"))
        print(f"   Echec creation dossier : {err}")
        # Enregistrer l'echec dans le suivi
        _register_publication({
            "date_publication": _now(),
            "commune":          str(_commune),
            "code_insee":       str(_code_insee),
            "geometre":         str(_geometre),
            "ref_dossier":      str(_ref),
            "type_plan":        _type_plan,
            "id_geofoncier":    "",
            "document_source":  "",
            "statut":           f"ECHEC - {str(err)[:80]}",
            "fichier_json":     fname,
        })
        return True

    id_dossier = result_dossier.get("id_dossier", "")
    print(f"   Dossier cree (ID : {id_dossier})")

    # 2. Chercher le document source associe
    base_name = re.sub(r'_plan_[^.]+\.json$', '', fname)
    pdf_path  = _find_source_doc(base_name)

    upload_ok = False
    if pdf_path:
        upload_result = upload_document_to_dossier(
            id_dossier, pdf_path
        )
        if upload_result.get("success"):
            print(f"   Document uploade : {os.path.basename(pdf_path)}")
            upload_ok = True
        else:
            print(f"   Dossier OK, mais ECHEC upload document : "
                  f"{upload_result.get('error_msg', '')[:200]}")
    else:
        print(f"   Aucun document source trouve dans '{INPUTS_DIR}/' pour '{base_name}.*'")

    # 3. Enregistrement dans le registre de suivi
    _register_publication({
        "date_publication": _now(),
        "commune":          str(_commune),
        "code_insee":       str(_code_insee),
        "geometre":         str(_geometre),
        "ref_dossier":      str(_ref),
        "type_plan":        _type_plan,
        "id_geofoncier":    str(id_dossier),
        "document_source":  os.path.basename(pdf_path) if pdf_path else "",
        "statut":           "OK" if upload_ok else "OK (sans document)",
        "fichier_json":     fname,
    })

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────────────────────

def run_loop():
    _ensure_dirs()
    iteration = 0

    print("=" * 60)
    print(" Export Géofoncier — Service continu")
    print(f" Mode           : {'DRY RUN (simulation)' if DRY_RUN else 'PRODUCTION'}")
    print(f" Scan toutes les: {SCAN_INTERVAL_SEC}s")
    print(f" Dossier source : {os.path.abspath(OUTPUTS_DIR)}")
    print(f" Archives       : {os.path.abspath(ARCHIVES_DIR)}")
    print(" Arrêt          : Ctrl+C")
    print("=" * 60)

    # Pré-chargement / vérification du token au démarrage
    try:
        tok = get_valid_token()
        if tok:
            print(f"\n[Token] Token actif chargé au démarrage.")
        else:
            print("\n⚠️  Aucun token disponible. Ajoutez GEOFONCIER_LOGIN et GEOFONCIER_PASSWORD dans .env")
    except Exception as e:
        print(f"\n⚠️  Erreur token au démarrage : {e}")

    while True:
        iteration += 1
        print(f"\n[{_now()}] 🔍 Scan #{iteration} en cours...")

        # Tous les patterns JSON produits par main.py :
        #   *_plan_moderne.json  (fallback modern_plan_extractor)
        #   *_plan_GENERIC.json, *_plan_DMPC.json, *_plan_PVa.json...
        json_files = [
            p for p in glob.glob(os.path.join(OUTPUTS_DIR, "*_plan_*.json"))
            if not os.path.abspath(p).startswith(os.path.abspath(ARCHIVES_DIR))
        ]

        if not json_files:
            print(f"   Aucun fichier à traiter. Prochain scan dans {SCAN_INTERVAL_SEC}s.")
        else:
            print(f"   {len(json_files)} fichier(s) trouvé(s).")
            for json_path in sorted(json_files):
                try:
                    done = process_json(json_path)
                    if done:
                        _archive_file(json_path)
                except Exception as exc:
                    print(f"   💥 Exception inattendue sur {os.path.basename(json_path)} : {exc}")
                    # On archive quand même pour ne pas bloquer la boucle
                    _archive_file(json_path)

        try:
            time.sleep(SCAN_INTERVAL_SEC)
        except KeyboardInterrupt:
            break


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        run_loop()
    except KeyboardInterrupt:
        print(f"\n\n[{_now()}] ⛔ Arrêt du service demandé par l'utilisateur.")
