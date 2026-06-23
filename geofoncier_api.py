import os
import json
import re
import requests
import datetime
import unicodedata
from pathlib import Path

# Configuration de l'environnement (Sandbox par défaut)
GEOFONCIER_ENV = os.environ.get('GEOFONCIER_ENV', 'sandbox').lower()

if GEOFONCIER_ENV == 'production':
    BASE_URL = "https://api2.geofoncier.fr/api"
else:
    BASE_URL = "https://api2.preprod.geofoncier.fr/api"

# Identifiants du cabinet (chargés depuis .env)
GEOFONCIER_API_KEY    = os.environ.get('GEOFONCIER_API_KEY', '')
ENR_CAB_DETENTEUR     = os.environ.get('GEOFONCIER_CAB_DETENTEUR', '1992C100001')  # GEO-SIAPP
ENR_GE_CREATEUR       = os.environ.get('GEOFONCIER_GE_CREATEUR',   '05141')        # Lionnel Robert

# Mapping géomètre -> identifiant cabinet créateur
CABINETS_CREATEURS = {
    "HARROIS": "1987I004169",
    "BARRIAL": "1977I003499",
    # Ajouter les autres géomètres ici au fur et à mesure
    # "SERRET":  "XXXXXXXXXXX",
    # "DUPUY":   "XXXXXXXXXXX",
}

def get_headers():
    """Génère les headers d'authentification pour l'API Géofoncier."""
    if not GEOFONCIER_API_KEY:
        print("⚠️  ATTENTION: GEOFONCIER_API_KEY non définie. L'appel risque d'échouer avec une erreur 401.")
    return {
        "Authorization": GEOFONCIER_API_KEY,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _nc(t: str) -> str:
    """Normalise une chaîne pour comparaison (accents, casse, ponctuation)."""
    nfkd = unicodedata.normalize("NFKD", str(t))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[-''`]", " ", s).upper()
    return re.sub(r"[^A-Z0-9 ]", " ", s).strip()


_commune_db_cache = None

def _load_commune_db() -> list:
    """Charge la base de communes depuis les fichiers JSON locaux (avec cache)."""
    global _commune_db_cache
    if _commune_db_cache is not None:
        return _commune_db_cache
    db = []
    base_dir = os.path.dirname(__file__)
    for fname in ["ardeche.json", "communes_france.json"]:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            try:
                for e in json.load(open(fpath, encoding="utf-8")):
                    n = e.get("nom", "").strip()
                    if n:
                        db.append({"officiel": n, "code": e.get("code", "")})
            except Exception:
                pass
    _commune_db_cache = db
    return db


def get_insee_from_commune(commune_name: str) -> tuple[str, str]:
    """
    Retourne (nom_officiel, code_insee) pour un nom de commune donné.
    Utilise RapidFuzz si disponible, sinon correspondance exacte normalisée.
    Retourne ("", "") si non trouvé.
    """
    if not commune_name:
        return "", ""

    db = _load_commune_db()
    if not db:
        return commune_name, ""

    commune_norm = _nc(commune_name)

    # Recherche par code INSEE direct (si commune_name est déjà un code)
    if re.match(r'^\d{5}$', commune_name.strip()):
        for e in db:
            if str(e.get("code", "")) == commune_name.strip():
                return e["officiel"], e["code"]

    try:
        from rapidfuzz import process as rfp, fuzz
        noms = [_nc(e["officiel"]) for e in db]
        matches = rfp.extract(commune_norm, noms, scorer=fuzz.token_set_ratio, score_cutoff=80)
        if matches:
            best = min(matches, key=lambda m: abs(len(m[0]) - len(commune_norm)))
            e = db[best[2]]
            return e["officiel"], e.get("code", "")
    except ImportError:
        # Fallback : correspondance exacte normalisée
        for e in db:
            if _nc(e["officiel"]) == commune_norm:
                return e["officiel"], e.get("code", "")

    return commune_name, ""


def format_date_iso(date_str: str, annee_full: int | None = None) -> str:
    """
    Convertit une date extraite en format YYYY-MM-DD attendu par l'API Géofoncier.

    Si la date extraite est juste une année (int), retourne YYYY-01-01.
    Si la date est vide mais annee_full est fournie, retourne YYYY-01-01.
    Sinon tente de parser les formats JJ/MM/AAAA, JJ.MM.AAAA, etc.
    """
    if annee_full and (not date_str or str(date_str).strip() in ("", "nan", "None")):
        return f"{annee_full:04d}-01-01"

    text = str(date_str).strip()

    # Format ISO déjà correct
    if re.match(r'^\d{4}-\d{2}-\d{2}$', text):
        return text

    # JJ/MM/AAAA ou JJ.MM.AAAA ou JJ-MM-AAAA
    m = re.match(r'^(\d{1,2})[./-](\d{1,2})[./-](\d{4})$', text)
    if m:
        j, mo, a = m.groups()
        return f"{int(a):04d}-{int(mo):02d}-{int(j):02d}"

    # JJ/MM/AA (2 chiffres) → on suppose 19xx si > 7, 20xx sinon
    m2 = re.match(r'^(\d{1,2})[./-](\d{1,2})[./-](\d{2})$', text)
    if m2:
        j, mo, a2 = m2.groups()
        a_int = int(a2)
        annee = 2000 + a_int if a_int <= 7 else 1900 + a_int
        return f"{annee:04d}-{int(mo):02d}-{int(j):02d}"

    if annee_full:
        return f"{annee_full:04d}-01-01"

    return datetime.datetime.now().strftime("%Y-%m-%d")


def format_cadastre(commune_nom: str, section: str, parcelle_str, code_insee: str = "") -> list:
    """
    Formate les données cadastrales pour le champ 'cadastre' de l'API Géofoncier.
    """
    cad_list = []
    if not section or parcelle_str is None:
        return cad_list

    section = str(section).strip().upper()
    # Gestion de multiples parcelles (liste ou chaîne séparée par virgules)
    if isinstance(parcelle_str, list):
        parcelles = [str(p).strip() for p in parcelle_str if str(p).strip()]
    else:
        parcelles = [p.strip() for p in str(parcelle_str).replace('&', ',').split(',') if p.strip()]

    for p in parcelles:
        # N'extraire que les chiffres
        digits = ''.join(filter(str.isdigit, p))
        if digits:
            try:
                p_num = int(digits)
                cad_list.append({
                    "cad_prefixe": "000",
                    "cad_section": section,
                    "cad_parcelle": p_num,
                })
            except ValueError:
                pass
    return cad_list

def verify_parcel_ign(code_insee, section, numero):
    """
    Vérifie l'existence d'une parcelle via l'API publique de l'IGN.
    Retourne True si trouvée, False sinon.
    """
    if not code_insee or not section or not numero:
        return False
        
    # Formatage de la section (souvent sur 2 lettres/chiffres ex: '0A' ou 'AB')
    section_formatted = str(section).zfill(2) if section.isdigit() else str(section).upper()
    # Formatage du numéro de parcelle (souvent sur 4 chiffres ex: '0014')
    numero_formatted = str(numero).zfill(4)
    
    url = f"https://apicarto.ign.fr/api/cadastre/parcelle?code_insee={code_insee}&section={section_formatted}&numero={numero_formatted}"
    
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and "features" in data and len(data["features"]) > 0:
                # La parcelle existe
                print(f"✅ Contrôle IGN réussi : La parcelle {section_formatted} {numero_formatted} existe sur la commune {code_insee}.")
                return True
    except Exception as e:
        print(f"⚠️ Erreur lors de l'appel à l'API IGN: {e}")
        
    print(f"⚠️ Contrôle IGN : Parcelle {section_formatted} {numero_formatted} non trouvée (Code INSEE: {code_insee}).")
    return False

def get_parcel_geometry(code_insee: str, section: str, numero) -> dict | None:
    """
    Récupère la géométrie GeoJSON d'une parcelle cadastrale via l'API IGN apicarto.

    Retourne un dict :
        {
            "centroid":  [lat, lon],       # Centroïde pour centrer la carte
            "geojson":   {...},            # Feature GeoJSON complet pour Folium
            "found":     True/False
        }
    Retourne None en cas d'erreur réseau.
    """
    if not code_insee or not section or not numero:
        return {"centroid": None, "geojson": None, "found": False}

    section_fmt = str(section).strip().upper().zfill(2)
    numero_fmt  = str(numero).zfill(4)

    url = (
        f"https://apicarto.ign.fr/api/cadastre/parcelle"
        f"?code_insee={code_insee}&section={section_fmt}&numero={numero_fmt}"
    )

    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return {"centroid": None, "geojson": None, "found": False}

        data = resp.json()
        features = data.get("features", [])
        if not features:
            return {"centroid": None, "geojson": None, "found": False}

        feature = features[0]
        geom = feature.get("geometry", {})
        coords = geom.get("coordinates", [])

        # Calcul du centroïde (moyenne des points du polygone extérieur)
        centroid = None
        if geom.get("type") == "Polygon" and coords:
            ring = coords[0]
            # Coordonnées IGN sont en [lon, lat]
            lons = [pt[0] for pt in ring]
            lats = [pt[1] for pt in ring]
            centroid = [sum(lats) / len(lats), sum(lons) / len(lons)]
        elif geom.get("type") == "MultiPolygon" and coords:
            all_pts = [pt for poly in coords for ring in poly for pt in ring]
            lons = [pt[0] for pt in all_pts]
            lats = [pt[1] for pt in all_pts]
            centroid = [sum(lats) / len(lats), sum(lons) / len(lons)]

        return {"centroid": centroid, "geojson": feature, "found": True}

    except Exception as e:
        print(f"[IGN] Erreur get_parcel_geometry : {e}")
        return None



def get_parcel_by_coordinates(lat: float, lon: float) -> dict | None:
    """
    Interroge l'API IGN pour trouver la parcelle située sous un point donné.
    Retourne {"section": str, "numero": str, "code_insee": str} ou None.
    """
    try:
        url = f'https://apicarto.ign.fr/api/cadastre/parcelle?geom={{"type":"Point","coordinates":[{lon},{lat}]}}'
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and data.get("features") and len(data["features"]) > 0:
                props = data["features"][0]["properties"]
                return {
                    "section": props.get("section", ""),
                    "numero": props.get("numero", ""),
                    "code_insee": props.get("code_insee", "")
                }
    except Exception as e:
        print(f"[IGN] Erreur get_parcel_by_coordinates : {e}")
    return None

def geocode_commune(commune_name: str, code_insee: str = "") -> list | None:
    """
    Géocode une commune via l'API Nominatim (OpenStreetMap) ou l'API Géo.
    Retourne [lat, lon] ou None.
    """
    # Priorité : API geo.api.gouv.fr (plus précise pour les communes françaises)
    if code_insee:
        try:
            url = f"https://geo.api.gouv.fr/communes/{code_insee}?fields=centre&format=json"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                coords = data.get("centre", {}).get("coordinates", [])
                if coords and len(coords) == 2:
                    return [coords[1], coords[0]]  # [lat, lon]
        except Exception:
            pass

    # Fallback : Nominatim
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={commune_name},+France&format=json&limit=1"
        resp = requests.get(url, timeout=5, headers={"User-Agent": "GeofonciErPipeline/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return [float(data[0]["lat"]), float(data[0]["lon"])]
    except Exception:
        pass

    # Ardèche : fallback ultime (Privas)
    return [44.7356, 4.5990]


def create_geofoncier_dossier(metadata, dry_run: bool = False) -> dict:

    """
    Crée une pastille (dossier) sur Géofoncier.

    Paramètres :
      metadata  : dict de métadonnées structuré OU chemin vers un fichier JSON.
                  Clés attendues dans le dict :
                    - geometre        : nom du géomètre (ex: "HARROIS")
                    - ref_dossier     : référence (ex: "97050")
                    - commune         : nom de commune
                    - code_insee      : code INSEE (5 caractères) - optionnel, calculé sinon
                    - section         : section cadastrale (ex: "AL")
                    - parcelles       : liste d'ints ou string des numéros de parcelle
                    - date_dossier    : date brute ou annee_full (int)
                    - annee_full      : année 4 chiffres (ex: 1997) - optionnel
                    - op_codes_gf     : liste de codes op. Géofoncier (ex: ["Da"])
      dry_run   : si True, affiche le payload sans l'envoyer.

    Retour :
      {"success": bool, "id_dossier": str, "payload": dict, ...}
    """
    # ── Chargement des métadonnées ─────────────────────────────────────────
    if isinstance(metadata, (str, Path)):
        try:
            with open(metadata, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            return {"success": False, "error": f"Fichier non trouvé: {metadata}"}
    elif isinstance(metadata, dict):
        data = metadata
    else:
        return {"success": False, "error": "Type de métadonnées non reconnu."}

    # ── Extraction des champs ────────────────────────────────────────────────
    geometre      = str(data.get("geometre", "")).strip().upper()
    ref_dossier   = str(data.get("ref_dossier", "")).strip()[:15].replace(" ", "_")
    commune       = data.get("commune", "")
    section       = data.get("section", "")
    parcelles     = data.get("parcelles", [])
    annee_full    = data.get("annee_full", None)
    date_raw      = data.get("date_dossier", data.get("Date", ""))
    op_codes_gf   = data.get("op_codes_gf", [])

    if not ref_dossier:
        return {"success": False, "error": "Clé 'ref_dossier' manquante ou vide."}

    # ── Cabinet créateur selon géomètre ────────────────────────────────────
    enr_cab_createur = CABINETS_CREATEURS.get(geometre, ENR_CAB_DETENTEUR)
    if geometre and geometre not in CABINETS_CREATEURS:
        print(f"⚠️ Géomètre '{geometre}' non répertorié dans CABINETS_CREATEURS. Utilisation du cabinet détenteur par défaut.")

    # ── Code INSEE ─────────────────────────────────────────────────────────
    code_insee = str(data.get("code_insee", "")).strip()
    if not code_insee and commune:
        _, code_insee = get_insee_from_commune(commune)
    if not code_insee:
        print(f"⚠️ Code INSEE introuvable pour la commune '{commune}'. Le versement risque d'échouer.")

    # ── Date dossier ────────────────────────────────────────────────────────
    date_iso = format_date_iso(date_raw, annee_full)

    # ── Bloc cadastre ────────────────────────────────────────────────────────
    if isinstance(parcelles, list) and all(isinstance(p, int) for p in parcelles):
        cadastre_data = [
            {"cad_prefixe": "000", "cad_section": section.strip().upper(), "cad_parcelle": p}
            for p in parcelles if p
        ]
    else:
        cadastre_data = format_cadastre(commune, section, parcelles)

    # ── Payload ──────────────────────────────────────────────────────────────
    payload = {
        "enr_cab_createur":  enr_cab_createur,
        "enr_ref_dossier":   ref_dossier,
        "enr_cab_detenteur": ENR_CAB_DETENTEUR,
        "enr_ge_createur":   ENR_GE_CREATEUR,
        "enr_code_insee":    code_insee,
        "enr_date_dossier":  date_iso,
        "op_code":           op_codes_gf if op_codes_gf else [],
        "enr_visible":       False,
        "enr_statut":        data.get("enr_statut", "Achevé"),
        "enr_memo":          f"Archive importée automatiquement. Commune: {commune}",
    }
    if cadastre_data:
        payload["cadastre"] = cadastre_data

    # ── Appel API ──────────────────────────────────────────────────────────
    url = f"{BASE_URL}/dossiersoge/dossiers/"
    print(f"Preparation création dossier Géofoncier [{GEOFONCIER_ENV.upper()}]")
    print(f"  URL     : {url}")
    print(f"  Payload : {json.dumps(payload, indent=2, ensure_ascii=False)}")

    if dry_run:
        print("⚠️ [MODE DRY RUN] - Aucune donnée envoyée.")
        return {"success": True, "id_dossier": "DRY_RUN_ID", "payload": payload, "dry_run": True}

    response = requests.post(url, headers=get_headers(), json=payload)
    if response.status_code == 201:
        resp_data = response.json()
        id_dossier = resp_data.get("id") or resp_data.get("id_dossier", "")
        print(f"Succes ! Dossier créé avec l'ID : {id_dossier}")
        return {"success": True, "id_dossier": id_dossier, "payload": payload}
    else:
        print(f"Erreur creation dossier ({response.status_code}): {response.text}")
        return {"success": False, "error_code": response.status_code, "error_msg": response.text}


def upload_document_to_dossier(id_dossier, file_path, doc_description="Archive PDF", dry_run=False):
    """
    Rattache un document (PDF/TIF) à un dossier Géofoncier existant.
    """
    url = f"{BASE_URL}/dossiersoge/documents/"
    
    if not os.path.exists(file_path):
        return {"success": False, "error_msg": "Fichier physique introuvable"}
        
    print(f"📤 Upload du document {os.path.basename(file_path)} vers le dossier {id_dossier}...")
    
    # Construction du multipart/form-data
    data = {
        "dossier": id_dossier,
        "doc_visible": "true",
        "doc_code": "01", # Code document, "01" est souvent Croquis/DA, à adapter selon nomenclature Géofoncier
        "doc_description": doc_description
    }
    
    if dry_run:
        print("⚠️ [MODE DRY RUN] - Aucun document réellement uploadé.")
        return {"success": True, "dry_run": True}
        
    with open(file_path, 'rb') as f:
        files = {
            "file": (os.path.basename(file_path), f, "application/pdf") # ou image/tiff selon
        }
        
        response = requests.post(url, headers=get_headers(), data=data, files=files)
        
        if response.status_code == 200:
            print("✅ Document uploadé avec succès !")
            return {"success": True}
        else:
            print(f"❌ Erreur lors de l'upload ({response.status_code}):")
            print(response.text)
            return {"success": False, "error_code": response.status_code, "error_msg": response.text}

if __name__ == "__main__":
    # Test d'affichage de la config
    print("--- Module API Géofoncier Initialisé ---")
    print(f"Environnement cible : {GEOFONCIER_ENV}")
    print(f"Base URL : {BASE_URL}")
    print("Prêt pour la création de pastilles.")
