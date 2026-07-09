import os
import json
import re
import base64
import requests
import datetime
import unicodedata
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration de l'environnement (Sandbox par défaut)
GEOFONCIER_ENV = os.environ.get('GEOFONCIER_ENV', 'sandbox').lower()

if GEOFONCIER_ENV == 'production':
    BASE_URL   = "https://api2.geofoncier.fr/api"
    _TOKEN_URL = "https://api2.geofoncier.fr/token"
else:
    BASE_URL   = "https://api2.preprod.geofoncier.fr/api"
    _TOKEN_URL = "https://api2.preprod.geofoncier.fr/token"

# ── Authentification ──────────────────────────────────────────────────────
# Identifiants du cabinet (chargés depuis .env)
# GEOFONCIER_API_KEY : token de départ issu du .env (fallback si pas de login/mdp)
# GEOFONCIER_LOGIN   : pseudonyme du compte (ex: GHA), PAS l'adresse email
# GEOFONCIER_PASSWORD: mot de passe du compte
_GEOFONCIER_API_KEY = os.environ.get('GEOFONCIER_API_KEY', '')
GEOFONCIER_LOGIN    = os.environ.get('GEOFONCIER_LOGIN', '')
GEOFONCIER_PASSWORD = os.environ.get('GEOFONCIER_PASSWORD', '')
ENR_CAB_DETENTEUR   = os.environ.get('GEOFONCIER_CAB_DETENTEUR', '1992C100001')  # GEO-SIAPP
ENR_GE_CREATEUR     = os.environ.get('GEOFONCIER_GE_CREATEUR',   '05141')        # Lionnel Robert

# Cache interne du token courant
_current_token: str = _GEOFONCIER_API_KEY  # on part du token .env si présent
_token_expiry: datetime.datetime | None = None


def _decode_jwt_expiry(token_str: str) -> datetime.datetime | None:
    """
    Décode la date d'expiration (claim 'exp') depuis un JWT sans vérification de signature.
    Retourne un datetime UTC ou None si échec.
    """
    try:
        raw = token_str.strip()
        if raw.lower().startswith('bearer '):
            raw = raw[7:].strip()
        parts = raw.split('.')
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        payload_b64 += '=' * (4 - len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64)
        payload = json.loads(payload_json)
        exp = payload.get('exp')
        if exp:
            return datetime.datetime.utcfromtimestamp(int(exp))
    except Exception as e:
        print(f"[Token] Impossible de decoder l'expiration JWT : {e}")
    return None


def refresh_geofoncier_token() -> str:
    """
    Obtient un nouveau token Bearer depuis l'API Geofoncier.

    Methode : GET /token avec Authorization: Basic base64(login:password)
    C'est la meme methode que sur https://api2.geofoncier.fr/token.html
    (bouton Authorize → username + password → Execute).

    Le login est le PSEUDONYME du compte (ex: GHA), pas l'adresse email.
    Le token a une duree de vie de 10h (36000 secondes).

    Met a jour le cache interne et retourne 'Bearer <jwt>'.
    Leve une RuntimeError si l'authentification echoue.
    """
    global _current_token, _token_expiry

    if not GEOFONCIER_LOGIN or not GEOFONCIER_PASSWORD:
        raise RuntimeError(
            "[Token] GEOFONCIER_LOGIN et GEOFONCIER_PASSWORD doivent etre definis dans le .env "
            "pour rafraichir automatiquement le token. "
            "IMPORTANT : GEOFONCIER_LOGIN est votre PSEUDONYME (ex: GHA), pas votre email."
        )

    print(f"[Token] Demande d'un nouveau token pour '{GEOFONCIER_LOGIN}' sur {_TOKEN_URL} ...")

    # Encodage Basic Auth : base64(login:password)
    credentials = f"{GEOFONCIER_LOGIN}:{GEOFONCIER_PASSWORD}"
    basic_token = base64.b64encode(credentials.encode('utf-8')).decode('ascii')

    headers = {
        "Authorization": f"Basic {basic_token}",
        "Accept": "*/*",
    }
    try:
        resp = requests.get(_TOKEN_URL, headers=headers, timeout=15)
    except requests.RequestException as e:
        raise RuntimeError(f"[Token] Erreur reseau : {e}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"[Token] Echec authentification ({resp.status_code}) : {resp.text[:300]}"
        )

    data = resp.json()
    access_token = data.get('access_token', '')
    if not access_token:
        raise RuntimeError("[Token] Reponse OK mais 'access_token' absent.")

    _current_token = f"Bearer {access_token}"
    _token_expiry = _decode_jwt_expiry(_current_token)
    expires_in = data.get('expires_in', '?')
    if _token_expiry:
        print(f"[Token] Nouveau token valide jusqu'au {_token_expiry.strftime('%Y-%m-%d %H:%M:%S')} UTC ({expires_in}s)")
    else:
        print(f"[Token] Nouveau token obtenu (expires_in={expires_in}s).")
    return _current_token


def get_valid_token() -> str:
    """
    Retourne le token Bearer courant, en le rafraichissant automatiquement
    si GEOFONCIER_LOGIN/PASSWORD sont definis ET si le token expire dans
    moins de 30 minutes (ou est deja expire).
    """
    global _current_token, _token_expiry

    # Decoder l'expiration du token initial au premier appel
    if _token_expiry is None and _current_token:
        _token_expiry = _decode_jwt_expiry(_current_token)

    if GEOFONCIER_LOGIN and GEOFONCIER_PASSWORD:
        now_utc = datetime.datetime.utcnow()
        margin  = datetime.timedelta(minutes=30)  # 30 min avant expiration (token valable 10h)
        if _token_expiry is None or now_utc + margin >= _token_expiry:
            try:
                return refresh_geofoncier_token()
            except RuntimeError as e:
                print(f"[Token] AVERTISSEMENT - rafraichissement echoue : {e}")
                print("[Token] Poursuite avec l'ancien token.")

    return _current_token

# Mapping géomètre -> identifiant cabinet créateur
# Compléter avec les identifiants réels des cabinets concernés
CABINETS_CREATEURS = {
    "HARROIS": "1987I004169",
    "BARRIAL": "1977I003499",
    "SERRET":  os.environ.get("GEOFONCIER_CAB_SERRET",  ""),  # Cabinet Fernand Serret
    "RACAT":   os.environ.get("GEOFONCIER_CAB_RACAT",   ""),  # Cabinet Racat
    "CEYTE":   os.environ.get("GEOFONCIER_CAB_CEYTE",   ""),  # Cabinet Ceyte
    "DUPUY":   os.environ.get("GEOFONCIER_CAB_DUPUY",   "1965I002702"),  # Cabinet Dupuy Roger
    "LACOUR":  os.environ.get("GEOFONCIER_CAB_LACOUR",  ""),  # Cabinet Lacour Jacques
    "ROBERT":  os.environ.get("GEOFONCIER_CAB_ROBERT",  ""),  # Cabinet Robert Lionnel
}

# Mapping nature d'acte -> doc_code Geofoncier (valeur par defaut = 9DIz = Autre)
# La liste exhaustive est disponible via GET /dossiersoge/codesdocuments/
NATURE_ACTE_TO_DOC_CODE = {
    "DMPC":                  "9DIz",  # A affiner avec le vrai code GF
    "BORNAGE":               "9DIz",
    "DIVISION_PARCELLAIRE":  "9DIz",
    "LOTISSEMENT":           "9DIz",
    "REUNION_PARCELLAIRE":   "9DIz",
    "RECONNAISSANCE_LIMITES":"9DIz",
    "AUTRE":                 "9DIz",
}

_doc_codes_cache: dict | None = None

def get_doc_codes(force_refresh: bool = False) -> dict:
    """
    Récupère la liste des codes documents autorisés via l'API Géofoncier.
    Cache en mémoire pour éviter les appels répétés.
    Retourne {code: libelle} ou {} en cas d'erreur.
    """
    global _doc_codes_cache
    if _doc_codes_cache is not None and not force_refresh:
        return _doc_codes_cache
    try:
        url = f"{BASE_URL}/dossiersoge/codesdocuments/"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # L'API retourne une liste d'objets {code, libelle, ...}
            if isinstance(data, list):
                _doc_codes_cache = {str(d.get("code", d.get("doc_code", ""))): d.get("libelle", d.get("doc_libelle", "")) for d in data if d}
            elif isinstance(data, dict):
                _doc_codes_cache = data
            else:
                _doc_codes_cache = {}
            print(f"[GeoFoncier] {len(_doc_codes_cache)} codes documents chargés depuis l'API.")
            return _doc_codes_cache
        else:
            print(f"[GeoFoncier] Impossible de charger les codes documents ({resp.status_code})")
    except Exception as e:
        print(f"[GeoFoncier] Erreur get_doc_codes : {e}")
    _doc_codes_cache = {}
    return {}


def latlon_to_lambert93(lat: float, lon: float) -> tuple[float, float] | None:
    """
    Convertit des coordonnées WGS84 (lat, lon) en Lambert 93 (Est, Nord)
    requis par le champ 'localisant' de l'API Géofoncier.
    Utilise pyproj si disponible, sinon une approximation analytique.
    Retourne (Est, Nord) ou None si échec.
    """
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        east, north = transformer.transform(lon, lat)
        return round(east, 2), round(north, 2)
    except ImportError:
        pass
    # Approximation analytique (précision ~10m, acceptable pour un localisant)
    import math
    a = 6378137.0; e = 0.0818191908426215
    lambda_c = math.radians(3.0); phi_c = math.radians(46.5)
    n = 0.7256077650
    F = 11754255.426
    e_c = (e * math.sin(phi_c))
    C = F * math.exp(-n * math.log(math.tan(math.pi/4 + phi_c/2) * ((1-e_c)/(1+e_c))**(e/2)))
    phi = math.radians(lat); lam = math.radians(lon)
    e_phi = (e * math.sin(phi))
    R = C * math.exp(-n * math.log(math.tan(math.pi/4 + phi/2) * ((1-e_phi)/(1+e_phi))**(e/2)))
    theta = n * (lam - lambda_c)
    X0, Y0 = 700000.0, 6600000.0
    Est = X0 + R * math.sin(theta)
    Nord = Y0 + F - R * math.cos(theta)
    return round(Est, 2), round(Nord, 2)

def get_headers():
    """Génère les headers d'authentification pour l'API Géofoncier (token auto-rafraîchi)."""
    token = get_valid_token()
    if not token:
        print("⚠️  ATTENTION: Aucun token disponible. L'appel risque d'échouer avec une erreur 401.")
    return {
        "Authorization": token,
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
    raw_list = parcelle_str if isinstance(parcelle_str, list) else [parcelle_str]
    parcelles = []
    for item in raw_list:
        for p in str(item).replace('&', ',').split(','):
            if p.strip():
                parcelles.append(p.strip())

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
    Si plusieurs parcelles sont fournies, vérifie la première.
    Retourne True si trouvée, False sinon.
    """
    if not code_insee or not section or not numero:
        return False
        
    # Formatage de la section (souvent sur 2 lettres/chiffres ex: '0A' ou 'AB')
    section_formatted = str(section).zfill(2) if str(section).isdigit() else str(section).upper()
    
    # Extraction de la première parcelle si liste ou chaîne avec virgules
    if isinstance(numero, list):
        numeros = [str(p).strip() for p in numero if str(p).strip()]
    else:
        numeros = [p.strip() for p in str(numero).replace('&', ',').split(',') if p.strip()]
        
    if not numeros:
        return False
        
    # Extraire uniquement les chiffres du premier numéro
    digits = ''.join(filter(str.isdigit, numeros[0]))
    if not digits:
        return False
        
    # Formatage du numéro de parcelle (souvent sur 4 chiffres ex: '0014')
    numero_formatted = digits.zfill(4)
    
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


def create_geofoncier_dossier(metadata) -> dict:

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
    indication    = str(data.get("indication", "")).upper()
    n_ordre       = str(data.get("n_ordre", "")).strip()

    if not ref_dossier:
        return {"success": False, "error": "Clé 'ref_dossier' manquante ou vide."}

    # ── Cabinet créateur selon géomètre ────────────────────────────────────
    enr_cab_createur = CABINETS_CREATEURS.get(geometre) or ENR_CAB_DETENTEUR
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

    # ── Localisant (coordonnées Lambert 93 de la pastille) ──
    localisants = []
    lat_lon = data.get("lat_lon")  # [lat, lon] ou None
    if lat_lon and len(lat_lon) == 2:
        conv = latlon_to_lambert93(float(lat_lon[0]), float(lat_lon[1]))
        if conv:
            localisants.append({"loc_coord_est": float(conv[0]), "loc_coord_nord": float(conv[1])})
            print(f"  [API] Localisant Lambert93 : Est={conv[0]}, Nord={conv[1]}")

    # Mapping du statut UI vers l'enum API (minuscules avec accents)
    ui_status = data.get("enr_statut", "Achevé")
    status_map = {
        "Achevé": "achevé",
        "Indéterminé": "indéterminé",
        "En cours": "en production",
        "Annulé": "annulé",
        "Archivé": "achevé"
    }
    api_status = status_map.get(ui_status, "achevé")

    # ── Extraction DMPC (Numéro de DA) ───────────────────────────────────
    dmpc_refs = []
    # Le numéro de DA est stocké dans le champ N_Ordre de l'interface
    if n_ordre and n_ordre.lower() not in ["nan", "none", ""]:
        # Nettoyage basique : extraire les chiffres et les lettres collées
        da_match = re.search(r'(\d+)([A-Za-z]?)', n_ordre)
        if da_match:
            digits = str(int(da_match.group(1))) # Retire les zéros de début
            letter = da_match.group(2).upper()
            if not letter:
                letter = "A"
            num_dmpc = digits + letter
            # Par défaut, le préfixe DMPC sur GF est souvent le département
            dept = code_insee[:2] + "0" if code_insee else "000"
            dmpc_refs.append({
                "dmpc_prefixe": dept,
                "dmpc_ref": num_dmpc
            })

    # Le géomètre créateur doit obligatoirement avoir exercé dans le cabinet créateur.
    # L'ID du cabinet (ex: 1987I004169) contient généralement l'ID du GE à la fin (04169).
    ge_createur_deduit = enr_cab_createur[-5:] if enr_cab_createur and len(enr_cab_createur) >= 5 else ENR_GE_CREATEUR

    # ── Payload ─────────────────────────────────────────────────────────
    payload = {
        "enr_cab_createur":  enr_cab_createur,
        "enr_ref_dossier":   ref_dossier,
        "enr_cab_detenteur": ENR_CAB_DETENTEUR,
        "enr_ge_createur":   ge_createur_deduit,
        "enr_code_insee":    code_insee,
        "enr_date_dossier":  date_iso,
        "op_code":           op_codes_gf if op_codes_gf else [],
        "enr_visible":       False,
        "enr_statut":        api_status,
        "enr_memo":          f"Archive importée automatiquement. Commune: {commune}",
    }
    if cadastre_data:
        payload["cadastre"] = cadastre_data
    if localisants:
        payload["localisant"] = localisants
    if dmpc_refs:
        payload["dmpc_ref"] = dmpc_refs

    # ── Appel API ──────────────────────────────────────────────────────────
    url = f"{BASE_URL}/dossiersoge/dossiers/"
    print(f"Preparation création dossier Géofoncier [{GEOFONCIER_ENV.upper()}]")
    print(f"  URL     : {url}")
    print(f"  Payload : {json.dumps(payload, indent=2, ensure_ascii=False)}")

    response = requests.post(url, headers=get_headers(), json=payload)
    
    # ── PATCH ROBUSTE : Fallback si dmpc_ref est rejeté
    if response.status_code == 400 and "dmpc_ref" in response.text and "dmpc_ref" in payload:
        print(f"⚠️ Rejet de dmpc_ref par l'API: {response.text}")
        print("🔄 Nouvelle tentative de versement SANS le dmpc_ref pour garantir la publication...")
        payload.pop("dmpc_ref")
        response = requests.post(url, headers=get_headers(), json=payload)

    if response.status_code == 201:
        resp_data = response.json()
        id_dossier = resp_data.get("id") or resp_data.get("id_dossier", "")
        print(f"Succes ! Dossier créé avec l'ID : {id_dossier}")
        return {"success": True, "id_dossier": id_dossier, "payload": payload}
    else:
        print(f"Erreur creation dossier ({response.status_code}): {response.text}")
        return {"success": False, "error_code": response.status_code, "error_msg": response.text}


def upload_document_to_dossier(
    id_dossier: str,
    file_path: str,
    doc_description: str = "Archive PDF",
    doc_code: str = "",
    nature_acte: str = "AUTRE"
) -> dict:
    """
    Rattache un document (PDF/TIF/JPG) à un dossier Géofoncier existant.
    — doc_code  : code GF explicite (prioritaire). Si vide, déduit depuis nature_acte.
    — nature_acte : clé de NATURE_ACTE_TO_DOC_CODE (ex: 'DMPC', 'BORNAGE').
    """
    if not os.path.exists(file_path):
        return {"success": False, "error_msg": f"Fichier introuvable : {file_path}"}

    # Résolution du doc_code
    if not doc_code:
        doc_code = NATURE_ACTE_TO_DOC_CODE.get(nature_acte, "9DIz")
        # Tentative de récupération depuis l'API si le cache est disponible
        _codes = get_doc_codes()
        if _codes and doc_code not in _codes:
            # Trouver le premier code disponible contenant "arpentage" ou "plan"
            for c, lib in _codes.items():
                if any(kw in lib.lower() for kw in ["arpentage", "plan", "document"]):
                    doc_code = c
                    break

    # Détection automatique du MIME type
    ext = os.path.splitext(file_path)[1].lower()
    _MIME_MAP = {".pdf": "application/pdf", ".tif": "image/tiff", ".tiff": "image/tiff",
                 ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                 ".dxf": "application/dxf", ".dwg": "application/dwg"}
    mime_type = _MIME_MAP.get(ext, "application/octet-stream")

    print(f"\U0001f4e4 Upload '{os.path.basename(file_path)}' (code={doc_code}, mime={mime_type}) -> dossier {id_dossier}")

    form_data = {
        "dossier":         id_dossier,
        "doc_visible":     "true",
        "doc_code":        doc_code,
        "doc_description": doc_description or f"Archive {os.path.basename(file_path)}",
    }

    # L'API attend multipart/form-data — NE PAS inclure Content-Type dans headers (requests le gère)
    headers_upload = {k: v for k, v in get_headers().items() if k.lower() != "content-type"}
    with open(file_path, "rb") as fh:
        files = {"file": (os.path.basename(file_path), fh, mime_type)}
        response = requests.post(f"{BASE_URL}/dossiersoge/documents/",
                                 headers=headers_upload, data=form_data, files=files)
    if response.status_code in (200, 201):
        print("\u2705 Document uploadé avec succès !")
        return {"success": True, "doc_code": doc_code}
    else:
        print(f"\u274c Erreur upload ({response.status_code}): {response.text[:300]}")
        return {"success": False, "error_code": response.status_code, "error_msg": response.text}

if __name__ == "__main__":
    # Test d'affichage de la config
    print("--- Module API Géofoncier Initialisé ---")
    print(f"Environnement cible : {GEOFONCIER_ENV}")
    print(f"Base URL : {BASE_URL}")
    print("Prêt pour la création de pastilles.")
