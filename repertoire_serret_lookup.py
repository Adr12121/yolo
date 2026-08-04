# -*- coding: utf-8 -*-
"""
repertoire_serret_lookup.py
===========================
Module de résolution de la référence de dossier Fernand SERRET.

À partir des métadonnées extraites d'un plan (commune, année), ce module
recherche dans le répertoire Excel des archives SERRET et retourne les
dossiers correspondants triés par similarité de l'affaire/donneur d'ordre.

Structure du fichier Excel (colonnes issues de review_serret.py) :
  N° Dossier             : Référence du dossier (ex: "89/123", "123/79")
  Date                   : Date (ex: "30/7/89", "14/07/85")
  Commune                : Nom de la commune en majuscules
  Désignation de l'Affaire : Description de l'opération (texte libre)
  Donneur d'ordre        : Nom du client (texte libre)
  Notes                  : Notes de validation manuelle

Numéro de cabinet Géofoncier : 03622 (Fernand SERRET)

Chemin prioritaire : Z:\\_ArchivesSERRET\\Repertoire_Archives_SERRET.xlsx
Fallback local     : ..\\Extraction_Archives_SERRET\\outputs_livrets\\Repertoire_Archives_SERRET.xlsx
"""

import os
import re
import glob
import unicodedata
import functools
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constante cabinet Géofoncier ─────────────────────────────────────────────
CABINET_ID_SERRET = "03622"

# ── Chemins de recherche de l'Excel ─────────────────────────────────────────
_EXCEL_RESEAU = r"Z:\_ArchivesSERRET\Repertoire_Archives_SERRET.xlsx"
# Fallback local — chemin relatif depuis le dossier du module (yolo/)
_EXCEL_LOCAL_1 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Extraction_Archives_SERRET", "outputs_livrets", "Repertoire_Archives_SERRET.xlsx"
)
# Fallback local — chemin absolu direct
_EXCEL_LOCAL_2 = r"C:\Users\Topo_4\Documents\AT_PFE\Extraction_Archives_SERRET\outputs_livrets\Repertoire_Archives_SERRET.xlsx"

# ── Jeu de géomètres utilisant ce répertoire ────────────────────────────────
GEOMETRES_REPERTOIRE_SERRET = {"SERRET"}

# ── Seuil minimum pour la commune (fuzzy) ────────────────────────────────────
_COMMUNE_SCORE_MIN = 65

# ── Message d'avertissement permanent ────────────────────────────────────────
AVERTISSEMENT_REGISTRES_PARTIELS = (
    "⚠️ Le répertoire de Fernand SERRET est issu de registres manuscrits numérisés par OCR. "
    "Certaines entrées peuvent être incomplètes ou mal reconnues. "
    "En cas de doute, consultez les registres papier originaux."
)


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalise un texte pour la comparaison fuzzy (accents, casse, ponctuation)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[-''`]", " ", s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_year_from_date_serret(date_str: str) -> int | None:
    """
    Extrait l'année depuis une date Serret au format jj/mm/aa ou jj/mm/aaaa.
    Ex: "30/7/89" → 1989, "14/07/85" → 1985, "28/4/79" → 1979
    """
    if not date_str:
        return None
    m = re.search(r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})", str(date_str))
    if m:
        yr = int(m.group(3))
        if yr < 100:
            return 2000 + yr if yr <= 30 else 1900 + yr
        return yr
    # Tentative : juste l'année seule
    m2 = re.search(r"\b(19\d{2}|20[0-2]\d)\b", str(date_str))
    if m2:
        return int(m2.group(1))
    return None


# ── Chargement de l'Excel ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _find_excel_path() -> str | None:
    """Détermine le chemin vers le fichier Excel Serret (réseau en priorité)."""
    # 1. Chemin réseau officiel
    if os.path.exists(_EXCEL_RESEAU):
        return _EXCEL_RESEAU

    # 2. Fallbacks locaux
    for local_path in (_EXCEL_LOCAL_1, _EXCEL_LOCAL_2):
        norm = os.path.normpath(local_path)
        if os.path.exists(norm):
            return norm

    # 3. Recherche générique dans les sous-dossiers
    candidates = glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "**", "Repertoire_Archives_SERRET.xlsx"),
        recursive=True
    )
    for c in candidates:
        if "~$" not in c:
            return c

    return None


def load_serret_df() -> pd.DataFrame:
    """
    Charge le fichier Excel Serret en DataFrame.
    Non mis en cache (le fichier peut être mis à jour entre sessions).
    """
    excel_path = _find_excel_path()
    if not excel_path:
        raise FileNotFoundError(
            f"Répertoire Excel SERRET introuvable.\n"
            f"Chemin réseau cherché : {_EXCEL_RESEAU}\n"
            f"Chemin local cherché  : {os.path.normpath(_EXCEL_LOCAL)}\n"
            "Lancez d'abord extract_serret.py puis review_serret.py pour générer le répertoire."
        )

    df = pd.read_excel(excel_path)
    df.columns = [str(c).strip() for c in df.columns]

    # Mapper les colonnes (insensible aux accents/casse)
    col_map = {}
    for col in df.columns:
        col_norm = normalize_text(col)
        if "N DOSSIER" in col_norm or ("DOSSIER" in col_norm and "N" in col_norm[:3]):
            col_map[col] = "n_dossier"
        elif "DATE" in col_norm:
            col_map[col] = "date"
        elif "COMMUNE" in col_norm:
            col_map[col] = "commune"
        elif "AFFAIRE" in col_norm or "DESIGNATION" in col_norm:
            col_map[col] = "affaire"
        elif "DONNEUR" in col_norm or "CLIENT" in col_norm or "ORDRE" in col_norm:
            col_map[col] = "client"
        elif "NOTE" in col_norm:
            col_map[col] = "notes"

    df = df.rename(columns=col_map)

    # Vérifier colonnes essentielles
    for required in ("n_dossier", "commune"):
        if required not in df.columns:
            raise ValueError(
                f"Colonne '{required}' manquante dans l'Excel Serret "
                f"(colonnes détectées : {list(df.columns)})"
            )

    # Typage et nettoyage
    df["n_dossier"] = df["n_dossier"].astype(str).str.strip()
    df["commune"]   = df["commune"].astype(str).str.strip().str.upper()
    df["commune_norm"] = df["commune"].apply(normalize_text)

    # Extraire l'année depuis la colonne Date
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str.strip()
        df["annee"] = df["date"].apply(_extract_year_from_date_serret)
    else:
        df["date"]  = ""
        df["annee"] = None

    for col in ("affaire", "client", "notes"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    # Filtrer les lignes vides (n_dossier vide ou "nan")
    df = df[df["n_dossier"].str.lower().ne("nan") & df["n_dossier"].ne("")]

    return df.reset_index(drop=True)


# ── Score de similarité affaire/client ────────────────────────────────────────

def _score_affaire_client(affaire_excel: str, client_excel: str,
                           hint_affaire: str = "", hint_client: str = "") -> float:
    """
    Calcule un score de similarité (0-100) entre l'affaire/client du registre
    et un texte hint optionnel (noms de propriétaires vus sur le plan).
    Sans hint, retourne 0 (tri par défaut = ordre dans le registre).
    """
    if not hint_affaire and not hint_client:
        return 0.0
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return 0.0
    texte_excel = normalize_text(f"{affaire_excel} {client_excel}")
    texte_hint  = normalize_text(f"{hint_affaire} {hint_client}")
    if not texte_excel or not texte_hint:
        return 0.0
    return float(fuzz.token_set_ratio(texte_hint, texte_excel))


# ── Fonction principale de recherche ─────────────────────────────────────────

def find_dossier_serret(
    commune: str,
    annee: int | None = None,
    hint_affaire: str = "",
    hint_client: str = "",
) -> dict:
    """
    Recherche les dossiers Serret correspondant à la commune et l'année.

    Paramètres :
      commune       : nom de commune (OCR du plan, ex: "Aubenas")
      annee         : année à 4 chiffres (ex: 1989), ou None
      hint_affaire  : texte de l'objet du plan (optionnel, pour tri)
      hint_client   : nom du donneur d'ordre vu sur le plan (optionnel, pour tri)

    Retour :
    {
      "status"          : "CANDIDATS" | "NO_MATCH" | "ERREUR",
      "candidats"       : [list de dicts triés par score desc],
      "nb_candidats"    : int,
      "commune_cherchee": str,
      "annee_cherchee"  : int | None,
      "avertissement"   : str,
      "source_excel"    : str,
      "message"         : str,
    }

    Structure d'un candidat :
    {
      "ref_dossier" : "89/123",   ← N° Dossier tel que dans le registre
      "n_dossier"   : "89/123",
      "date"        : "30/7/89",
      "annee"       : 1989,
      "commune"     : "AUBENAS",
      "affaire"     : "DIVISION PARCELLE",
      "client"      : "DUPONT Jean",
      "notes"       : "",
      "score_commune": 95,
      "score_affaire": 72,   ← 0 si pas de hint
    }
    """
    try:
        df = load_serret_df()
    except (FileNotFoundError, ValueError) as e:
        return {
            "status": "ERREUR",
            "message": str(e),
            "avertissement": AVERTISSEMENT_REGISTRES_PARTIELS,
            "candidats": [],
        }

    source_excel = _find_excel_path() or "inconnu"

    # ── Étape 1 : Filtre Commune (fuzzy) ─────────────────────────────────────
    commune_norm = normalize_text(commune)
    if not commune_norm:
        return {
            "status": "NO_MATCH",
            "message": "Commune non renseignée.",
            "avertissement": AVERTISSEMENT_REGISTRES_PARTIELS,
            "candidats": [],
            "source_excel": source_excel,
        }

    try:
        from rapidfuzz import fuzz
        scores_commune = df["commune_norm"].apply(
            lambda c: fuzz.token_set_ratio(commune_norm, c)
        )
        df_commune = df[scores_commune >= _COMMUNE_SCORE_MIN].copy()
        df_commune = df_commune.assign(_score_commune=scores_commune[df_commune.index])
    except ImportError:
        df_commune = df[df["commune_norm"] == commune_norm].copy()
        df_commune = df_commune.assign(_score_commune=100)

    if df_commune.empty:
        return {
            "status": "NO_MATCH",
            "message": (
                f"Commune '{commune}' introuvable dans le répertoire Serret. "
                "Vérifiez l'orthographe ou consultez les registres papier."
            ),
            "avertissement": AVERTISSEMENT_REGISTRES_PARTIELS,
            "candidats": [],
            "commune_cherchee": commune,
            "annee_cherchee": annee,
            "source_excel": source_excel,
        }

    # ── Étape 2 : Filtre Année (si fournie) ──────────────────────────────────
    if annee is not None:
        df_annee = df_commune[df_commune["annee"] == annee].copy()
        # Si l'année exacte ne donne rien, on élargit ±2 ans (dates OCR imprécises)
        if df_annee.empty:
            df_annee = df_commune[
                df_commune["annee"].apply(
                    lambda a: a is not None and abs(a - annee) <= 2
                )
            ].copy()
        df_travail = df_annee if not df_annee.empty else df_commune.copy()
    else:
        df_travail = df_commune.copy()

    if df_travail.empty:
        return {
            "status": "NO_MATCH",
            "message": (
                f"Aucun dossier trouvé pour '{commune}'"
                + (f" en {annee}" if annee else "")
                + "."
            ),
            "avertissement": AVERTISSEMENT_REGISTRES_PARTIELS,
            "candidats": [],
            "commune_cherchee": commune,
            "annee_cherchee": annee,
            "source_excel": source_excel,
        }

    # ── Étape 3 : Score affaire/client (tri, pas filtre) ─────────────────────
    candidats = []
    hint_a = str(hint_affaire).strip()
    hint_c = str(hint_client).strip()

    for _, row in df_travail.iterrows():
        score_a = _score_affaire_client(
            row.get("affaire", ""),
            row.get("client", ""),
            hint_a, hint_c
        )
        n_dos = str(row["n_dossier"])
        # Protection NaN pandas : row["annee"] peut être float('nan')
        _a_raw = row.get("annee")
        try:
            _annee_int = int(_a_raw) if (_a_raw is not None and str(_a_raw) != "nan") else None
        except (ValueError, TypeError):
            _annee_int = None
        candidats.append({
            "ref_dossier":   n_dos,
            "n_dossier":     n_dos,
            "date":          str(row.get("date", "")),
            "annee":         _annee_int,
            "commune":       str(row["commune"]),
            "affaire":       str(row.get("affaire", "")),
            "client":        str(row.get("client", "")),
            "notes":         str(row.get("notes", "")),
            "score_commune": int(row.get("_score_commune", 0)),
            "score_affaire": int(score_a),
        })

    # Tri : score_affaire DESC, puis score_commune DESC, puis date ASC
    candidats.sort(key=lambda c: (-c["score_affaire"], -c["score_commune"]))

    nb = len(candidats)
    annee_msg = f" en {annee}" if annee else ""
    has_hint = bool(hint_a or hint_c)

    return {
        "status": "CANDIDATS",
        "candidats": candidats,
        "nb_candidats": nb,
        "commune_cherchee": commune,
        "annee_cherchee": annee,
        "avertissement": AVERTISSEMENT_REGISTRES_PARTIELS,
        "source_excel": source_excel,
        "message": (
            f"{nb} dossier(s) trouvé(s) pour '{commune}'{annee_msg}. "
            + ("Le plus probable est affiché en premier (similarité de l'affaire). " if has_hint else "")
            + "Sélectionnez le bon dossier dans le tableau."
        ),
    }


# ── Auto-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  TEST repertoire_serret_lookup.py")
    print("=" * 65)

    p = _find_excel_path()
    print(f"\n[INFO] Fichier Excel trouvé : {p}")

    if not p:
        print("[ERREUR] Aucun fichier Excel Serret trouvé. Vérifiez les chemins.")
    else:
        try:
            df_test = load_serret_df()
            print(f"[OK] {len(df_test)} dossiers chargés.")
            print(f"     Colonnes : {list(df_test.columns)}")
            print(f"     Communes (10 premières) : {sorted(df_test['commune'].dropna().unique())[:10]}")

            communes = df_test["commune"].dropna().unique()
            if len(communes) > 0:
                test_commune = communes[0]
                print(f"\n[TEST 1] Commune='{test_commune}'")
                res = find_dossier_serret(test_commune)
                print(f"  Status    : {res['status']}")
                print(f"  Candidats : {res.get('nb_candidats', 0)}")
                if res.get("candidats"):
                    c = res["candidats"][0]
                    print(f"  1er : {c['ref_dossier']} | {c['commune']} | {c['date']} | {c['affaire'][:40]}")

            print("\n[TEST 2] Commune inconnue")
            res2 = find_dossier_serret("ZORGLUB_SUR_MER")
            print(f"  Status : {res2['status']} — {res2['message']}")

        except Exception as e:
            print(f"[ERREUR] {e}")

    print("\n" + "=" * 65)
