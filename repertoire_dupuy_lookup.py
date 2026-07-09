# -*- coding: utf-8 -*-
"""
repertoire_dupuy_lookup.py
==========================
Module de résolution de la référence de dossier Roger DUPUY.

À partir des métadonnées extraites d'un plan (commune, année), ce module
recherche dans le répertoire Excel des archives DUPUY et retourne les
dossiers correspondants triés par similarité des noms de propriétaires.

IMPORTANT : Les registres Dupuy ne sont pas tous numérisés. Ce module
affiche systématiquement un avertissement à cet effet.

Structure du fichier Excel (colonnes) :
  Année               : Entier 4 chiffres (1971, 1975...)
  N° Dossier          : Numéro séquentiel du dossier
  Commune             : Nom de la commune
  Anciens Propriétaires  : Texte libre OCR
  Nouveaux Propriétaires : Texte libre OCR
  Notes               : Notes de validation manuelle

Chemin prioritaire : Z:\\_ArchivesDUPUY\\Repertoire_Archives_DUPUY.xlsx
Fallback local     : outputs\\Repertoire_Archives_DUPUY.xlsx (dossier extraction)

Format de la référence dossier : YY + N_Dossier
  Exemple : Année=1970, N°=123 → "70123"
  Exemple : Année=1975, N°=5508 → "755508"
"""

import os
import re
import glob
import unicodedata
import functools
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Chemins de recherche de l'Excel ─────────────────────────────────────────
_EXCEL_RESEAU   = r"Z:\_ArchivesDUPUY\Repertoire_Archives_DUPUY.xlsx"
_EXCEL_LOCAL    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Extraction_Archives_DUPUY", "outputs", "Repertoire_Archives_DUPUY.xlsx"
)

# ── Jeu de géomètres utilisant ce répertoire ────────────────────────────────
GEOMETRES_REPERTOIRE_DUPUY = {"DUPUY"}

# ── Seuil minimum pour la commune (fuzzy) ────────────────────────────────────
_COMMUNE_SCORE_MIN = 70

# ── Message d'avertissement permanent ────────────────────────────────────────
AVERTISSEMENT_REGISTRES_PARTIELS = (
    "⚠️ Les registres de Roger DUPUY ne sont pas tous numérisés dans la base. "
    "Même en l'absence de résultat, le dossier peut exister physiquement dans les archives. "
    "En cas de doute, consultez les registres papier."
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


# ── Chargement de l'Excel ─────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _find_excel_path() -> str | None:
    """Détermine le chemin vers le fichier Excel Dupuy (réseau en priorité)."""
    # 1. Chemin réseau officiel
    if os.path.exists(_EXCEL_RESEAU):
        return _EXCEL_RESEAU

    # 2. Fallback local (répertoire Extraction_Archives_DUPUY/outputs/)
    local_norm = os.path.normpath(_EXCEL_LOCAL)
    if os.path.exists(local_norm):
        return local_norm

    # 3. Recherche générique dans les sous-dossiers du module
    candidates = glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "**", "Repertoire_Archives_DUPUY.xlsx"),
        recursive=True
    )
    for c in candidates:
        if "~$" not in c:
            return c

    return None


def load_dupuy_df() -> pd.DataFrame:
    """
    Charge le fichier Excel Dupuy en DataFrame.
    Non mis en cache (le fichier peut être mis à jour entre sessions).
    """
    excel_path = _find_excel_path()
    if not excel_path:
        raise FileNotFoundError(
            f"Répertoire Excel DUPUY introuvable.\n"
            f"Chemin réseau cherché : {_EXCEL_RESEAU}\n"
            f"Chemin local cherché  : {os.path.normpath(_EXCEL_LOCAL)}\n"
            "Lancez d'abord review_dupuy.py pour valider et exporter les archives."
        )

    df = pd.read_excel(excel_path)

    # Normaliser les noms de colonnes (robustesse aux variations d'encodage)
    df.columns = [str(c).strip() for c in df.columns]

    # Mapper les colonnes attendues (insensible aux accents/casse)
    col_map = {}
    for col in df.columns:
        col_norm = normalize_text(col)
        if "ANNEE" in col_norm or "ANN" in col_norm:
            col_map[col] = "annee"
        elif "DOSSIER" in col_norm or "NUMERO" in col_norm or "N DOSSIER" in col_norm:
            col_map[col] = "n_dossier"
        elif "COMMUNE" in col_norm:
            col_map[col] = "commune"
        elif "ANCIENS" in col_norm and "PROPRI" in col_norm:
            col_map[col] = "prop_anciens"
        elif "NOUVEAUX" in col_norm and "PROPRI" in col_norm:
            col_map[col] = "prop_nouveaux"
        elif "NOTE" in col_norm:
            col_map[col] = "notes"

    df = df.rename(columns=col_map)

    # Vérifier colonnes essentielles
    for required in ("annee", "n_dossier", "commune"):
        if required not in df.columns:
            raise ValueError(
                f"Colonne '{required}' manquante dans l'Excel Dupuy "
                f"(colonnes détectées : {list(df.columns)})"
            )

    # Typage
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    df = df[df["annee"].notna()].copy()
    df["annee"] = df["annee"].astype(int)
    df["n_dossier"] = df["n_dossier"].astype(str).str.strip().str.replace(r"\*$", "", regex=True)
    df["commune"] = df["commune"].astype(str).str.strip().str.upper()
    df["commune_norm"] = df["commune"].apply(normalize_text)

    for col in ("prop_anciens", "prop_nouveaux", "notes"):
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)

    return df.reset_index(drop=True)


# ── Construction de la référence dossier ─────────────────────────────────────

def build_ref_dupuy(annee: int, n_dossier: str) -> str:
    """
    Construit la référence dossier Dupuy : YY + N_Dossier.
    Exemples :
      annee=1970, n_dossier="123"  → "70123"
      annee=1975, n_dossier="5508" → "755508"
    """
    yy = str(annee % 100).zfill(2)
    nd = str(n_dossier).strip().lstrip("0") or "0"
    return f"{yy}{nd}"


# ── Score de similarité propriétaires ────────────────────────────────────────

def _score_proprietaires(anciens_excel: str, nouveaux_excel: str,
                          hint_anciens: str = "", hint_nouveaux: str = "") -> float:
    """
    Calcule un score de similarité (0-100) entre les propriétaires du registre
    et un éventuel texte hint (optionnel).
    Sans hint, retourne 0 (tri par défaut = ordre dans le registre).
    """
    if not hint_anciens and not hint_nouveaux:
        return 0.0

    try:
        from rapidfuzz import fuzz
    except ImportError:
        return 0.0

    texte_excel = normalize_text(f"{anciens_excel} {nouveaux_excel}")
    texte_hint  = normalize_text(f"{hint_anciens} {hint_nouveaux}")

    if not texte_excel or not texte_hint:
        return 0.0

    return float(fuzz.token_set_ratio(texte_hint, texte_excel))


# ── Fonction principale de recherche ─────────────────────────────────────────

def find_dossier_dupuy(
    commune: str,
    annee: int | None = None,
    hint_anciens: str = "",
    hint_nouveaux: str = "",
) -> dict:
    """
    Recherche les dossiers Dupuy correspondant à la commune et l'année.

    Paramètres :
      commune        : nom de commune (OCR du plan, ex: "Saint-Privat")
      annee          : année à 4 chiffres (ex: 1975), ou None
      hint_anciens   : texte des anciens propriétaires (optionnel, pour tri)
      hint_nouveaux  : texte des nouveaux propriétaires (optionnel, pour tri)

    Retour :
    {
      "status"        : "CANDIDATS" | "NO_MATCH" | "ERREUR",
      "candidats"     : [list de dicts triés par score_prop desc],
      "nb_candidats"  : int,
      "commune_cherchee": str,
      "annee_cherchee": int | None,
      "avertissement" : str,   # toujours présent
      "source_excel"  : str,   # chemin du fichier utilisé
      "message"       : str,
    }

    Structure d'un candidat :
    {
      "ref_dossier"   : "755508",
      "annee"         : 1975,
      "n_dossier"     : "5508",
      "commune"       : "AILHON",
      "prop_anciens"  : "...",
      "prop_nouveaux" : "...",
      "notes"         : "...",
      "score_commune" : 95,
      "score_prop"    : 72,   # 0 si pas de hint
    }
    """
    try:
        df = load_dupuy_df()
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
        # Fallback : correspondance exacte normalisée
        df_commune = df[df["commune_norm"] == commune_norm].copy()
        df_commune = df_commune.assign(_score_commune=100)

    if df_commune.empty:
        return {
            "status": "NO_MATCH",
            "message": (
                f"Commune '{commune}' introuvable dans le répertoire Dupuy. "
                "Vérifiez l'orthographe ou consultez les archives papier."
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
        if not df_annee.empty:
            df_travail = df_annee
        else:
            # Année non trouvée → on garde tous les résultats de la commune
            # mais on le signale dans le message
            df_travail = df_commune.copy()
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

    # ── Étape 3 : Score propriétaires (tri, pas filtre) ──────────────────────
    candidats = []
    hint_a = str(hint_anciens).strip()
    hint_n = str(hint_nouveaux).strip()

    for _, row in df_travail.iterrows():
        score_p = _score_proprietaires(
            row.get("prop_anciens", ""),
            row.get("prop_nouveaux", ""),
            hint_a, hint_n
        )
        ref = build_ref_dupuy(int(row["annee"]), str(row["n_dossier"]))
        candidats.append({
            "ref_dossier":   ref,
            "annee":         int(row["annee"]),
            "n_dossier":     str(row["n_dossier"]),
            "commune":       str(row["commune"]),
            "prop_anciens":  str(row.get("prop_anciens", "")),
            "prop_nouveaux": str(row.get("prop_nouveaux", "")),
            "notes":         str(row.get("notes", "")),
            "score_commune": int(row.get("_score_commune", 0)),
            "score_prop":    int(score_p),
        })

    # Tri : score_prop DESC, puis score_commune DESC, puis annee ASC
    candidats.sort(key=lambda c: (-c["score_prop"], -c["score_commune"], c["annee"]))

    nb = len(candidats)
    annee_msg = f" en {annee}" if annee else ""
    has_hint = bool(hint_a or hint_n)

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
            + ("Le plus probable est affiché en premier (similarité des propriétaires). " if has_hint else "")
            + "Sélectionnez le bon dossier dans le tableau."
        ),
    }


# ── Auto-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  TEST repertoire_dupuy_lookup.py")
    print("=" * 65)

    # Test chemin Excel
    p = _find_excel_path()
    print(f"\n[INFO] Fichier Excel trouvé : {p}")

    if not p:
        print("[ERREUR] Aucun fichier Excel Dupuy trouvé. Vérifiez les chemins.")
    else:
        try:
            df_test = load_dupuy_df()
            print(f"[OK] {len(df_test)} dossiers chargés.")
            print(f"     Colonnes : {list(df_test.columns)}")
            print(f"     Années   : {sorted(df_test['annee'].unique())}")
            print(f"     Communes : {sorted(df_test['commune'].unique())[:10]} ...")

            # Test 1 : commune présente
            communes = df_test["commune"].unique()
            if len(communes) > 0:
                test_commune = communes[0]
                test_annee = int(df_test[df_test["commune"] == test_commune]["annee"].iloc[0])
                print(f"\n[TEST 1] Commune='{test_commune}', Année={test_annee}")
                res = find_dossier_dupuy(test_commune, test_annee)
                print(f"  Status     : {res['status']}")
                print(f"  Candidats  : {res['nb_candidats']}")
                if res["candidats"]:
                    c = res["candidats"][0]
                    print(f"  1er résultat : {c['ref_dossier']} | {c['commune']} | Anciens: {c['prop_anciens'][:40]}...")

            # Test 2 : commune inconnue
            print("\n[TEST 2] Commune inconnue")
            res2 = find_dossier_dupuy("ZORGLUB_SUR_MER", 1975)
            print(f"  Status : {res2['status']} — {res2['message']}")

            # Test 3 : build_ref_dupuy
            print("\n[TEST 3] build_ref_dupuy")
            for a, n, expected in [(1970, "123", "70123"), (1975, "5508", "755508"), (2003, "42", "0342")]:
                r = build_ref_dupuy(a, n)
                ok = "✅" if r == expected else "❌"
                print(f"  {ok} {a}/{n} → '{r}' (attendu: '{expected}')")

        except Exception as e:
            print(f"[ERREUR] {e}")

    print("\n" + "=" * 65)
