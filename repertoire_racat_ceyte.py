"""
repertoire_racat_ceyte.py
=========================
Module de résolution de la référence de dossier Racat & Ceyte.

À partir des métadonnées extraites d'un plan (commune, section, parcelles, année),
ce module recherche la ligne correspondante dans le répertoire Excel du cabinet
et retourne la référence de dossier ainsi que les informations de l'acte.

Structure du fichier Excel (colonnes 0-indexed) :
  Col A  (0)  : N° du registre   → identifiant principal du dossier
  Col B  (1)  : N° du cadastre
  Col C  (2)  : Nom du vendeur
  Col D  (3)  : Notaire
  Col E  (4)  : Commune
  Col F  (5)  : Section
  Col G  (6)  : N° parcelle à l'origine
  Col H  (7)  : N° parcelle à l'acquéreur  ← priorité de recherche
  Col I  (8)  : N° parcelle au vendeur
  Col J  (9)  : Nom et adresse de l'acquéreur
  Col K  (10) : Date d'envoi au cadastre   → source de l'année

Avertissement documentaire :
  Le répertoire couvre la période janvier 1959 – février 1964.
  Tout dossier postérieur à cette date ne sera PAS trouvé dans ce répertoire
  et doit être traité manuellement.
"""

import os
import re
import glob
import unicodedata
import datetime
import functools
import pandas as pd


# ── Chemin vers le répertoire Excel ──────────────────────────────────────────
_EXCEL_DIR   = os.path.join(os.path.dirname(__file__), "_RepertoireRACATetCEYTE")
_EXCEL_NAME  = "RepertoireRACATetCEYTE.xlsx"
_EXCEL_SHEET = "Feuil1"

# ── Date limite du répertoire ─────────────────────────────────────────────────
# Le répertoire s'arrête en février 1964.
# Tout plan dont la date dépasse cette limite ne pourra pas être retrouvé.
_DATE_LIMITE = datetime.date(1964, 2, 29)  # fin février 1964

# ── Géomètres utilisant ce répertoire ─────────────────────────────────────────
GEOMETRES_REPERTOIRE_RC = {"RACAT", "CEYTE"}

# ── Seuil de score fuzzy pour la commune ──────────────────────────────────────
_COMMUNE_SCORE_MIN = 75


# ── Mapping codes opération (même table que Harrois/Barrial) ──────────────────
OP_CODE_MAPPING = {
    "DA":  "Da",   # Document d'Arpentage
    "DB":  "Bo",   # Délimitation / Bornage
    "PT":  "Pt",   # Plan Topographique
    "PP":  "Pp",   # Plan de Propriété
    "RF":  "Rf",   # Remembrement Foncier
    "RC":  "Rc",   # Relevé de Consistance
    "EJ":  "Ej",   # Expertise Judiciaire
    "CU":  "Cu",   # Certificat d'Urbanisme
    "EL":  "El",   # Expertise Locative
    "SC":  "Sc",   # Servitude / Cession
    "PI":  "Pi",   # Plan d'Implantation
    "PF":  "Pf",   # Plan Foncier
    "LO":  "Lo",   # Location
    "IM":  "Im",   # Immeuble
    "CO":  "Co",   # Copropriété
}


def normalize_commune(text: str) -> str:
    """
    Normalise un nom de commune pour comparaison :
    - Supprime les accents, passe en majuscules
    - Remplace tirets et apostrophes par espaces
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[-''`]", " ", s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_year_from_date_col(val) -> int | None:
    """
    Extrait l'année depuis la valeur brute de la colonne K (datetime ou string).
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.year
    # Essai texte
    text = str(val).strip()
    m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if m:
        return int(m.group(1))
    return None


def _clean_parcelle_str(raw) -> list[int]:
    """
    Extrait les numéros de parcelles depuis une cellule (peut contenir
    des chaînes multi-valeurs séparées par \n, virgules ou tirets).
    Retourne une liste d'entiers uniques.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return []
    # Séparer sur sauts de ligne, virgules, espaces
    tokens = re.split(r"[\n\r,;/\s]+", s)
    result = []
    seen = set()
    for t in tokens:
        # Garder uniquement les parties numériques (ignore suffixes comme -1182)
        nums = re.findall(r"\d+", t)
        for n in nums:
            try:
                v = int(n)
                if 1 <= v <= 99999 and v not in seen:
                    seen.add(v)
                    result.append(v)
            except ValueError:
                continue
    return result


def clean_parcelles_for_lookup_rc(raw_str: str) -> list[int]:
    """
    Version compatible avec l'interface : nettoie une chaîne OCR brute
    de parcelles pour n'extraire que des entiers valides (1–9999).
    """
    if not raw_str or str(raw_str).strip() in ("", "nan", "None", "[]"):
        return []
    tokens = re.findall(r"\b\d+\b", str(raw_str))
    result = []
    seen = set()
    for t in tokens:
        try:
            v = int(t)
            if 1 <= v <= 9999 and v not in seen:
                seen.add(v)
                result.append(v)
        except ValueError:
            continue
    return result


@functools.lru_cache(maxsize=1)
def load_repertoire_rc_df() -> pd.DataFrame:
    """
    Charge le fichier Excel Racat & Ceyte.
    Résultat mis en cache (chargé une seule fois par session).

    Colonnes du DataFrame résultant :
      n_registre, n_cadastre, vendeur, notaire, commune, section,
      parcelle_origine, parcelle_acquereur, parcelle_vendeur, acquereur, annee
    """
    excel_path = os.path.join(_EXCEL_DIR, _EXCEL_NAME)
    if not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"Répertoire Racat & Ceyte introuvable : {excel_path}"
        )

    df_raw = pd.read_excel(excel_path, sheet_name=_EXCEL_SHEET, header=None)

    # Renommer les colonnes utiles (0-indexed)
    rename_map = {
        0:  "n_registre",
        1:  "n_cadastre",
        2:  "vendeur",
        3:  "notaire",
        4:  "commune",
        5:  "section",
        6:  "parcelle_origine",
        7:  "parcelle_acquereur",
        8:  "parcelle_vendeur",
        9:  "acquereur",
        10: "date_cadastre",
    }
    df = df_raw.rename(columns=rename_map)

    # Garder uniquement les colonnes renommées
    cols_to_keep = [c for c in rename_map.values() if c in df.columns]
    df = df[cols_to_keep].copy()

    # Filtrer : ne garder que les lignes avec un n_registre numérique
    df = df[pd.to_numeric(df["n_registre"], errors="coerce").notna()].copy()
    df["n_registre"] = pd.to_numeric(df["n_registre"], errors="coerce").astype(int)

    # Normaliser la commune et la section
    df["commune"]  = df["commune"].astype(str).str.strip().str.upper()
    df["section"]  = df["section"].astype(str).str.strip().str.upper()

    # Extraire l'année depuis la colonne date
    df["annee"] = df["date_cadastre"].apply(_extract_year_from_date_col)

    # Listes de parcelles (les 3 colonnes) converties en listes d'entiers
    df["parc_orig_list"]  = df["parcelle_origine"].apply(_clean_parcelle_str)
    df["parc_acq_list"]   = df["parcelle_acquereur"].apply(_clean_parcelle_str)
    df["parc_vend_list"]  = df["parcelle_vendeur"].apply(_clean_parcelle_str)

    # Combiné : toutes les parcelles du dossier pour recherche élargie
    def _combine(row):
        s = set()
        s.update(row["parc_acq_list"])   # priorité acquéreur
        s.update(row["parc_orig_list"])
        s.update(row["parc_vend_list"])
        return list(s)
    df["parc_all"] = df.apply(_combine, axis=1)

    return df.reset_index(drop=True)


def check_hors_repertoire(annee_plan: int | None, mois_plan: int | None = None) -> bool:
    """
    Retourne True si la date du plan est postérieure à février 1964
    (= hors couverture du répertoire Racat/Ceyte).

    annee_plan : année à 4 chiffres (ex: 1967, 2003)
    mois_plan  : mois (1-12), optionnel
    """
    if annee_plan is None:
        return False
    if annee_plan > 1964:
        return True
    if annee_plan == 1964:
        # Hors répertoire si après février (mois >= 3, ou mois inconnu)
        if mois_plan is None or mois_plan >= 3:
            return True
    return False


def find_dossier_rc(
    commune: str,
    section: str,
    parcelles_raw: str,
    annee_plan: int | None = None,
    mois_plan: int | None = None,
) -> dict:
    """
    Fonction principale de résolution de dossier pour Racat & Ceyte.

    Paramètres :
      commune      : nom de commune extrait du plan
      section      : section cadastrale OCR (ex: "B")
      parcelles_raw: chaîne brute des parcelles extraites
      annee_plan   : année à 4 chiffres si connue (ex: 1962)
      mois_plan    : mois si connu (ex: 3 pour mars)

    Retour :
    {
      "status"          : "MATCH_UNIQUE" | "CANDIDATS" | "NO_MATCH" | "HORS_REPERTOIRE" | "ERREUR",
      "avertissement"   : str | None,   # note lacune documentaire si applicable
      "ref_dossier"     : "59/488",     # format YY/N_REGISTRE si MATCH_UNIQUE
      "annee"           : 1959,
      "n_registre"      : 488,
      "commune_excel"   : "AUBENAS",
      "section_excel"   : "B",
      "parcelle_acq"    : [1683, 1682],
      "parcelle_orig"   : [1659],
      "parcelle_vend"   : [],
      "vendeur"         : "Ollier Henri...",
      "acquereur"       : "Société...",
      "candidats"       : [...],  # si CANDIDATS
      "parcelles_utilisees": [1683],
    }
    """
    # ── Vérification hors-répertoire en premier ──────────────────────────────
    hors_rep = check_hors_repertoire(annee_plan, mois_plan)
    avertissement = None
    if hors_rep:
        avertissement = (
            "⚠ Le répertoire Racat & Ceyte ne couvre que jusqu'à février 1964. "
            "Ce dossier (postérieur à cette date) ne peut pas être retrouvé dans le répertoire — "
            "une recherche manuelle est nécessaire."
        )
        return {
            "status": "HORS_REPERTOIRE",
            "avertissement": avertissement,
            "message": avertissement,
            "parcelles_utilisees": [],
        }

    # ── Chargement du DataFrame ──────────────────────────────────────────────
    try:
        df = load_repertoire_rc_df()
    except FileNotFoundError as e:
        return {"status": "ERREUR", "message": str(e)}

    # ── Étape 1 : Filtre Commune (fuzzy) ────────────────────────────────────
    commune_norm = normalize_commune(commune)
    if not commune_norm:
        return {"status": "NO_MATCH", "message": "Commune vide ou non renseignée.", "parcelles_utilisees": []}

    try:
        from rapidfuzz import fuzz
        commune_scores = df["commune"].apply(
            lambda c: fuzz.token_set_ratio(commune_norm, normalize_commune(c))
        )
        df_commune = df[commune_scores >= _COMMUNE_SCORE_MIN].copy()
    except ImportError:
        df_commune = df[
            df["commune"].apply(normalize_commune) == commune_norm
        ].copy()

    if df_commune.empty:
        return {
            "status": "NO_MATCH",
            "message": f"Commune '{commune}' introuvable dans le répertoire Racat/Ceyte.",
            "parcelles_utilisees": [],
        }

    # ── Étape 2 : Filtre Parcelle ────────────────────────────────────────────
    # Stratégie : priorité parcelle acquéreur, puis toutes les parcelles
    parcelles_propres = clean_parcelles_for_lookup_rc(parcelles_raw)
    df_travail = pd.DataFrame()

    if parcelles_propres:
        # Priorité 1 : parcelle acquéreur (la plus récente)
        mask_acq = df_commune["parc_acq_list"].apply(
            lambda lst: any(p in lst for p in parcelles_propres)
        )
        df_acq = df_commune[mask_acq].copy()

        if not df_acq.empty:
            df_travail = df_acq
        else:
            # Priorité 2 : toutes les parcelles combinées
            mask_all = df_commune["parc_all"].apply(
                lambda lst: any(p in lst for p in parcelles_propres)
            )
            df_travail = df_commune[mask_all].copy()

    if df_travail.empty:
        df_travail = df_commune.copy()

    # ── Étape 3 : Filtre Année (si connue) ──────────────────────────────────
    if annee_plan is not None and not df_travail.empty:
        df_annee = df_travail[df_travail["annee"] == annee_plan].copy()
        if not df_annee.empty:
            df_travail = df_annee
        # sinon on garde df_travail sans filtre année

    # ── Étape 4 : Arbitrage Section ─────────────────────────────────────────
    if len(df_travail) == 1:
        return _build_match_unique_rc(df_travail.iloc[0], parcelles_propres)

    if len(df_travail) > 1 and section:
        section_norm = str(section).strip().upper()
        try:
            from rapidfuzz import fuzz
            sec_scores = df_travail["section"].apply(
                lambda s: fuzz.ratio(section_norm, str(s).strip().upper())
            )
            best_score = sec_scores.max()
            if best_score >= 60:
                df_sec = df_travail[sec_scores == best_score].copy()
                if len(df_sec) == 1:
                    return _build_match_unique_rc(df_sec.iloc[0], parcelles_propres)
                df_travail = df_sec
        except ImportError:
            df_sec = df_travail[df_travail["section"] == section_norm].copy()
            if len(df_sec) == 1:
                return _build_match_unique_rc(df_sec.iloc[0], parcelles_propres)
            if not df_sec.empty:
                df_travail = df_sec

    # ── Retour CANDIDATS ─────────────────────────────────────────────────────
    if not df_travail.empty:
        candidats = _build_candidats_list_rc(df_travail)
        return {
            "status": "CANDIDATS",
            "avertissement": None,
            "candidats": candidats,
            "parcelles_utilisees": parcelles_propres,
            "message": (
                f"{len(candidats)} candidat(s) trouvé(s). "
                "Sélectionnez le bon dossier dans le tableau."
            ),
        }

    return {
        "status": "NO_MATCH",
        "avertissement": None,
        "message": "Aucune correspondance trouvée dans le répertoire Racat/Ceyte.",
        "parcelles_utilisees": parcelles_propres,
    }


def _build_ref_dossier_rc(row) -> str:
    """
    Construit la référence dossier : AA/N_REGISTRE
    Ex : annee=1959, n_registre=488  →  "59/488"
    """
    annee = row.get("annee")
    n_reg = row.get("n_registre")
    if annee:
        aa = int(annee) % 100
        return f"{aa:02d}/{int(n_reg)}"
    return str(int(n_reg))


def _build_match_unique_rc(row, parcelles_utilisees: list) -> dict:
    """Construit le dict de retour pour un match unique."""
    return {
        "status": "MATCH_UNIQUE",
        "avertissement": None,
        "ref_dossier": _build_ref_dossier_rc(row),
        "annee": int(row["annee"]) if pd.notna(row.get("annee")) else None,
        "n_registre": int(row["n_registre"]),
        "commune_excel": str(row.get("commune", "")),
        "section_excel": str(row.get("section", "")),
        "parcelle_acq": row.get("parc_acq_list", []),
        "parcelle_orig": row.get("parc_orig_list", []),
        "parcelle_vend": row.get("parc_vend_list", []),
        "vendeur": str(row.get("vendeur", "")),
        "acquereur": str(row.get("acquereur", "")),
        "notaire": str(row.get("notaire", "")),
        "parcelles_utilisees": parcelles_utilisees,
    }


def _build_candidats_list_rc(df: pd.DataFrame) -> list[dict]:
    """Construit la liste de candidats pour affichage dans l'interface."""
    result = []
    for _, row in df.iterrows():
        result.append({
            "ref_dossier":   _build_ref_dossier_rc(row),
            "annee":         int(row["annee"]) if pd.notna(row.get("annee")) else None,
            "date_cadastre": str(row.get("date_cadastre", "")),
            "n_registre":    int(row["n_registre"]),
            "commune":       str(row.get("commune", "")),
            "section":       str(row.get("section", "")),
            "parcelle_acq":  row.get("parc_acq_list", []),
            "parcelle_orig": row.get("parc_orig_list", []),
            "vendeur":       str(row.get("vendeur", ""))[:60],
            "acquereur":     str(row.get("acquereur", ""))[:60],
        })
    return result


# ── Auto-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST repertoire_racat_ceyte.py")
    print("=" * 60)

    # Test 1 : cas réel — Aubenas, section B, parcelle acquéreur 1683, année 1959
    print("\n[TEST 1] Aubenas / section B / parcelle 1683 / 1959")
    r = find_dossier_rc("Aubenas", "B", "1683", annee_plan=1959)
    print(f"  Status : {r['status']}")
    if r["status"] == "MATCH_UNIQUE":
        print(f"  Ref    : {r['ref_dossier']}")
        print(f"  Vendeur: {r['vendeur'][:50]}")
        print("  OK — match unique trouve")
    elif r["status"] == "CANDIDATS":
        print(f"  {len(r['candidats'])} candidats")
        for c in r["candidats"][:3]:
            print(f"    - {c['ref_dossier']} | {c['commune']} {c['section']} | acq:{c['parcelle_acq']}")
    else:
        print(f"  Message : {r.get('message','')}")

    # Test 2 : hors répertoire — date 1967
    print("\n[TEST 2] Hors répertoire — année 1967")
    r2 = find_dossier_rc("Aubenas", "B", "1683", annee_plan=1967)
    print(f"  Status : {r2['status']}")
    assert r2["status"] == "HORS_REPERTOIRE"
    print("  OK — HORS_REPERTOIRE détecté")

    # Test 3 : hors répertoire — mars 1964
    print("\n[TEST 3] Hors répertoire — mars 1964")
    r3 = find_dossier_rc("Rocles", "C", "1074", annee_plan=1964, mois_plan=3)
    print(f"  Status : {r3['status']}")
    assert r3["status"] == "HORS_REPERTOIRE"
    print("  OK — mars 1964 hors répertoire")

    # Test 4 : commune inconnue
    print("\n[TEST 4] Commune inconnue")
    r4 = find_dossier_rc("ZORGLUB_SUR_MER", "A", "999")
    assert r4["status"] == "NO_MATCH"
    print(f"  Status : {r4['status']} — OK")

    print("\n" + "=" * 60)
    print("Tous les tests terminés.")
    print("=" * 60)
