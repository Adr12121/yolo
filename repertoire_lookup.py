"""
repertoire_lookup.py
====================
Module de résolution de la référence de dossier Harrois / Barrial.

À partir des métadonnées extraites d'un plan (commune, section, parcelles, année),
ce module recherche la ligne correspondante dans le répertoire Excel des archives
et retourne la référence de dossier (format YY+NNN) ainsi que le type d'opération.

Structure du fichier Excel (colonnes 0-indexed) :
  Col A (0) : Année (float: 77.0, 97.0, 2004.0...)
  Col B (1) : N° séquentiel dans l'année
  Col C (2) : Code département
  Col D (3) : Commune
  Col E (4) : Section
  Col F (5) : Parcelle (ancienne, int ou NaN)
  Col G (6) : Propriétaires anciens
  Col H (7) : Propriétaires nouveaux
  Col I (8) : Code opération (DA, DB, PT, PP, RF...)
"""

import os
import re
import glob
import unicodedata
import functools
import pandas as pd


# ── Chemin vers le répertoire Excel ─────────────────────────────────────────
_EXCEL_GLOB = os.path.join(
    os.path.dirname(__file__),
    "_Répertoire des archives BARRIAL & HARROIS.pdf",
    "Répertoire des archives BARRIAL & HARROIS.xlsx"
)

# ── Mapping codes Excel → codes opération Géofoncier ────────────────────────
# À vérifier via le service B3 (/dossiersoge/codesoperations/) au premier appel réel.
# Ces valeurs sont les plus probables d'après la nomenclature Géofoncier standard.
OP_CODE_MAPPING = {
    "DA":  "Da",   # Document d'Arpentage
    "DB":  "Bo",   # Délimitation / Bornage
    "DB":  "Bo",   # alias
    "PT":  "Pt",   # Plan Topographique
    "PP":  "Pp",   # Plan de Propriété
    "RF":  "Rf",   # Remembrement Foncier
    "RC":  "Rc",   # Relevé de Consistance
    "EJ":  "Ej",   # Expertise Judiciaire
    "CU":  "Cu",   # Certificat d'Urbanisme / Copropriété Urbanisme
    "EL":  "El",   # Expertise Locative
    "SC":  "Sc",   # Servitude / Cession
    "PI":  "Pi",   # Plan d'Implantation
    "PF":  "Pf",   # Plan Foncier
    "LO":  "Lo",   # Location
    "IM":  "Im",   # Immeuble
    "CO":  "Co",   # Copropriété
    "DP":  "Da",   # Document de Parcellisation → DA par défaut
    "DADA": "Da",  # Double DA
    "DL":  "Da",   # alias DA
    "lM":  "Im",   # OCR de IM
    "Dl":  "Da",   # OCR de DA
    "Pl":  "Pt",   # OCR de PT
    "co":  "Co",
    "cu":  "Cu",
    "pp":  "Pp",
    "lM":  "Im",
}

# ── Géomètres utilisant ce répertoire ────────────────────────────────────────
GEOMETRES_REPERTOIRE = {"HARROIS", "BARRIAL"}

# ── Seuil de score fuzzy pour la commune ─────────────────────────────────────
_COMMUNE_SCORE_MIN = 75


@functools.lru_cache(maxsize=1)
def load_repertoire_df() -> pd.DataFrame:
    """
    Charge le fichier Excel du répertoire Harrois/Barrial.
    Résultat mis en cache (chargé une seule fois par session).

    Retourne un DataFrame avec des colonnes nommées :
      annee, n_seq, dept, commune, section, parcelle, prop_anciens, prop_nouveaux, op_code
    """
    # Recherche du fichier (robuste aux variations d'encodage de chemin)
    excel_path = None
    candidates = glob.glob(
        os.path.join(os.path.dirname(__file__), "**", "*.xlsx"),
        recursive=True
    )
    for c in candidates:
        if "~$" not in c and "BARRIAL" in c.upper():
            excel_path = c
            break

    if not excel_path or not os.path.exists(excel_path):
        raise FileNotFoundError(
            f"Répertoire Excel introuvable. Cherché dans : {os.path.dirname(__file__)}"
        )

    df = pd.read_excel(excel_path, header=None)

    # Renommer les colonnes utiles
    df.columns = list(range(df.shape[1]))
    df = df.rename(columns={
        0: "annee",
        1: "n_seq",
        2: "dept",
        3: "commune",
        4: "section",
        5: "parcelle",
        6: "prop_anciens",
        7: "prop_nouveaux",
        8: "op_code",
    })

    # Garder uniquement les lignes de données réelles (année numérique)
    df = df[pd.to_numeric(df["annee"], errors="coerce").notna()].copy()
    df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
    df["n_seq"] = pd.to_numeric(df["n_seq"], errors="coerce")
    df["parcelle"] = pd.to_numeric(df["parcelle"], errors="coerce")
    df["commune"] = df["commune"].astype(str).str.strip().str.upper()
    df["section"] = df["section"].astype(str).str.strip().str.upper()
    df["op_code"] = df["op_code"].astype(str).str.strip()

    # Normaliser l'année : si >= 1900, garder les 2 derniers chiffres pour uniformité
    # Ex: 2004.0 → 4 (mais on garde la valeur originale pour reconstruct la ref)
    df["annee_2ch"] = df["annee"].apply(
        lambda x: int(x) % 100 if pd.notna(x) else None
    )

    return df.reset_index(drop=True)


def normalize_commune(text: str) -> str:
    """
    Normalise un nom de commune pour comparaison :
    - Supprime les accents
    - Passe en majuscules
    - Remplace tirets et apostrophes par espaces
    - Supprime les caractères non alphanumériques
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[-''`]", " ", s).upper()
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_year_from_date(date_str: str) -> int | None:
    """
    Extrait l'année à 2 chiffres depuis une chaîne de date.

    Formats supportés :
      - "22.06.97"   → 97
      - "22/06/97"   → 97
      - "22.06.2009" → 9  (→ 09 dans le répertoire qui va de 77 à 2007)
      - "22.06.1997" → 97
      - "1997-01-01" → 97

    Retourne None si non parsable.
    """
    if not date_str or str(date_str).strip() in ("", "nan", "None"):
        return None

    text = str(date_str).strip()

    # Cas ISO : AAAA-MM-JJ (ex: 1997-01-01) — priorité absolue
    m_iso = re.match(r"^(19\d{2}|20\d{2})-(\d{2})-(\d{2})$", text)
    if m_iso:
        return int(m_iso.group(1)) % 100

    # Cherche un pattern de date avec séparateur JJ/MM/AA ou JJ/MM/AAAA
    m = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", text)
    if m:
        year_part = m.group(3)
        year_int = int(year_part)
        if year_int >= 1900:
            return year_int % 100  # 2009 → 9, 1997 → 97
        return year_int  # déjà 2 chiffres

    # Cherche une année seule à 4 chiffres
    m4 = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if m4:
        return int(m4.group(1)) % 100

    return None


def clean_parcelles_for_lookup(raw_str: str) -> list[int]:
    """
    Filtre les parcelles OCR brutes pour n'extraire que les vrais numéros.

    Garde uniquement les tokens purement numériques entre 1 et 9999.
    Ignore les chaînes mêlant lettres et chiffres (bruit OCR).

    Ex: "['9797', 'isC', '69', 'TerrainA', '287']" → [69, 287]
    """
    if not raw_str or str(raw_str).strip() in ("", "nan", "None", "[]"):
        return []

    # Extraire tous les tokens ressemblant à des chiffres
    tokens = re.findall(r"\b\d+\b", str(raw_str))
    result = []
    for t in tokens:
        try:
            v = int(t)
            if 1 <= v <= 9999:
                result.append(v)
        except ValueError:
            continue

    # Dédupliquer tout en conservant l'ordre
    seen = set()
    unique = []
    for v in result:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def map_op_code(code_excel: str) -> str:
    """
    Traduit un code opération du répertoire Excel vers le code API Géofoncier.
    Retourne une chaîne vide si non reconnu (à saisir manuellement).
    """
    if not code_excel or str(code_excel).strip() in ("", "nan"):
        return ""
    key = str(code_excel).strip()
    return OP_CODE_MAPPING.get(key, OP_CODE_MAPPING.get(key.upper(), ""))


def _annee_to_full(annee_2ch: int) -> int:
    """
    Convertit une année 2 chiffres en année 4 chiffres.
    Logique : <= 7 (i.e. 00-07) → 2000-2007, sinon → 1977-1999.
    (Le répertoire couvre 1977 à 2007.)
    """
    if annee_2ch <= 7:
        return 2000 + annee_2ch
    return 1900 + annee_2ch


def find_dossier(
    commune: str,
    section: str,
    parcelles_raw: str,
    annee_2ch: int | None = None,
) -> dict:
    """
    Fonction principale de résolution de dossier.

    Paramètres :
      commune       : nom de commune extrait du plan (ex: "Saint-Privat")
      section       : section cadastrale OCR (ex: "AC" — peut être erronée)
      parcelles_raw : chaîne brute des parcelles extraites (ex: "['69', 'bruit', ...]")
      annee_2ch     : année à 2 chiffres si connue (ex: 97 pour 1997), sinon None

    Retour :
    {
      "status"       : "MATCH_UNIQUE" | "CANDIDATS" | "NO_MATCH",
      "ref_dossier"  : "97050",      # présent si MATCH_UNIQUE
      "op_code_excel": "DA",         # présent si MATCH_UNIQUE
      "op_code_gf"   : "Da",         # présent si MATCH_UNIQUE
      "annee"        : 97,           # présent si MATCH_UNIQUE
      "annee_full"   : 1997,         # présent si MATCH_UNIQUE
      "n_dossier"    : 50,           # présent si MATCH_UNIQUE
      "commune_excel": "SAINT PRIVAT",
      "section_excel": "AL",
      "parcelle_excel": 69,
      "candidats"    : [             # présent si CANDIDATS, liste de dicts
          {"ref_dossier": ..., "commune": ..., "section": ..., ...},
          ...
      ],
      "parcelles_utilisees": [69],   # numéros propres extraits pour la recherche
    }
    """
    try:
        df = load_repertoire_df()
    except FileNotFoundError as e:
        return {"status": "ERREUR", "message": str(e)}

    # ── Étape 1 : Filtre Commune (fuzzy) ────────────────────────────────────
    commune_norm = normalize_commune(commune)
    if not commune_norm:
        return {"status": "NO_MATCH", "message": "Commune vide ou non renseignée."}

    try:
        from rapidfuzz import fuzz
        commune_scores = df["commune"].apply(
            lambda c: fuzz.token_set_ratio(commune_norm, normalize_commune(c))
        )
        df_commune = df[commune_scores >= _COMMUNE_SCORE_MIN].copy()
    except ImportError:
        # Fallback sans rapidfuzz : correspondance exacte normalisée
        df_commune = df[
            df["commune"].apply(normalize_commune) == commune_norm
        ].copy()

    if df_commune.empty:
        return {
            "status": "NO_MATCH",
            "message": f"Commune '{commune}' introuvable dans le répertoire.",
            "parcelles_utilisees": [],
        }

    # ── Étape 2 : Filtre Parcelle (numérique) ───────────────────────────────
    parcelles_propres = clean_parcelles_for_lookup(parcelles_raw)
    df_parcelle = pd.DataFrame()

    if parcelles_propres:
        df_parcelle = df_commune[
            df_commune["parcelle"].isin(parcelles_propres)
        ].copy()

    # Si pas de résultat avec parcelle → mode candidats sur commune seule
    if df_parcelle.empty:
        df_travail = df_commune.copy()
        fallback_parcelle = True
    else:
        df_travail = df_parcelle.copy()
        fallback_parcelle = False

    # ── Étape 3 : Filtre Année (si connue) ──────────────────────────────────
    if annee_2ch is not None and not df_travail.empty:
        df_annee = df_travail[df_travail["annee_2ch"] == annee_2ch].copy()
        if not df_annee.empty:
            df_travail = df_annee
        # Si filtre année donne 0 résultat → on l'ignore (on garde df_travail actuel)

    # ── Étape 4 : Arbitrage Section (si encore plusieurs) ───────────────────
    if len(df_travail) == 1:
        return _build_match_unique(df_travail.iloc[0], parcelles_propres)

    if len(df_travail) > 1 and section:
        section_norm = str(section).strip().upper()
        try:
            from rapidfuzz import fuzz
            sec_scores = df_travail["section"].apply(
                lambda s: fuzz.ratio(section_norm, str(s).strip().upper())
            )
            best_score = sec_scores.max()
            # Si une section ressemble bien → filtrer sur le meilleur score
            if best_score >= 60:
                df_sec = df_travail[sec_scores == best_score].copy()
                if len(df_sec) == 1:
                    return _build_match_unique(df_sec.iloc[0], parcelles_propres)
                df_travail = df_sec
        except ImportError:
            df_sec = df_travail[df_travail["section"] == section_norm].copy()
            if len(df_sec) == 1:
                return _build_match_unique(df_sec.iloc[0], parcelles_propres)
            if not df_sec.empty:
                df_travail = df_sec

    # ── Retour CANDIDATS ────────────────────────────────────────────────────
    if not df_travail.empty:
        candidats = _build_candidats_list(df_travail)
        return {
            "status": "CANDIDATS",
            "candidats": candidats,
            "parcelles_utilisees": parcelles_propres,
            "message": (
                f"{len(candidats)} candidat(s) trouvé(s). "
                "Sélectionnez le bon dossier dans le tableau."
            ),
        }

    return {
        "status": "NO_MATCH",
        "message": "Aucune correspondance trouvée dans le répertoire.",
        "parcelles_utilisees": parcelles_propres,
    }


def build_ref_dossier_from_row(row) -> str:
    """
    Construit la référence dossier depuis une ligne du DataFrame.
    Format : YYNNN (année 2 chiffres + numéro séquentiel 3 chiffres zero-padded)
    Ex : annee=97, n_seq=50 → "97050"
    """
    annee_2ch = int(row["annee"]) % 100
    n_seq = int(row["n_seq"])
    return f"{annee_2ch:02d}{n_seq:03d}"


def _build_match_unique(row, parcelles_utilisees: list) -> dict:
    """Construit le dict de retour pour un match unique."""
    op_code_excel = str(row.get("op_code", "")).strip()
    op_code_gf = map_op_code(op_code_excel)
    annee_val = int(row["annee"])
    annee_2ch = annee_val % 100
    annee_full = _annee_to_full(annee_2ch)
    ref = build_ref_dossier_from_row(row)

    return {
        "status": "MATCH_UNIQUE",
        "ref_dossier": ref,
        "op_code_excel": op_code_excel,
        "op_code_gf": op_code_gf,
        "annee": annee_2ch,
        "annee_full": annee_full,
        "n_dossier": int(row["n_seq"]),
        "commune_excel": str(row.get("commune", "")),
        "section_excel": str(row.get("section", "")),
        "parcelle_excel": int(row["parcelle"]) if pd.notna(row.get("parcelle")) else None,
        "prop_anciens": str(row.get("prop_anciens", "")),
        "prop_nouveaux": str(row.get("prop_nouveaux", "")),
        "parcelles_utilisees": parcelles_utilisees,
    }


def _build_candidats_list(df: pd.DataFrame) -> list[dict]:
    """Construit la liste de candidats pour affichage dans l'interface."""
    result = []
    for _, row in df.iterrows():
        annee_val = int(row["annee"])
        annee_2ch = annee_val % 100
        annee_full = _annee_to_full(annee_2ch)
        op_excel = str(row.get("op_code", "")).strip()
        result.append({
            "ref_dossier": build_ref_dossier_from_row(row),
            "annee": annee_2ch,
            "annee_full": annee_full,
            "n_seq": int(row["n_seq"]) if pd.notna(row.get("n_seq")) else None,
            "commune": str(row.get("commune", "")),
            "section": str(row.get("section", "")),
            "parcelle": int(row["parcelle"]) if pd.notna(row.get("parcelle")) else None,
            "prop_anciens": str(row.get("prop_anciens", "")),
            "prop_nouveaux": str(row.get("prop_nouveaux", "")),
            "op_code_excel": op_excel,
            "op_code_gf": map_op_code(op_excel),
        })
    return result


# ── Auto-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("TEST repertoire_lookup.py")
    print("=" * 60)

    # Test 1 : cas concret — Saint-Privat, parcelle 69, année 97
    print("\n[TEST 1] Saint-Privat / parcelle 69 / année 97")
    result = find_dossier(
        commune="Saint-Privat",
        section="AC",           # OCR erronée intentionnellement
        parcelles_raw="['69', 'bruit_ocr', 'HARROIS', '1234567']",
        annee_2ch=97,
    )
    print(f"  Status       : {result['status']}")
    if result["status"] == "MATCH_UNIQUE":
        print(f"  Référence    : {result['ref_dossier']}")
        print(f"  Section Excel: {result['section_excel']}  (OCR avait 'AC')")
        print(f"  Parcelle     : {result['parcelle_excel']}")
        print(f"  Op Excel     : {result['op_code_excel']} -> GF: {result['op_code_gf']}")
        print(f"  Parcelles OK : {result['parcelles_utilisees']}")
        assert result["ref_dossier"] == "97050", f"ERREUR : attendu 97050, obtenu {result['ref_dossier']}"
        print("  ✅ ref_dossier == '97050' — VALIDÉ")
    else:
        print(f"  Message : {result.get('message', '')}")
        if result["status"] == "CANDIDATS":
            for c in result["candidats"][:3]:
                print(f"    - {c['ref_dossier']} | {c['commune']} {c['section']} {c['parcelle']} | {c['op_code_excel']}")

    # Test 2 : commune inconnue
    print("\n[TEST 2] Commune inconnue")
    result2 = find_dossier("ZORGLUB_SUR_MER", "A", "12", None)
    print(f"  Status : {result2['status']} — {result2.get('message', '')}")
    assert result2["status"] == "NO_MATCH"
    print("  ✅ NO_MATCH — VALIDÉ")

    # Test 3 : extract_year_from_date
    print("\n[TEST 3] extract_year_from_date")
    tests_date = [
        ("22.06.97", 97),
        ("22.06.2009", 9),
        ("22/06/1997", 97),
        ("1997-01-01", 97),
        ("", None),
    ]
    for inp, expected in tests_date:
        got = extract_year_from_date(inp)
        status = "✅" if got == expected else "❌"
        print(f"  {status} '{inp}' → {got} (attendu: {expected})")

    # Test 4 : clean_parcelles_for_lookup
    print("\n[TEST 4] clean_parcelles_for_lookup")
    raw = "['9797', 'isC', '69', 'TerrainA', '287', 'GEO', '0', '10000']"
    clean = clean_parcelles_for_lookup(raw)
    print(f"  Input  : {raw}")
    print(f"  Output : {clean}")
    assert 69 in clean and 287 in clean, "ERREUR: 69 ou 287 absent"
    # 9797 est valide (<=9999), 0 et 10000 ne le sont pas
    assert 9797 in clean, "ERREUR: 9797 devrait être inclus (<=9999)"
    assert 0 not in clean, "ERREUR: 0 devrait être exclu (<1)"
    assert 10000 not in clean, "ERREUR: 10000 devrait être exclu (>9999)"
    print("  Valeurs retenues :", clean)
    print("  ✅ Parcelles valides extraites correctement")

    # Test 5 : map_op_code
    print("\n[TEST 5] map_op_code")
    for code, expected_gf in [("DA", "Da"), ("DB", "Bo"), ("PT", "Pt"), ("XX", "")]:
        got = map_op_code(code)
        status = "✅" if got == expected_gf else "❌"
        print(f"  {status} '{code}' → '{got}' (attendu: '{expected_gf}')")

    print("\n" + "=" * 60)
    print("Tous les tests terminés.")
    print("=" * 60)
