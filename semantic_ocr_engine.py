# pyre-ignore-all-errors
# pyright: reportMissingImports=false
"""
semantic_ocr_engine.py
======================
Moteur de compréhension sémantique pour l'OCR cadastral.

PRINCIPE :
  Avant de faire matcher un texte OCR brut, on détermine d'abord QUEL TYPE de
  donnée est attendue dans ce champ (commune, échelle, section, dossier…).
  Chaque type de champ a ses propres règles de validation et de correction.

  Ex : si le label du champ est "COMMUNE", on cherche dans ardeche.json.
       si le label est "ECHELLE", on valide le format 1/XXXX.
       si le label est "GEOMETRE", on accepte un nom propre libre.

INTÉGRATION :
  Appelé depuis process_document_hybrid() et extract_kpis_from_layout()
  comme couche de post-traitement et de validation de cohérence.
"""

import re
import json
import os
import math
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


# ===========================================================================
# 1. REGISTRE DES TYPES DE CHAMPS CADASTRAUX
# ===========================================================================

# Mots-clés qui permettent d'identifier le type d'un champ dans le document
FIELD_KEYWORDS: Dict[str, List[str]] = {
    "commune": [
        "commune", "commue", "c.n.e", "territoire", "ville de",
        "localite", "localité", "lieu dit", "lieudit"
    ],
    "echelle": [
        "echelle", "échelle", "ech.", "ech :", "scale", "1/", "1 /"
    ],
    "section": [
        "section", "sect.", "section cadastrale", "feuille"
    ],
    "parcelle": [
        "parcelle", "parcelles", "n° parcelle", "numéro parcelle", "lot"
    ],
    "geometre": [
        "géomètre", "geometre", "géomètre-expert", "cabinet", "bureau",
        "dessiné par", "dressé par", "établi par", "signé"
    ],
    "dossier": [
        "dossier", "n° dossier", "référence", "affaire", "n° d'ordre",
        "ordre", "inscription", "numéro"
    ],
    "proprietaire": [
        "propriétaire", "proprietaire", "demandeur", "client", "sci",
        "m.", "mme.", "mr.", "mlle"
    ],
    "date": [
        "date", "le ", "fait à", "dressé le", "établi le", "signé le"
    ],
    "departement": [
        "département", "departement", "dept", "dep."
    ],
}

# Pour chaque type de champ, les règles de validation de la valeur lue
FIELD_VALIDATORS: Dict[str, Any] = {
    "echelle": {
        "pattern": r"1\s*/\s*\d{3,6}",
        "corrections": [
            (r"(?i)\bl\b", "1"),   # l → 1
            (r"(?i)\bI\b", "1"),   # I → 1
            (r"[fFtT]/", "1/"),    # f/ → 1/
            (r"/(\d+)O", r"/\g<1>0"),  # 1O → 10
        ],
        "examples": ["1/500", "1/1000", "1/2000", "1/5000"],
    },
    "section": {
        "pattern": r"^[A-Z]{1,2}\d{0,2}$",
        "corrections": [
            (r"0", "O"),
            (r"(?<!\d)1(?!\d)", "I"),
        ],
        "examples": ["A", "B", "AB", "ZD"],
    },
    "date": {
        "pattern": r"\b(19|20)\d{2}\b|\b\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}\b",
        "corrections": [],
        "examples": ["01/01/2010", "15.03.1985", "2024"],
    },
}


# ===========================================================================
# 2. IDENTIFIANT DU TYPE DE CHAMP
# ===========================================================================

def identify_field_type(label_text: str) -> str:
    """
    Détermine le type de champ à partir du texte du label/en-tête.

    Ex:
      "Commune de"  → "commune"
      "Echelle :"   → "echelle"
      "Section B"   → "section"
      "Dossier n°"  → "dossier"

    Retourne "inconnu" si aucun type identifié.
    """
    if not label_text:
        return "inconnu"

    txt = _normalize(label_text)

    # Ordre : du plus spécifique au plus général
    for field_type, keywords in FIELD_KEYWORDS.items():
        for kw in keywords:
            if _normalize(kw) in txt:
                return field_type

    return "inconnu"


def _normalize(text: str) -> str:
    """Normalise un texte pour la comparaison (minuscules, sans accents)."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", ascii_text.lower()).strip()


# ===========================================================================
# 3. VALIDATEUR ET CORRECTEUR SÉMANTIQUE
# ===========================================================================

class SemanticFieldCorrector:
    """
    Corrige et valide une valeur OCR en fonction du type de champ attendu.

    Utilise :
    - Des règles regex (ex: format échelle 1/XXXX)
    - La base de communes JSON pour le type "commune"
    - Des corrections d'ambiguïtés OCR courantes (O/0, l/1, etc.)
    """

    def __init__(self, commune_db: List[Dict[str, Any]], script_dir: str = "."):
        self.commune_db = commune_db
        self.script_dir = script_dir
        # Index normalisé pour la recherche rapide
        self._commune_index: Dict[str, Dict] = {
            e["normalise"]: e for e in commune_db
        }

    def correct(self, raw_text: str, field_type: str) -> Dict[str, Any]:
        """
        Valide et corrige une valeur selon son type de champ.

        Retourne :
            {
                "valeur": str,          # valeur corrigée
                "confiance": float,     # 0.0 à 1.0
                "methode": str,         # description de la correction
                "valide": bool,         # True si le format est valide
            }
        """
        if not raw_text or raw_text.strip() in ("", "nan", "[VIDE]"):
            return {"valeur": "", "confiance": 0.0, "methode": "vide", "valide": False}

        txt = raw_text.strip()

        if field_type == "commune":
            return self._correct_commune(txt)
        elif field_type == "echelle":
            return self._correct_echelle(txt)
        elif field_type == "section":
            return self._correct_section(txt)
        elif field_type == "date":
            return self._correct_date(txt)
        else:
            # Champs libres (géomètre, dossier, propriétaire…)
            return {"valeur": txt, "confiance": 0.7, "methode": "libre", "valide": True}

    # ── Commune ──────────────────────────────────────────────────────────────

    def _correct_commune(self, txt: str) -> Dict[str, Any]:
        """
        Recherche intelligente dans la base de communes.

        Cascade :
        1. Correspondance exacte normalisée
        2. Correspondance préfixe
        3. Fuzzy matching (rapidfuzz)
        4. Smith-Waterman local (depuis main.py)
        """
        try:
            from rapidfuzz import process as rp, fuzz as rf
        except ImportError:
            return {"valeur": txt, "confiance": 0.3, "methode": "no_rapidfuzz", "valide": False}

        if not self.commune_db:
            return {"valeur": txt, "confiance": 0.3, "methode": "no_db", "valide": False}

        norm = _normalize_commune(txt)

        # 1. Correspondance exacte
        if norm in self._commune_index:
            e = self._commune_index[norm]
            return {
                "valeur": e["officiel"],
                "confiance": 1.0,
                "methode": "exact",
                "valide": True,
                "code": e.get("code", ""),
            }

        # 2. Préfixe
        prefix_matches = [e for e in self.commune_db if e["normalise"].startswith(norm) and len(norm) >= 4]
        if len(prefix_matches) == 1:
            e = prefix_matches[0]
            return {
                "valeur": e["officiel"],
                "confiance": 0.90,
                "methode": "prefixe",
                "valide": True,
                "code": e.get("code", ""),
            }

        # 3. Fuzzy + longueur asymétrique
        noms = [e["normalise"] for e in self.commune_db]
        results = rp.extract(norm, noms, scorer=rf.WRatio, limit=5)

        best_score = 0.0
        best_entry = None
        for match_str, score, idx in results:
            # Bonus si le nom OCR est un préfixe du nom officiel
            cand = self.commune_db[idx]
            if cand["normalise"].startswith(norm):
                score = min(100, score + 10)
            # Pénalité longueur : si l'OCR est BEAUCOUP plus court (abréviation tolérée)
            n_ocr = sum(1 for c in norm if c.isalpha())
            n_com = sum(1 for c in cand["normalise"] if c.isalpha())
            if n_ocr > 0 and n_com > n_ocr:
                # Tolérant pour les abréviations
                sigma = 1.3
            else:
                sigma = 0.45
            ratio = (n_com / n_ocr) if n_ocr > 0 else 1.0
            mu = math.log(1.3)
            log_r = math.log(ratio) if ratio > 0 else -10.0
            z = (log_r - mu) / sigma
            len_factor = max(0.2, math.exp(-0.5 * z * z))
            adj_score = score * len_factor

            if adj_score > best_score:
                best_score = adj_score
                best_entry = cand

        if best_entry and best_score >= 45:
            confiance = min(1.0, best_score / 100.0)
            return {
                "valeur": best_entry["officiel"],
                "confiance": confiance,
                "methode": f"fuzzy({best_score:.0f}%)",
                "valide": best_score >= 70,
                "code": best_entry.get("code", ""),
            }

        return {"valeur": txt, "confiance": 0.2, "methode": "echec", "valide": False}

    # ── Échelle ──────────────────────────────────────────────────────────────

    def _correct_echelle(self, txt: str) -> Dict[str, Any]:
        """Valide et corrige le format d'une échelle (1/XXXX)."""
        validator = FIELD_VALIDATORS["echelle"]

        # Appliquer les corrections OCR
        corrected = txt
        for pattern, repl in validator["corrections"]:
            corrected = re.sub(pattern, repl, corrected)

        # Extraire le pattern 1/XXXX
        m = re.search(r"1\s*/\s*(\d+)", corrected)
        if m:
            valeur = f"1/{m.group(1)}"
            return {"valeur": valeur, "confiance": 0.95, "methode": "regex_echelle", "valide": True}

        # Essai avec séparateurs alternatifs
        m2 = re.search(r"1\s*[:\.\-]\s*(\d{3,6})", corrected)
        if m2:
            valeur = f"1/{m2.group(1)}"
            return {"valeur": valeur, "confiance": 0.80, "methode": "regex_echelle_alt", "valide": True}

        return {"valeur": txt, "confiance": 0.3, "methode": "format_inconnu", "valide": False}

    # ── Section cadastrale ────────────────────────────────────────────────────

    def _correct_section(self, txt: str) -> Dict[str, Any]:
        """Valide et corrige une section cadastrale (lettre(s) + chiffre optionnel)."""
        corrected = txt.strip().upper()
        # Remplacer les confusions OCR courantes
        corrected = re.sub(r"\b0\b", "O", corrected)
        corrected = re.sub(r"(?<![A-Z])1(?![0-9])", "I", corrected)

        # Extraire la section
        m = re.search(r"\b([A-Z]{1,2}\d{0,2})\b", corrected)
        if m:
            valeur = m.group(1)
            return {"valeur": valeur, "confiance": 0.90, "methode": "regex_section", "valide": True}

        return {"valeur": txt, "confiance": 0.4, "methode": "format_inconnu", "valide": False}

    # ── Date ─────────────────────────────────────────────────────────────────

    def _correct_date(self, txt: str) -> Dict[str, Any]:
        """Extrait et valide une date."""
        # Chercher d'abord une année
        m_year = re.search(r"\b(19|20)\d{2}\b", txt)
        if m_year:
            return {"valeur": m_year.group(0), "confiance": 0.85, "methode": "annee", "valide": True}

        # Date complète
        m_full = re.search(r"\b(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4})\b", txt)
        if m_full:
            valeur = f"{m_full.group(1)}/{m_full.group(2)}/{m_full.group(3)}"
            return {"valeur": valeur, "confiance": 0.80, "methode": "date_complete", "valide": True}

        return {"valeur": txt, "confiance": 0.3, "methode": "format_inconnu", "valide": False}


def _normalize_commune(text: str) -> str:
    """Normalisation spécifique pour les communes (tirets→espaces, St→Saint, etc.)."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[-''`]", " ", s)
    s = s.upper()
    s = re.sub(r"\bST\b", "SAINT", s)
    s = re.sub(r"\bSTE\b", "SAINTE", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return " ".join(s.split())


# ===========================================================================
# 4. ANALYSEUR DE DOCUMENT COMPLET
# ===========================================================================

class DocumentSemanticAnalyzer:
    """
    Analyse un document complet en comprenant la structure de ses champs.

    Reçoit les détections brutes OCR (liste de {texte, bbox, ...}) et
    produit une extraction structurée en utilisant les règles sémantiques.

    Gère :
    - Documents modernes (plans 2010+) : layout imprimé avec labels explicites
    - Vieux livrets : colonnes manuscrites avec en-têtes fixes
    """

    # Valeurs d'échelles cadastrales valides (pour validation rapide)
    ECHELLES_VALIDES = {
        "1/200", "1/500", "1/1000", "1/1250", "1/2000",
        "1/2500", "1/5000", "1/10000", "1/25000",
    }

    def __init__(self, commune_db: List[Dict[str, Any]], script_dir: str = "."):
        self.corrector = SemanticFieldCorrector(commune_db, script_dir)
        self.commune_db = commune_db

    def analyze(self, raw_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse une page de détections OCR et extrait les KPIs sémantiques.

        raw_detections : liste de {texte, bbox, confiance, type_ocr}

        Retourne :
            {
                "commune": {...},
                "echelle": {...},
                "section": {...},
                "dossier": {...},
                "geometre": {...},
                "proprietaire": {...},
                "date": {...},
            }
        """
        kpis: Dict[str, Any] = {}

        # Trier les détections par position Y (haut → bas)
        sorted_dets = sorted(
            raw_detections,
            key=lambda d: (d.get("bbox", [0, 0, 0, 0])[1] if d.get("bbox") else 0)
        )

        # Construire une liste de (texte, bbox) pour l'analyse
        lines = [(d.get("texte", "").strip(), d.get("bbox", [])) for d in sorted_dets]

        for i, (texte, bbox) in enumerate(lines):
            if not texte:
                continue

            # Identifier si cette ligne EST un label de champ
            field_type = identify_field_type(texte)

            if field_type != "inconnu":
                # Chercher la valeur dans la même ligne (après ":")
                valeur_meme_ligne = self._extract_inline_value(texte, field_type)

                # Si pas de valeur sur la même ligne, chercher sur la ligne suivante
                if not valeur_meme_ligne and i + 1 < len(lines):
                    next_texte = lines[i + 1][0]
                    next_field = identify_field_type(next_texte)
                    if next_field == "inconnu" and len(next_texte) > 1:
                        valeur_meme_ligne = next_texte

                if valeur_meme_ligne:
                    result = self.corrector.correct(valeur_meme_ligne, field_type)
                    # Garder le meilleur résultat pour chaque type de champ
                    if field_type not in kpis or result["confiance"] > kpis[field_type]["confiance"]:
                        kpis[field_type] = result
                        kpis[field_type]["raw"] = valeur_meme_ligne

        return kpis

    def _extract_inline_value(self, texte: str, field_type: str) -> Optional[str]:
        """
        Extrait la valeur qui suit un label sur la même ligne.

        Ex: "Commune : Vals-les-Bains" → "Vals-les-Bains"
            "Echelle 1/1000"           → "1/1000"
            "Section B"                → "B"
        """
        # Retirer les mots-clés du champ
        keywords = FIELD_KEYWORDS.get(field_type, [])
        txt = texte
        for kw in sorted(keywords, key=len, reverse=True):
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            txt = pattern.sub("", txt, count=1)

        # Retirer les séparateurs de début
        txt = re.sub(r"^[\s:;\-\|\.]+", "", txt).strip()

        if len(txt) > 1:
            return txt
        return None


# ===========================================================================
# 5. MOTEUR DE COHÉRENCE CROISÉE
# ===========================================================================

class CrossFieldValidator:
    """
    Vérifie la cohérence entre les différents KPIs extraits.

    Règles métier :
    - Si commune = "Vals-les-Bains" et département = "07", cohérent.
    - Si commune ≠ commune dans l'en-tête du livret, incohérent → alerte.
    - Si échelle = "1/50000" pour un plan de bornage, anormal → alerte.
    - Si la commune est dans le CSV comme "Barnas" mais le score < 70%,
      proposer "Vals-les-Bains" si c'est la commune la plus fréquente du livret.
    """

    def validate(
        self,
        kpis: Dict[str, Any],
        commune_db: List[Dict[str, Any]],
        doc_type: str = "inconnu"
    ) -> Dict[str, Any]:
        """
        Retourne les KPIs enrichis avec des alertes de cohérence.
        """
        alerts = []
        suggestions = {}

        # Règle 1 : l'échelle doit être dans les valeurs cadastrales connues
        echelle = kpis.get("echelle", {}).get("valeur", "")
        if echelle and echelle not in DocumentSemanticAnalyzer.ECHELLES_VALIDES:
            # Chercher la valeur la plus proche
            closest = self._closest_echelle(echelle, DocumentSemanticAnalyzer.ECHELLES_VALIDES)
            if closest:
                alerts.append(f"Échelle '{echelle}' non standard → suggestion : {closest}")
                suggestions["echelle"] = closest

        # Règle 2 : cohérence commune / département
        commune_info = kpis.get("commune", {})
        code = commune_info.get("code", "")
        if code and not code.startswith("07") and not code.startswith("26"):
            alerts.append(f"Commune (INSEE {code}) hors zone Ardèche/Drôme attendue.")

        # Règle 3 : cohérence type de document / échelle
        if doc_type in ("Plan de Bornage", "Document d'Arpentage (DMPC)"):
            echelle_num = self._parse_echelle(echelle)
            if echelle_num and echelle_num > 5000:
                alerts.append(f"Échelle 1/{echelle_num} inhabituelle pour un plan de bornage.")

        return {
            **kpis,
            "_alerts": alerts,
            "_suggestions": suggestions,
        }

    def _closest_echelle(self, echelle: str, valides: set) -> Optional[str]:
        """Trouve l'échelle valide la plus proche numériquement."""
        num = self._parse_echelle(echelle)
        if not num:
            return None
        best = None
        best_diff = float("inf")
        for v in valides:
            n = self._parse_echelle(v)
            if n and abs(n - num) < best_diff:
                best_diff = abs(n - num)
                best = v
        return best

    def _parse_echelle(self, echelle: str) -> Optional[int]:
        """Extrait le dénominateur d'une échelle."""
        m = re.search(r"1\s*/\s*(\d+)", str(echelle))
        if m:
            return int(m.group(1))
        return None


# ===========================================================================
# 6. FONCTION UTILITAIRE PRINCIPALE
# ===========================================================================

def process_with_semantic_context(
    raw_detections: List[Dict[str, Any]],
    commune_db: List[Dict[str, Any]],
    doc_type: str = "inconnu",
    script_dir: str = ".",
) -> Dict[str, Any]:
    """
    Point d'entrée principal du moteur sémantique.

    Usage depuis main.py :
        from semantic_ocr_engine import process_with_semantic_context
        kpis_sem = process_with_semantic_context(page_results, commune_db, type_doc)
        # Fusionner avec les KPIs existants
        kpis.update({k: v["valeur"] for k, v in kpis_sem.items() if v.get("confiance", 0) > 0.7})

    Retourne un dict de KPIs validés avec leur confiance et leurs alertes.
    """
    analyzer = DocumentSemanticAnalyzer(commune_db, script_dir)
    validator = CrossFieldValidator()

    kpis = analyzer.analyze(raw_detections)
    kpis_validated = validator.validate(kpis, commune_db, doc_type)

    # Log des alertes
    for alert in kpis_validated.get("_alerts", []):
        print(f"  [SemanticEngine] ⚠️  {alert}")

    return kpis_validated


# ===========================================================================
# 7. APPRENTISSAGE : INTÉGRATION DES CORRECTIONS UTILISATEUR
# ===========================================================================

class CorrectionLearner:
    """
    Mémorise les corrections faites par l'utilisateur dans l'interface
    de validation et les réapplique automatiquement lors des prochains
    traitements du même géomètre.

    Format de stockage (JSON) :
    {
      "geometre_id": "livretFernand",
      "field_corrections": {
        "commune": {"BAINS": "Vals-les-Bains", "VALS": "Vals-les-Bains"},
        "echelle": {"1f1000": "1/1000"},
        "section": {"0": "O"},
      },
      "context_corrections": [
        {"label_context": "COMMUNE", "ocr_brut": "BAINS",
         "correction": "Vals-les-Bains", "count": 3}
      ]
    }
    """

    def __init__(self, geometre_id: str, base_dir: str = "writer_styles"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(script_dir, base_dir, f"{geometre_id}_semantic.json")
        self.geometre_id = geometre_id
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "geometre_id": self.geometre_id,
            "field_corrections": {},
            "context_corrections": [],
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def record(self, field_type: str, ocr_brut: str, correction: str) -> None:
        """Enregistre une correction utilisateur."""
        ocr_norm = _normalize(ocr_brut)
        self.data.setdefault("field_corrections", {}).setdefault(field_type, {})[ocr_norm] = correction

        # Mettre à jour le compteur dans context_corrections
        existing = next(
            (c for c in self.data.get("context_corrections", [])
             if c.get("field_type") == field_type and c.get("ocr_brut") == ocr_norm),
            None
        )
        if existing:
            existing["count"] = existing.get("count", 1) + 1
            existing["correction"] = correction
        else:
            self.data.setdefault("context_corrections", []).append({
                "field_type": field_type,
                "ocr_brut": ocr_norm,
                "correction": correction,
                "count": 1,
            })

        self.save()
        print(f"  [CorrectionLearner] Appris : [{field_type}] '{ocr_brut}' → '{correction}'")

    def lookup(self, field_type: str, ocr_brut: str) -> Optional[str]:
        """Cherche une correction déjà apprise."""
        ocr_norm = _normalize(ocr_brut)
        corrections = self.data.get("field_corrections", {}).get(field_type, {})
        return corrections.get(ocr_norm)

    def get_all_corrections(self) -> Dict[str, Dict[str, str]]:
        """Retourne toutes les corrections par type de champ."""
        return self.data.get("field_corrections", {})
