"""
coherence_checker.py — Vérification de cohérence inter-champs (4 couches)
─────────────────────────────────────────────────────────────────────────
COUCHE A : Rule Engine  — règles métier Python pures (instantané)
COUCHE B : Pydantic     — validation schéma typé par type de plan (instantané)
COUCHE C : Ollama LLM   — raisonnement local (optionnel, ~5-15s, graceful)
COUCHE D : Knowledge Graph — base communes/sections/dossiers (instantané)
"""

import re, json, os, unicodedata
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# ─── Chargement de la base de connaissances (D) ──────────────────────────────
_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
_KNOWLEDGE_BASE: Dict = {}

def _load_kb() -> Dict:
    global _KNOWLEDGE_BASE
    if not _KNOWLEDGE_BASE and os.path.exists(_KB_PATH):
        try:
            with open(_KB_PATH, encoding="utf-8") as f:
                _KNOWLEDGE_BASE = json.load(f)
        except Exception as e:
            print(f"[Coherence] Impossible de charger knowledge_base.json : {e}")
    return _KNOWLEDGE_BASE

def _save_kb_discovery(commune: str, section: str):
    """Enrichit la base de connaissances si une nouvelle association commune/section est validée."""
    kb = _load_kb()
    communes = kb.get("communes", {})
    if commune not in communes:
        communes[commune] = {"sections_connues": [], "prefixes_dossier": []}
    sections = communes[commune].get("sections_connues", [])
    if section and section not in sections:
        sections.append(section)
        communes[commune]["sections_connues"] = sections
        kb["communes"] = communes
        try:
            with open(_KB_PATH, "w", encoding="utf-8") as f:
                json.dump(kb, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


# ─── Types de résultats ───────────────────────────────────────────────────────
NIVEAU_ERREUR    = "ERREUR"
NIVEAU_ALERTE    = "ALERTE"
NIVEAU_INFO      = "INFO"

def _issue(level: str, rule: str, message: str, fields: List[str], source: str) -> Dict:
    return {"level": level, "rule": rule, "message": message, "fields": fields, "source": source}


# ═══════════════════════════════════════════════════════════════════════════════
# COUCHE A — Rule Engine métier
# ═══════════════════════════════════════════════════════════════════════════════
_VILLES_DEPARTEMENTS = {
    "ARDECHE", "DROME", "GARD", "LOIRE", "RHONE", "HERAULT", "LOZERE",
    "FRANCE", "VALS LES BAINS", "AUBENAS", "PRIVAS", "ANNONAY",
    "GUILHERAND", "GUILHERAND GRANGES", "VALLON PONT D ARC",
}

def _nc(t: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(t))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9 ]", " ", s.upper()).strip()

def _extract_year_from_dossier(n_ordre: str) -> Optional[int]:
    """Extrait l'année d'un numéro de dossier modern (ex: A09032 → 2009, A22-145 → 2022)."""
    m = re.match(r'^[A-Z](\d{2})', n_ordre.upper().strip())
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy <= 35 else 1900 + yy
    return None

def _extract_year_from_date(date_str: str) -> Optional[int]:
    m = re.search(r'\b(\d{4})\b', str(date_str))
    if m:
        return int(m.group(1))
    return None

def rule_engine_check(champs: Dict, type_plan: str) -> List[Dict]:
    issues = []

    commune   = str(champs.get("commune", {}).get("valeur", "") if isinstance(champs.get("commune"), dict) else "").strip()
    section   = str(champs.get("section", {}).get("valeur", "") if isinstance(champs.get("section"), dict) else "").strip()
    date_val  = str(champs.get("date", {}).get("valeur", "") if isinstance(champs.get("date"), dict) else "").strip()
    n_ordre   = str(champs.get("n_ordre", {}).get("valeur", "") if isinstance(champs.get("n_ordre"), dict) else "").strip()
    n_dossier = str(champs.get("n_dossier", {}).get("valeur", "") if isinstance(champs.get("n_dossier"), dict) else "").strip()
    geometre  = str(champs.get("geometre", {}).get("valeur", "") if isinstance(champs.get("geometre"), dict) else "").strip()
    echelle   = str(champs.get("echelle", {}).get("valeur", "") if isinstance(champs.get("echelle"), dict) else "").strip()

    prop_anc_val = champs.get("proprietaires_anciens", {})
    prop_nouv_val = champs.get("proprietaires_nouveaux", {})
    prop_anc  = str(prop_anc_val.get("valeur", "") if isinstance(prop_anc_val, dict) else "").strip()
    prop_nouv = str(prop_nouv_val.get("valeur", "") if isinstance(prop_nouv_val, dict) else "").strip()

    # R1 — Date dans plage plausible
    yr = _extract_year_from_date(date_val)
    if yr is not None:
        if not (1900 <= yr <= 2030):
            issues.append(_issue(NIVEAU_ERREUR, "DATE_HORS_PLAGE",
                f"Année {yr} hors de la plage 1900-2030 pour un document cadastral.",
                ["date"], "rule_engine"))
    elif date_val:
        issues.append(_issue(NIVEAU_ALERTE, "DATE_FORMAT_INCONNU",
            f"Date '{date_val}' : année non extraite, format suspect.",
            ["date"], "rule_engine"))

    # R2 — Cohérence année dossier ↔ date du document
    ref_dossier = n_ordre or n_dossier
    if ref_dossier:
        yr_dossier = _extract_year_from_dossier(ref_dossier)
        if yr_dossier and yr:
            if yr < yr_dossier:
                issues.append(_issue(NIVEAU_ERREUR, "DATE_DOSSIER_INCOHERENCE",
                    f"Le dossier '{ref_dossier}' suggère l'année {yr_dossier}, "
                    f"mais la date du document est {yr} (antérieure).",
                    ["n_ordre", "date"], "rule_engine"))
            elif yr > yr_dossier + 3:
                issues.append(_issue(NIVEAU_ALERTE, "DATE_DOSSIER_ECART",
                    f"Écart de {yr - yr_dossier} ans entre le dossier ({yr_dossier}) "
                    f"et la date du document ({yr}). Vérifier.",
                    ["n_ordre", "date"], "rule_engine"))

    # R3 — Commune obligatoire si section présente
    if section and not commune:
        issues.append(_issue(NIVEAU_ALERTE, "SECTION_SANS_COMMUNE",
            f"La section '{section}' est renseignée mais la commune est vide.",
            ["commune", "section"], "rule_engine"))

    # R4 — Propriétaires anciens ≠ nouveaux (doublon OCR suspect)
    if prop_anc and prop_nouv and len(prop_anc) > 5:
        if _nc(prop_anc) == _nc(prop_nouv):
            issues.append(_issue(NIVEAU_ALERTE, "PROP_ANC_NOUV_IDENTIQUES",
                f"Propriétaires anciens et nouveaux identiques : '{prop_anc}'. "
                "Probablement un glissement OCR.",
                ["proprietaires_anciens", "proprietaires_nouveaux"], "rule_engine"))

    # R5 — Géomètre ne doit pas être une ville ou un département
    if geometre:
        geo_nc = _nc(geometre)
        if geo_nc in _VILLES_DEPARTEMENTS:
            issues.append(_issue(NIVEAU_ERREUR, "GEOMETRE_EST_VILLE",
                f"Le géomètre '{geometre}' ressemble à un nom de ville/département.",
                ["geometre"], "rule_engine"))
        elif len(geometre) < 5:
            issues.append(_issue(NIVEAU_ALERTE, "GEOMETRE_TROP_COURT",
                f"Géomètre '{geometre}' trop court pour être un vrai nom.",
                ["geometre"], "rule_engine"))
        else:
            # ── FIX 6 : R5b — Géomètre suspect (aucun nom propre parmi les mots) ──
            # Un vrai nom de géomètre/cabinet contient au moins un mot qui
            # commence par une majuscule suivie de minuscules (nom propre).
            # Si tous les mots sont en majuscules ou sont des mots fonctionnels,
            # c'est probablement un fragment de légende de cartouche.
            mots_geo = geometre.strip().split()
            _MOTS_FONCTIONNELS_GEO = {
                "par", "le", "la", "les", "du", "de", "des", "au", "aux",
                "dresse", "dressé", "dresser", "cachet", "service", "bureau",
                "origine", "dorigine", "apres", "après", "plon",
            }
            has_proper = any(
                re.match(r'^[A-ZÀÂÆÇÉÈÊËÎÏÔÙÛÜŸ][a-zàâæçéèêëîïôùûüÿ]{2,}$', m)
                for m in mots_geo
                if m.lower() not in _MOTS_FONCTIONNELS_GEO
            )
            if len(mots_geo) >= 3 and not has_proper:
                issues.append(_issue(NIVEAU_ALERTE, "GEOMETRE_SUSPECT",
                    f"Géomètre '{geometre}' : aucun nom propre identifié parmi les mots "
                    f"({len(mots_geo)} mots, aucune majuscule-minuscule). "
                    "Probablement un fragment de légende de cartouche.",
                    ["geometre"], "rule_engine"))

    # R6 — Échelle cohérente pour le cadastre (1/200 à 1/10000)
    if echelle:
        m_ech = re.search(r'(\d{3,5})', echelle.replace(" ", ""))
        if m_ech:
            val_ech = int(m_ech.group(1))
            if not (200 <= val_ech <= 10000):
                issues.append(_issue(NIVEAU_ALERTE, "ECHELLE_HORS_NORME",
                    f"Échelle 1/{val_ech} inhabituelle pour un plan cadastral (attendu : 1/200 à 1/10000).",
                    ["echelle"], "rule_engine"))

    # R7 — Format n_ordre compatible avec le type de plan
    if n_ordre and type_plan == "DMPC":
        if re.match(r'^\d{1,5}[A-Z]$', n_ordre.upper()):
            issues.append(_issue(NIVEAU_INFO, "DOSSIER_FORMAT_INATTENDU",
                f"Format DA classique '{n_ordre}' sur un DMPC (attendu: format récent A09...).",
                ["n_ordre"], "rule_engine"))

    # R8 — Section doit être des lettres pour PVa/PLa
    if section and type_plan in ("PVa", "PLa"):
        if re.match(r'^\d+$', section):
            issues.append(_issue(NIVEAU_ALERTE, "SECTION_CHIFFRES_SEULEMENT",
                f"Section '{section}' entièrement numérique sur un {type_plan} (attendu: lettres).",
                ["section"], "rule_engine"))

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# COUCHE B — Pydantic Validation
# ═══════════════════════════════════════════════════════════════════════════════
def pydantic_check(champs: Dict, type_plan: str) -> List[Dict]:
    issues = []
    try:
        from pydantic import BaseModel, field_validator, model_validator
        from pydantic import ValidationError
        from typing import Optional as Opt

        def gv(field: str) -> str:
            v = champs.get(field, {})
            val = v.get("valeur", "") if isinstance(v, dict) else ""
            return str(val).strip() if val else ""

        class PlanModel(BaseModel):
            commune:   str = ""
            section:   str = ""
            date:      str = ""
            n_ordre:   str = ""
            geometre:  str = ""
            echelle:   str = ""
            type_plan: str = ""

            @field_validator("date")
            @classmethod
            def validate_date(cls, v):
                if not v:
                    return v
                if not re.search(r'\d{4}', v):
                    raise ValueError(f"Date '{v}' sans année reconnaissable.")
                return v

            @field_validator("section")
            @classmethod
            def validate_section(cls, v):
                if v and not re.match(r'^[A-Za-z]{1,3}\d{0,2}$', v):
                    raise ValueError(f"Section '{v}' format non standard.")
                return v

            @field_validator("echelle")
            @classmethod
            def validate_echelle(cls, v):
                if v:
                    m = re.search(r'(\d{3,5})', v.replace(" ", ""))
                    if m and not (200 <= int(m.group(1)) <= 10000):
                        raise ValueError(f"Échelle '{v}' hors norme cadastrale.")
                return v

            @model_validator(mode="after")
            def cross_validate(self):
                if self.section and not self.commune:
                    raise ValueError("Section renseignée mais commune absente.")
                return self

        try:
            PlanModel(
                commune=gv("commune"),
                section=gv("section"),
                date=gv("date"),
                n_ordre=gv("n_ordre"),
                geometre=gv("geometre"),
                echelle=gv("echelle"),
                type_plan=type_plan,
            )
        except ValidationError as e:
            for err in e.errors():
                loc = err.get("loc", ("",))
                # model_validator errors ont un loc vide () — on utilise "model" comme champ
                field_loc = str(loc[0]) if loc else "model"
                issues.append(_issue(
                    NIVEAU_ALERTE, f"PYDANTIC_{err['type'].upper()}",
                    err["msg"],
                    [field_loc], "pydantic"
                ))

    except ImportError:
        issues.append(_issue(NIVEAU_INFO, "PYDANTIC_ABSENT",
            "Pydantic non installé (pip install pydantic). Couche B ignorée.",
            [], "pydantic"))

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# COUCHE C — Ollama LLM (optionnel, graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════════
_OLLAMA_AVAILABLE: Optional[bool] = None

def _check_ollama() -> bool:
    global _OLLAMA_AVAILABLE
    if _OLLAMA_AVAILABLE is not None:
        return _OLLAMA_AVAILABLE
    try:
        import subprocess, json
        # Contournement pare-feu Windows via curl.exe
        res = subprocess.run(["curl.exe", "-s", "http://127.0.0.1:11434/api/tags"], 
                             capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and "models" in res.stdout:
            _OLLAMA_AVAILABLE = True
        else:
            _OLLAMA_AVAILABLE = False
    except Exception:
        _OLLAMA_AVAILABLE = False
    if not _OLLAMA_AVAILABLE:
        print("[Coherence] Ollama non disponible — couche C désactivée (normal si non installé).")
    return _OLLAMA_AVAILABLE


def ollama_check(champs: Dict, type_plan: str) -> List[Dict]:
    issues = []
    if not _check_ollama():
        return issues

    try:
        import subprocess, json as _json

        # Résumé compact des champs pour le prompt
        def gv(f):
            v = champs.get(f, {})
            val = v.get("valeur", "") if isinstance(v, dict) else ""
            return str(val).strip() if val else "(vide)"

        summary = (
            f"Type de document: {type_plan}\n"
            f"Commune: {gv('commune')}\n"
            f"Section: {gv('section')}\n"
            f"Date: {gv('date')}\n"
            f"N° Dossier: {gv('n_ordre') or gv('n_dossier')}\n"
            f"Géomètre: {gv('geometre')}\n"
            f"Échelle: {gv('echelle')}\n"
            f"Prop. anciens: {gv('proprietaires_anciens')}\n"
            f"Prop. nouveaux: {gv('proprietaires_nouveaux')}\n"
            f"Parcelles: {gv('parcelles')}\n"
        )

        prompt = (
            "Tu es un expert en géomatique et en documents cadastraux français.\n"
            "Voici les champs extraits d'un document par OCR :\n\n"
            f"{summary}\n"
            "Identifie les incohérences, les valeurs suspectes ou les contradictions entre ces champs.\n"
            "Réponds UNIQUEMENT en JSON, sans texte avant ou après, au format :\n"
            '{"issues": [{"level": "ERREUR"|"ALERTE"|"INFO", "fields": ["field1"], "message": "..."}]}'
        )

        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512}
        }

        # Écriture dans un fichier temporaire pour curl.exe
        os.makedirs("outputs", exist_ok=True)
        payload_path = os.path.join(os.getcwd(), "outputs", "ollama_coherence.json")
        with open(payload_path, "w", encoding="utf-8") as f:
            _json.dump(payload, f)
            
        win_payload_path = payload_path.replace("/mnt/c/", "C:\\").replace("/", "\\")
        cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate", 
               "-H", "Content-Type: application/json", "-d", f"@{win_payload_path}"]
               
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # I7: 60s timeout
        
        if r.returncode == 0 and r.stdout.strip():
            raw = _json.loads(r.stdout).get("response", "")
            # Extraire le JSON de la réponse
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                data = _json.loads(m.group(0))
                for item in data.get("issues", []):
                    issues.append(_issue(
                        item.get("level", NIVEAU_ALERTE),
                        "LLM_COHERENCE",
                        item.get("message", ""),
                        item.get("fields", []),
                        "ollama_llm"
                    ))

    except subprocess.TimeoutExpired:
        print("[Coherence] Ollama — timeout (> 60s). Couche C ignorée pour ce document.")
    except Exception as e:
        print(f"[Coherence] Ollama — erreur lors de l'appel : {type(e).__name__}: {e}")

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# COUCHE D — Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════════
def knowledge_graph_check(champs: Dict, type_plan: str) -> List[Dict]:
    issues = []
    kb = _load_kb()
    communes_kb = kb.get("communes", {})
    if not communes_kb:
        return issues

    def gv(f):
        v = champs.get(f, {})
        val = v.get("valeur", "") if isinstance(v, dict) else ""
        return str(val).strip() if val else ""

    commune = gv("commune")
    section = gv("section")
    n_ordre = gv("n_ordre") or gv("n_dossier")

    # Trouver la commune dans la base (fuzzy)
    commune_entry = None
    commune_key = None
    if commune:
        comm_nc = _nc(commune)
        for key, val in communes_kb.items():
            if _nc(key) == comm_nc:
                commune_entry = val
                commune_key = key
                break
        # Fuzzy si pas trouvé exactement
        if not commune_entry:
            try:
                from rapidfuzz import process as rfp, fuzz as rfz
                keys = list(communes_kb.keys())
                best = rfp.extractOne(comm_nc, [_nc(k) for k in keys], scorer=rfz.WRatio)
                if best and best[1] >= 85:
                    commune_key = keys[best[2]]
                    commune_entry = communes_kb[commune_key]
            except ImportError:
                pass

    if commune_entry:
        # D1 — Section connue dans cette commune ?
        sections_connues = commune_entry.get("sections_connues", [])
        if section and sections_connues:
            if section.upper() not in [s.upper() for s in sections_connues]:
                issues.append(_issue(NIVEAU_ALERTE, "SECTION_INCONNUE_COMMUNE",
                    f"La section '{section}' n'est pas répertoriée pour '{commune_key}'. "
                    f"Sections connues : {', '.join(sections_connues)}.",
                    ["commune", "section"], "knowledge_graph"))
            else:
                # Auto-enrichissement silencieux si déjà connu
                pass

        # D2 — Préfixe dossier connu pour cette commune ?
        prefixes = commune_entry.get("prefixes_dossier", [])
        if n_ordre and prefixes:
            yr_d = _extract_year_from_dossier(n_ordre)
            if yr_d:
                pfx = f"A{str(yr_d)[2:].zfill(2)}"
                if pfx not in prefixes:
                    issues.append(_issue(NIVEAU_INFO, "DOSSIER_PREFIXE_NOUVEAU",
                        f"Le préfixe '{pfx}' (déduit de '{n_ordre}') n'est pas encore "
                        f"enregistré pour '{commune_key}'. "
                        "Si le document est valide, il sera ajouté automatiquement.",
                        ["n_ordre"], "knowledge_graph"))

        # D3 — Type de plan fréquent dans cette commune ?
        types_freq = commune_entry.get("types_plans_frequents", [])
        if types_freq and type_plan not in types_freq:
            issues.append(_issue(NIVEAU_INFO, "TYPE_PLAN_RARE_COMMUNE",
                f"Le type '{type_plan}' est rare pour '{commune_key}' "
                f"(types habituels : {', '.join(types_freq)}).",
                ["commune"], "knowledge_graph"))

    else:
        if commune:
            issues.append(_issue(NIVEAU_INFO, "COMMUNE_ABSENTE_KB",
                f"Commune '{commune}' absente de la base de connaissances. "
                "Elle sera ajoutée si la validation humaine confirme.",
                ["commune"], "knowledge_graph"))

    # Auto-enrichissement : si commune + section sont validées, on enrichit la base
    # O4 : L'enrichissement automatique KB est désactivé — il sera déclenché UNIQUEMENT
    # après validation humaine dans app_validation.py (bouton 'Enregistrer & Suivant').
    # Raison : une section OCR erronée pourrait sinon être persistée définitivement.
    # if commune_key and section and not issues:
    #     _save_kb_discovery(commune_key, section.upper())

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# FONCTION MAÎTRE — Combine les 4 couches
# ═══════════════════════════════════════════════════════════════════════════════
def check_coherence(champs: Dict, type_plan: str, use_llm: bool = True) -> Dict:
    """
    Vérifie la cohérence de l'ensemble des champs extraits via 4 couches :
    A (rules) → B (pydantic) → C (ollama, optionnel) → D (knowledge graph)

    Retourne un rapport structuré :
    {
        "coherence_score": 0.85,      # Score global 0-1
        "status": "CONFORME",          # CONFORME / ALERTE / REJET
        "issues": [...],               # Liste de tous les problèmes
        "issues_by_field": {...},      # Problèmes groupés par champ
        "summary": "..."               # Résumé lisible
    }
    """
    all_issues: List[Dict] = []

    print(f"\n  [Coherence] == Verification 4 couches (type: {type_plan}) ==")

    # Couche A
    issues_a = rule_engine_check(champs, type_plan)
    all_issues.extend(issues_a)
    print(f"    [A] Rule Engine  : {len(issues_a)} problème(s)")

    # Couche B
    issues_b = pydantic_check(champs, type_plan)
    all_issues.extend(issues_b)
    print(f"    [B] Pydantic     : {len(issues_b)} problème(s)")

    # Couche C (seulement si demandé et Ollama dispo)
    if use_llm:
        issues_c = ollama_check(champs, type_plan)
        all_issues.extend(issues_c)
        print(f"    [C] Ollama LLM   : {len(issues_c)} problème(s)")
    else:
        print(f"    [C] Ollama LLM   : désactivé")

    # Couche D
    issues_d = knowledge_graph_check(champs, type_plan)
    all_issues.extend(issues_d)
    print(f"    [D] Knowledge DB : {len(issues_d)} problème(s)")

    # Déduplication (éviter les doublons A+B sur le même champ/règle)
    seen = set()
    unique_issues = []
    for iss in all_issues:
        key = (iss["rule"], tuple(sorted(iss["fields"])))
        if key not in seen:
            seen.add(key)
            unique_issues.append(iss)

    # Calcul du score de cohérence
    n_errors   = sum(1 for i in unique_issues if i["level"] == NIVEAU_ERREUR)
    n_alerts   = sum(1 for i in unique_issues if i["level"] == NIVEAU_ALERTE)
    n_info     = sum(1 for i in unique_issues if i["level"] == NIVEAU_INFO)

    # I2 : Les INFO d'enrichissement (COMMUNE_ABSENTE_KB, DOSSIER_PREFIXE_NOUVEAU) ne
    # pénalisent plus le score — ce sont des informations neutres, pas des problèmes.
    score = max(0.0, 1.0 - (n_errors * 0.25) - (n_alerts * 0.08))
    score = round(score, 2)

    if n_errors > 0:
        status = "REJET"
    elif n_alerts > 0:
        status = "ALERTE"
    else:
        status = "CONFORME"

    # Grouper par champ
    by_field: Dict[str, List] = {}
    for iss in unique_issues:
        for f in iss["fields"]:
            by_field.setdefault(f, []).append(iss)

    # Résumé lisible
    if status == "CONFORME":
        summary = f"Aucune incohérence détectée. Score : {score:.0%}"
    else:
        parts = []
        if n_errors: parts.append(f"{n_errors} erreur(s) bloquante(s)")
        if n_alerts: parts.append(f"{n_alerts} alerte(s)")
        if n_info:   parts.append(f"{n_info} info(s)")
        summary = f"{status} — {', '.join(parts)}. Score : {score:.0%}"

    # Log
    for iss in unique_issues:
        icon = "[X]" if iss["level"] == NIVEAU_ERREUR else ("[i]" if iss["level"] == NIVEAU_ALERTE else "[i]")
        print(f"    {icon} [{iss['source']}] {iss['rule']}: {iss['message']}")

    print(f"  [Coherence] Resultat : {status} ({score:.0%}) - {summary}")

    return {
        "coherence_score": score,
        "status": status,
        "issues": unique_issues,
        "issues_by_field": by_field,
        "summary": summary,
        "counts": {"erreurs": n_errors, "alertes": n_alerts, "infos": n_info},
    }
