import streamlit as st
import pandas as pd
import os, glob, json, re, unicodedata
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="Validation Cadastrale", layout="wide", page_icon="🗺️")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.header { background: linear-gradient(135deg,#1a2b4a,#2563eb);
    color:#fff; padding:1rem 1.5rem; border-radius:8px; margin-bottom:1rem; }
.header h1 { margin:0; font-size:1.3rem; font-weight:700; }
.header p { margin:0.2rem 0 0; font-size:0.8rem; color:#93c5fd; }
.field-row { display:flex; align-items:center; padding:0.4rem 0.7rem;
    border-radius:6px; margin-bottom:0.3rem; cursor:pointer;
    border:1px solid #e5e7eb; background:#f9fafb; transition:all 0.15s; }
.field-row:hover { background:#eff6ff; border-color:#3b82f6; }
.field-row.active { background:#dbeafe; border-color:#2563eb; border-width:2px; }
.field-lbl { font-size:0.68rem; font-weight:600; text-transform:uppercase;
    color:#6b7280; letter-spacing:0.06em; width:130px; flex-shrink:0; }
.field-val { font-size:0.88rem; font-weight:500; color:#111827; flex:1; }
.field-val.empty { color:#9ca3af; font-style:italic; }
.match-hint { font-size:0.7rem; color:#16a34a; margin-top:0.05rem; }
.badge-ok { background:#dcfce7; color:#166534; border:1px solid #bbf7d0;
    padding:0.1rem 0.5rem; border-radius:12px; font-size:0.65rem; font-weight:600; }
.badge-? { background:#fef9c3; color:#854d0e; border:1px solid #fde68a;
    padding:0.1rem 0.5rem; border-radius:12px; font-size:0.65rem; font-weight:600; }
.zoom-caption { background:#1e3a5f; color:#fff; font-size:0.75rem; font-weight:600;
    padding:0.3rem 0.8rem; border-radius:6px 6px 0 0; text-align:center; }
div[data-testid="stSidebar"] { min-width:220px !important; }
</style>
""", unsafe_allow_html=True)

OUTPUTS_DIR = "outputs"
INPUTS_DIR  = "inputs"

# ─── Base communes ───────────────────────────────────────────────
@st.cache_data
def load_commune_db():
    db = []
    for fname in ["ardeche.json", "communes_france.json"]:
        if os.path.exists(fname):
            try:
                for e in json.load(open(fname, encoding="utf-8")):
                    n = e.get("nom","").strip()
                    if n: db.append({"officiel": n, "code": e.get("code","")})
            except Exception: pass
    return db

def _nc(t):
    nfkd = unicodedata.normalize("NFKD", str(t))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^A-Z0-9 ]"," ", re.sub(r"[-''`]"," ",s).upper()).strip()

@st.cache_data
def match_commune(text):
    if not text or len(text.strip()) < 3: return text, 0, ""
    try:
        from rapidfuzz import process as rfp, fuzz
    except ImportError: return text, 0, ""
    db = load_commune_db()
    if not db: return text, 0, ""
    noms = [_nc(e["officiel"]) for e in db]
    r = rfp.extractOne(_nc(text), noms, scorer=fuzz.WRatio)
    if r and r[1] >= 50:
        e = db[r[2]]
        return e["officiel"], r[1], e.get("code","")
    return text, 0, ""

# ─── CSV ─────────────────────────────────────────────────────────
all_csv = sorted(glob.glob(os.path.join(OUTPUTS_DIR,"*_plan_resultats.csv")))
if not all_csv:
    all_csv = sorted(glob.glob(os.path.join(OUTPUTS_DIR,"*_resultats.csv")))
if not all_csv:
    st.warning("Aucun résultat dans 'outputs/'. Lancez d'abord `python main.py`.")
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Document**")
    fichier_choisi = st.selectbox("Fichier", all_csv,
        format_func=lambda p: os.path.basename(p)
            .replace("_plan_resultats.csv","").replace("_resultats.csv",""))

@st.cache_data
def load_csv(p):
    try: return pd.read_csv(p, sep=";", encoding="utf-8-sig")
    except: return pd.read_csv(p, sep=";", encoding="utf-8")

df = load_csv(fichier_choisi)
base_name = os.path.basename(fichier_choisi)\
    .replace("_plan_resultats.csv","").replace("_resultats.csv","")

if "Confirmation_Status" not in df.columns:
    df["Confirmation_Status"] = "À valider"
DONE = ["Validé par l'humain","Corrigé automatiquement","Ignoré (Vide)"]
id_col = "ID" if "ID" in df.columns else "ID_Ligne"

with st.sidebar:
    n_val = len(df[df["Confirmation_Status"].isin(DONE)])
    st.progress(n_val / max(len(df),1))
    st.caption(f"{n_val}/{len(df)} validé(s)")
    plan_type = "GENERIC"
    for c in ["Type_Plan","Type_Document"]:
        if c in df.columns:
            v = df[c].dropna().unique()
            if len(v): plan_type = str(v[0]); break
    st.markdown(f"**Type :** `{plan_type}`")

# ─── JSON compagnon ───────────────────────────────────────────────
_json = {}
for suf in [f"_plan_{plan_type}.json","_plan_PLAN.json","_plan_moderne.json"]:
    jp = os.path.join(OUTPUTS_DIR, base_name+suf)
    if os.path.exists(jp):
        try: _json = json.load(open(jp, encoding="utf-8"))
        except: pass
        break

def champs_for_page(pn):
    for pg in _json.get("pages",[]):
        if pg.get("page")==pn: return pg.get("champs",{})
    return {}

def jval(champs, field):
    v = champs.get(field,{})
    if isinstance(v,dict):
        val = v.get("valeur","")
        return ", ".join(str(x) for x in val) if isinstance(val,list) else str(val)
    return ""

def jzone(champs, field):
    v = champs.get(field,{})
    return v.get("zone") if isinstance(v,dict) else None

# ─── Champs définis par type de plan ─────────────────────────────
# Libellés et ordre d'affichage selon le type
CHAMPS_LABELS = {
    "commune":    "Commune",
    "n_ordre":    "N° d'ordre / DA",
    "n_dossier":  "N° dossier",
    "section":    "Section",
    "feuille":    "Feuille",
    "date":       "Date",
    "echelle":    "Échelle",
    "geometre":   "Géomètre-Expert",
    "signataires":"Signataires",
    "proprietaires_anciens":  "Prop. anciens",
    "proprietaires_nouveaux": "Prop. nouveaux",
    "parcelles":  "Parcelles",
    "indication": "Objet / Indication",
}

# Champs prioritaires par type (dans l'ordre d'affichage)
CHAMPS_PAR_TYPE = {
    "PVa":  ["commune","n_ordre","section","feuille","date","echelle",
             "geometre","proprietaires_anciens","proprietaires_nouveaux",
             "parcelles","signataires","indication"],
    "PLa":  ["commune","n_ordre","section","feuille","date","echelle",
             "geometre","proprietaires_anciens","proprietaires_nouveaux",
             "parcelles","signataires","indication"],
    "DMPC": ["commune","n_ordre","n_dossier","section","feuille","date",
             "echelle","geometre","parcelles","signataires","indication"],
    "GENERIC": list(CHAMPS_LABELS.keys()),
}
CSV_TO_JSON = {
    "Commune":"commune","N_Ordre":"n_ordre","N_Dossier":"n_dossier",
    "Section":"section","Feuille":"feuille","Date":"date",
    "Echelle":"echelle","Geometre":"geometre","Signataires":"signataires",
    "Proprietaires_Anciens":"proprietaires_anciens",
    "Proprietaires_Nouveaux":"proprietaires_nouveaux",
    "Parcelles":"parcelles","Indication":"indication",
}
MULTI = {"parcelles","signataires","proprietaires_anciens","proprietaires_nouveaux"}

# ─── Page à traiter ───────────────────────────────────────────────
pending = df[~df["Confirmation_Status"].isin(DONE)][id_col].tolist() \
          if id_col in df.columns else []

st.markdown("""
<div class="header">
  <h1>🗺️ Validation Plans Cadastraux</h1>
  <p>Zoom sur le champ · Matching commune automatique · Correction manuelle</p>
</div>
""", unsafe_allow_html=True)

top_l, top_r = st.columns([1,2])
with top_l:
    if not pending:
        st.success("✅ Tout est validé")
        page_id = st.selectbox("Consulter", df[id_col].tolist())
    else:
        page_id = st.selectbox(f"Page ({len(pending)} restante(s))", pending)

row = df[df[id_col]==page_id].iloc[0]
page_num = int(row.get("Page",1))
champs = champs_for_page(page_num)

# champ actif (pour le zoom)
if "active_field" not in st.session_state:
    st.session_state.active_field = "commune"

# ─── Chargement image ─────────────────────────────────────────────
@st.cache_data
def load_img(base, pn, fichier_json=""):
    for p in [
        os.path.join(OUTPUTS_DIR, f"{base}_p{pn}_annote.jpg"),
        os.path.join(OUTPUTS_DIR, f"{base}_page_{pn}_annote.jpg"),
        os.path.join(OUTPUTS_DIR, f"{base}_annote.jpg"),
    ]:
        if os.path.exists(p): return Image.open(p).convert("RGB")
    # Depuis PDF
    pdf = fichier_json or os.path.join(INPUTS_DIR, base+".pdf")
    if os.path.exists(pdf):
        try:
            import fitz
            doc = fitz.open(pdf)
            pi = min(max(0,pn-1), doc.page_count-1)
            pix = doc[pi].get_pixmap(matrix=fitz.Matrix(150/72,150/72), alpha=False)
            img = Image.frombytes("RGB",[pix.width,pix.height], pix.samples)
            doc.close()
            return img
        except: pass
    return None

img = load_img(base_name, page_num, _json.get("fichier",""))

def crop_zone(image, zone, margin_frac=0.04):
    """Crop + marge autour de la zone fractionnelle."""
    if not image or not zone or len(zone)!=4: return None
    w,h = image.size
    x0,y0,x1,y1 = zone
    mg = int(min(w,h)*margin_frac)
    px0 = max(0, int(x0*w)-mg)
    py0 = max(0, int(y0*h)-mg)
    px1 = min(w, int(x1*w)+mg)
    py1 = min(h, int(y1*h)+mg)
    if px1<=px0 or py1<=py0: return None
    crop = image.crop((px0,py0,px1,py1))
    # Bordure rouge sur le crop
    draw = ImageDraw.Draw(crop)
    draw.rectangle([0,0,crop.width-1,crop.height-1], outline=(220,50,50), width=3)
    return crop

# ─── Layout principal ─────────────────────────────────────────────
col_zoom, col_form = st.columns([1, 1], gap="medium")

# ══ GAUCHE : zoom sur le champ actif ══════════════════════════════
with col_zoom:
    af = st.session_state.active_field
    zone = jzone(champs, af)
    lbl_af = CHAMPS_LABELS.get(af, af)
    val_af = jval(champs, af)

    if img and zone:
        cropped = crop_zone(img, zone)
        if cropped:
            st.markdown(f'<div class="zoom-caption">🔍 {lbl_af}</div>', unsafe_allow_html=True)
            st.image(cropped, use_container_width=True)
            if val_af:
                st.info(f"**Valeur extraite :** {val_af}")
        else:
            st.markdown(f'<div class="zoom-caption">🔍 {lbl_af} — zone introuvable</div>',
                        unsafe_allow_html=True)
            if img: st.image(img, use_container_width=True)
    elif img:
        st.markdown(f'<div class="zoom-caption">🔍 {lbl_af} — aucune zone localisée</div>',
                    unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    else:
        st.info("Image non disponible.")

    # Bouton pleine page
    if img:
        with st.expander("📄 Voir la page complète"):
            st.image(img, use_container_width=True)

# ══ DROITE : liste des champs + formulaire ════════════════════════
with col_form:
    ordered_fields = CHAMPS_PAR_TYPE.get(plan_type, CHAMPS_PAR_TYPE["GENERIC"])

    def get_val(jf):
        csv_col = next((c for c,j in CSV_TO_JSON.items() if j==jf), None)
        if csv_col and csv_col in df.columns:
            v = row.get(csv_col,"")
            if not pd.isna(v) and str(v).strip() not in ["","nan","None","Inconnu"]:
                return str(v).strip()
        return jval(champs, jf)

    # ── Résumé cliquable ─────────────────────────────────────────
    st.markdown("#### Champs détectés")
    st.caption("Cliquez sur un champ pour zoomer à gauche")

    for jf in ordered_fields:
        val = get_val(jf)
        lbl = CHAMPS_LABELS.get(jf, jf)
        has_zone = bool(jzone(champs, jf))
        is_active = (jf == st.session_state.active_field)
        css_class = "field-row active" if is_active else "field-row"
        val_class = "" if val else "empty"
        badge = f'<span class="badge-ok">✓</span>' if val else f'<span class="badge-?">?</span>'
        icon = "📍" if has_zone else "  "

        # Chaque ligne = un bouton invisible qui active le zoom
        if st.button(
            f"{icon} {lbl}  |  {val[:55] if val else '—'}",
            key=f"sel_{jf}",
            use_container_width=True,
            help=f"Zoomer sur {lbl}",
        ):
            st.session_state.active_field = jf
            st.rerun()

    st.divider()

    # ── Formulaire de correction ──────────────────────────────────
    st.markdown("#### Correction")
    with st.form("form_val"):
        edited = {}
        for jf in ordered_fields:
            csv_col = next((c for c,j in CSV_TO_JSON.items() if j==jf), None)
            if not csv_col: continue
            if csv_col not in df.columns and jf not in champs: continue

            val = get_val(jf)
            lbl = CHAMPS_LABELS.get(jf, jf)

            # Matching commune
            hint = ""
            if jf == "commune" and val:
                m_nom, m_score, m_code = match_commune(val)
                if m_nom != val and m_score >= 55:
                    hint = f"→ Base : **{m_nom}** ({m_score}%)" + (f" `{m_code}`" if m_code else "")
                    val = m_nom

            if jf in MULTI:
                edited[csv_col] = st.text_area(lbl, value=val, height=55, key=f"inp_{jf}")
            else:
                edited[csv_col] = st.text_input(lbl, value=val, key=f"inp_{jf}")

            if hint:
                st.markdown(f'<div class="match-hint">{hint}</div>', unsafe_allow_html=True)

        st.markdown("")
        c1, c2 = st.columns(2)
        with c1:
            ok = st.form_submit_button("✅ Valider", type="primary", use_container_width=True)
        with c2:
            vide = st.form_submit_button("⬜ Marquer vide", use_container_width=True)

        if ok or vide:
            idx = df[df[id_col]==page_id].index[0]
            df.at[idx,"Confirmation_Status"] = "Ignoré (Vide)" if vide else "Validé par l'humain"
            if ok:
                for csv_col, new_val in edited.items():
                    if csv_col not in df.columns: df[csv_col] = ""
                    df.at[idx, csv_col] = new_val
            df.to_csv(fichier_choisi, sep=";", index=False, encoding="utf-8-sig")
            st.cache_data.clear()
            st.rerun()

# ─── Tableau récap ────────────────────────────────────────────────
st.divider()
with st.expander("📊 Tableau récapitulatif"):
    cols = [id_col,"Confirmation_Status"]
    for c in ["Commune","N_Ordre","Section","Date","Type_Plan","Geometre"]:
        if c in df.columns: cols.append(c)
    cols = [c for c in cols if c in df.columns]
    filtre = st.multiselect("Filtrer statut", df["Confirmation_Status"].dropna().unique().tolist())
    dv = df[df["Confirmation_Status"].isin(filtre)] if filtre else df
    def cr(r):
        s = str(r.get("Confirmation_Status",""))
        if "Validé" in s: return ["background:#f0faf3"]*len(r)
        if "Ignoré" in s: return ["background:#f5f5f5"]*len(r)
        return [""]*len(r)
    try: st.dataframe(dv[cols].style.apply(cr, axis=1), use_container_width=True, height=280)
    except: st.dataframe(dv[cols], use_container_width=True, height=280)
