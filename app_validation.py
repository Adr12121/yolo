import streamlit as st
import pandas as pd
import os, glob, json, re, unicodedata
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import plotly.express as px  # O5: import déplacé ici (was in body causing re-import each rerun)

st.set_page_config(page_title="Géofoncier Pro | Validation", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, div, span, applet, object, iframe, h1, h2, h3, h4, h5, h6, p, blockquote, pre, a, abbr, acronym, address, big, cite, code, del, dfn, em, img, ins, kbd, q, s, samp, small, strike, strong, sub, sup, tt, var, b, u, i, center, dl, dt, dd, ol, ul, li, fieldset, form, label, legend, table, caption, tbody, tfoot, getad, tr, th, td, article, aside, canvas, details, embed, figure, figcaption, footer, header, hgroup, menu, nav, output, ruby, section, summary, time, mark, audio, video {
    font-family: 'Outfit', sans-serif;
}
.material-icons, .material-symbols-rounded, [class*="icon"], [class*="symbol"] {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}
.header { 
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    color: #f8fafc; 
    padding: 2rem 2.5rem; 
    border-radius: 16px; 
    margin-bottom: 2rem; 
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); 
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}
.header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; width: 100%; height: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
}
.header h1 { 
    margin: 0; 
    font-size: 2.2rem; 
    font-weight: 700; 
    letter-spacing: -0.02em;
    background: linear-gradient(to right, #ffffff, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header p { 
    margin: 0.5rem 0 0; 
    font-size: 1rem; 
    color: #94a3b8; 
    font-weight: 400;
    letter-spacing: 0.01em;
}
.match-hint { font-size: 0.8rem; margin-top: 0.3rem; font-weight: 500; transition: all 0.2s ease;}
.hint-ok { color: #059669; background: #d1fae5; padding: 2px 8px; border-radius: 12px; display: inline-block;}
.hint-warn { color: #d97706; background: #fef3c7; padding: 2px 8px; border-radius: 12px; display: inline-block;}
.hint-err { color: #dc2626; background: #fee2e2; padding: 2px 8px; border-radius: 12px; display: inline-block;}

div[data-testid="stSidebar"] { 
    min-width: 320px !important; 
    background-color: #f8fafc; 
    border-right: 1px solid #e2e8f0; 
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    border-radius: 8px; 
    border: 1px solid #e2e8f0;
    background-color: #f8fafc;
    transition: all 0.3s ease;
    padding: 0.5rem;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { 
    border-color: #6366f1; 
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2); 
    background-color: #ffffff;
}
.field-label { 
    font-size: 0.9rem; 
    font-weight: 600; 
    color: #475569; 
    margin-bottom: 6px; 
    text-transform: uppercase; 
    letter-spacing: 0.05em;
}
.form-container { 
    background: #ffffff; 
    padding: 2rem; 
    border-radius: 16px; 
    border: 1px solid #e2e8f0; 
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05), 0 4px 6px -2px rgba(0,0,0,0.025);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.form-container:hover {
    box-shadow: 0 20px 25px -5px rgba(0,0,0,0.05), 0 10px 10px -5px rgba(0,0,0,0.02);
}
.image-container { 
    background: #f1f5f9; 
    padding: 8px; 
    border-radius: 12px; 
    border: 1px solid #cbd5e1; 
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.05);
}
/* Smooth hover states for buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
</style>
""", unsafe_allow_html=True)

OUTPUTS_DIR = "outputs"
INPUTS_DIR  = "inputs"

# ─── Fonctions de Formatage Géofoncier ───────────────────────────
def format_section(text):
    if not text: return ""
    text = str(text).strip().upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    if len(text) > 2: text = text[:2]
    return text

def format_date(text):
    if not text: return ""
    text = str(text).strip()
    if re.match(r'^\d{8}$', text):
        return f"{text[:2]}/{text[2:4]}/{text[4:]}"
    if re.match(r'^\d{6}$', text):
        a = int(text[4:])
        p = "20" if a < 50 else "19"
        return f"{text[:2]}/{text[2:4]}/{p}{text[4:]}"
    m = re.search(r'(\d{1,2})[^\d](\d{1,2})[^\d](\d{4})', text)
    if m:
        j, ms, a = m.groups()
        return f"{int(j):02d}/{int(ms):02d}/{a}"
    return text

def format_echelle(text):
    if not text: return ""
    text = str(text).strip().lower()
    m = re.search(r'(?:1\s*[/:\\]\s*|\b1\s+)?(\d{2,5})', text)
    if m: return f"1/{m.group(1)}"
    return text

def clean_parcelles(text):
    if not text: return ""
    text = str(text).replace('\n', ' ').replace(' à ', '-').replace(' a ', '-')
    text = re.sub(r'[;:/|_+]+', ',', text)
    text = re.sub(r'\s+(et|ou|puis)\s+', ',', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=\d)\s+(?=\d)', ', ', text)
    text = re.sub(r'(?<=\d)\s+(?=[A-Za-z])', ', ', text)
    text = re.sub(r'(?<=[A-Za-z])\s+(?=[A-Za-z])', ', ', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s*,\s*', ', ', text)
    return text.strip(', ')

# ─── Base communes ───────────────────────────────────────────────
@st.cache_resource  # I6: cache_resource = chargé une seule fois au démarrage (pas vidé par cache_data.clear)
def load_commune_db():
    db = []
    for fname in ["ardeche.json", "communes_france.json"]:
        if os.path.exists(fname):
            try:
                for e in json.load(open(fname, encoding="utf-8")):
                    n = e.get("nom","").strip()
                    if n: db.append({"officiel": n, "code": e.get("code","")})
            except Exception as e_db:  # O6: exception visible
                st.warning(f"[CommuneDB] Erreur lecture {fname}: {e_db}")
    return db

def _nc(t):
    nfkd = unicodedata.normalize("NFKD", str(t))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"[-''`]"," ",s)
    s = re.sub(r"[^A-Z0-9 ]"," ", s.upper()).strip()
    s = re.sub(r'\bST\b', 'SAINT', s)
    s = re.sub(r'\bSTE\b', 'SAINTE', s)
    return s

@st.cache_data
def match_commune(text):
    if not text: return text, 0, ""
    
    db = load_commune_db()
    if not db: return text, 0, ""

    import re
    text_clean = text.strip()
    if re.match(r'^(?:07)?\d{3}$', text_clean):
        code_insee = text_clean if len(text_clean) == 5 else f"07{text_clean}"
        for c in db:
            if str(c.get("code", "")) == code_insee:
                return c["officiel"], 100.0, c.get("code","")

    try:
        from rapidfuzz import process as rfp, fuzz
    except ImportError: return text, 0, ""
    db = load_commune_db()
    if not db: return text, 0, ""
    noms = [_nc(e["officiel"]) for e in db]
    matches = rfp.extract(_nc(text), noms, scorer=fuzz.token_set_ratio, score_cutoff=80)
    if matches:
        # Tie-breaker: closest length
        best_match = min(matches, key=lambda m: abs(len(m[0]) - len(_nc(text))))
        e = db[best_match[2]]
        return e["officiel"], best_match[1], e.get("code","")
    return text, 0, ""

# ─── Chargement Données ─────────────────────────────────────────
all_csv = sorted(glob.glob(os.path.join(OUTPUTS_DIR,"*_plan_resultats.csv")))
if not all_csv:
    all_csv = sorted(glob.glob(os.path.join(OUTPUTS_DIR,"*_resultats.csv")))
if not all_csv:
    st.warning("Aucun résultat dans 'outputs/'. Lancez d'abord l'extraction OCR.")
    st.stop()

# ── Détection du changement de fichier ───────────────────────────────────
# On stocke le fichier actuellement chargé dans session_state.
# Si la sélection change, on efface TOUT avant de recharger.
if "fichier_actif" not in st.session_state:
    st.session_state["fichier_actif"] = None

with st.sidebar:
    st.markdown("### 📄 Fichier de travail")
    fichier_choisi = st.selectbox(
        "Sélection", all_csv,
        format_func=lambda p: os.path.basename(p).replace("_plan_resultats.csv","").replace("_resultats.csv",""),
        label_visibility="collapsed",
    )

# Si le fichier a changé (nouveau fichier) ou si sa date de modification a changé (réexécution en arrière-plan) → purge totale
current_mtime = os.path.getmtime(fichier_choisi) if os.path.exists(fichier_choisi) else 0
if st.session_state.get("fichier_actif") != fichier_choisi or st.session_state.get("fichier_actif_mtime") != current_mtime:
    st.cache_data.clear()
    # Supprimer tous les JSONs en cache (clés _json_*)
    for k in list(st.session_state.keys()):
        if k.startswith("_json_"):
            del st.session_state[k]
    st.session_state["fichier_actif"] = fichier_choisi
    st.session_state["fichier_actif_mtime"] = current_mtime


def load_csv(p):
    try: return pd.read_csv(p, sep=";", encoding="utf-8-sig")
    except: return pd.read_csv(p, sep=";", encoding="utf-8")

df = load_csv(fichier_choisi)
base_name = os.path.basename(fichier_choisi).replace("_plan_resultats.csv","").replace("_resultats.csv","")

if "Confirmation_Status" not in df.columns: df["Confirmation_Status"] = "À valider"
if "Code_INSEE" not in df.columns: df["Code_INSEE"] = ""

DONE = ["Validé par l'humain","Corrigé automatiquement","Ignoré (Vide)"]
id_col = "ID" if "ID" in df.columns else "ID_Ligne"

# Couleurs pour le rendu visuel
COLORS = {
    "commune": (16, 185, 129),       # Emerald
    "section": (245, 158, 11),       # Amber
    "parcelles": (239, 68, 68),      # Red
    "date": (59, 130, 246),          # Blue
    "geometre": (139, 92, 246),      # Purple
    "n_ordre": (14, 165, 233),       # Sky
    "n_dossier": (14, 165, 233),
    "anciennes_parcelles": (16, 185, 129), # Green
    "nouvelles_parcelles": (239, 68, 68),  # Red
    "indication": (100, 116, 139),
    "feuille": (245, 158, 11),
    "echelle": (100, 116, 139),
    "signataires": (139, 92, 246)
}

def get_color_hex(field):
    c = COLORS.get(field, (0,0,0))
    return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

with st.sidebar:
    st.markdown("---")
    n_val = len(df[df["Confirmation_Status"].isin(DONE)])
    st.progress(n_val / max(len(df),1))
    st.caption(f"**{n_val} / {len(df)}** documents validés")
    
    # Bouton de nettoyage Outputs (Point 4)
    if st.button("Nettoyer les images déjà validées", help="Supprime les images annotées des documents déjà validés pour libérer de l'espace disque."):
        deleted_count = 0
        for idx, r in df.iterrows():
            if r["Confirmation_Status"] == "Validé par l'humain":
                base_n = os.path.basename(fichier_choisi).replace("_plan_resultats.csv","").replace("_resultats.csv","")
                for img_path in glob.glob(os.path.join(OUTPUTS_DIR, f"{base_n}*_annote.jpg")):
                    try:
                        os.remove(img_path)
                        deleted_count += 1
                    except: pass
        st.success(f"Opération terminée. {deleted_count} fichiers supprimés.")

    plan_type = "GENERIC"
    for c in ["Type_Plan","Type_Document"]:
        if c in df.columns:
            v = df[c].dropna().unique()
            if len(v): plan_type = str(v[0]); break
    
    st.markdown(f"**Format détecté :** `{plan_type}`")
    pending = df[~df["Confirmation_Status"].isin(DONE)][id_col].tolist() if id_col in df.columns else []
    
    if not pending:
        st.success("Tout est validé.")
        page_id = st.selectbox("Revoir un document", df[id_col].tolist())
    else:
        page_id = st.selectbox(f"À traiter ({len(pending)} restant(s))", pending)

# ── Chargement JSON (invalidé à chaque changement de fichier via session_state) ──
_json_cache_key = f"_json_{base_name}"
if _json_cache_key not in st.session_state:
    _json_loaded = {}
    # Essayer tous les suffixes connus + fallback générique
    _json_suffixes = [
        f"_plan_{plan_type}.json",
        "_plan_PVa.json", "_plan_PLa.json", "_plan_DMPC.json",
        "_plan_GENERIC.json", "_plan_PLAN.json", "_plan_moderne.json",
    ]
    for suf in _json_suffixes:
        jp = os.path.join(OUTPUTS_DIR, base_name + suf)
        if os.path.exists(jp):
            try:
                _json_loaded = json.load(open(jp, encoding="utf-8"))
                break
            except Exception:
                pass
    # Fallback : chercher n'importe quel JSON commençant par base_name
    if not _json_loaded:
        for jp in glob.glob(os.path.join(OUTPUTS_DIR, base_name + "*.json")):
            try:
                _json_loaded = json.load(open(jp, encoding="utf-8"))
                if _json_loaded:
                    break
            except Exception:
                pass
    st.session_state[_json_cache_key] = _json_loaded

_json = st.session_state[_json_cache_key]

# On utilise directement la racine du JSON car elle contient les champs consolidés du document
champs_consolidated = _json if _json else {}

row = df[df[id_col]==page_id].iloc[0]
base_page_num = int(row.get("Page",1))

@st.cache_data
def load_img(base, pn, fichier_json=""):
    pdf = fichier_json or os.path.join(INPUTS_DIR, base+".pdf")
    if os.path.exists(pdf):
        try:
            import fitz
            doc = fitz.open(pdf)
            pi = min(max(0,pn-1), doc.page_count-1)
            # Rendu haute résolution pour la visionneuse
            pix = doc[pi].get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            img = Image.frombytes("RGB",[pix.width,pix.height], pix.samples)
            doc.close()
            return img
        except: pass
    # Fallback sur les images jpg
    for p in [os.path.join(OUTPUTS_DIR, f"{base}_p{pn}_annote.jpg"), os.path.join(OUTPUTS_DIR, f"{base}_page_{pn}_annote.jpg")]:
        if os.path.exists(p): return Image.open(p).convert("RGB")
    return None

@st.cache_data
def get_crop(base, pn, zone, fichier_json=""):
    if len(zone) != 4 or sum(zone) == 0: return None
    img = load_img(base, pn, fichier_json)
    if not img: return None
    w, h = img.size
    x0, y0, x1, y1 = zone[0]*w, zone[1]*h, zone[2]*w, zone[3]*h
    px, py = w * 0.05, h * 0.03
    return img.crop((max(0, x0-px), max(0, y0-py), min(w, x1+px), min(h, y1+py)))

def jval(champs_dict, field):
    v = champs_dict.get(field,{})
    if isinstance(v,dict):
        val = v.get("valeur","")
        return ", ".join(str(x) for x in val) if isinstance(val,list) else str(val)
    return ""

CHAMPS_LABELS = {
    "commune": "Commune", "n_ordre": "N° DA", "n_dossier": "N° Dossier",
    "section": "Section", "feuille": "Feuille", "date": "Date",
    "echelle": "Échelle", "geometre": "Géomètre", "signataires": "Signataires",
    "anciennes_parcelles": "Anciennes parcelles", "nouvelles_parcelles": "Nouvelles parcelles",
    "parcelles": "Parcelles", "indication": "Objet / Indication",
    "nature_acte_geofoncier": "Code Geofoncier"
}

CSV_TO_JSON = {
    "Commune":"commune","N_Ordre":"n_ordre","N_Dossier":"n_dossier",
    "Section":"section","Feuille":"feuille","Date":"date",
    "Echelle":"echelle","Geometre":"geometre","Signataires":"signataires",
    "Anciennes_Parcelles":"anciennes_parcelles",
    "Nouvelles_Parcelles":"nouvelles_parcelles",
    "Parcelles":"parcelles","Indication":"indication",
    "Nature_Acte_Geofoncier":"nature_acte_geofoncier"
}
MULTI = {"parcelles","signataires","anciennes_parcelles","nouvelles_parcelles","indication"}

st.markdown("""
<div class="header">
  <h1>Validation des archives cadastrales</h1>
  <p>Interface de contrôle de la qualité des extractions.</p>
</div>
""", unsafe_allow_html=True)

# ─── Bandeau de Cohérence ─────────────────────────────────────────────────
_coherence = champs_consolidated.get("_coherence", {})
if _coherence:
    _coh_status = _coherence.get("status", "")
    _coh_score  = _coherence.get("score", 1.0)
    _coh_counts = _coherence.get("counts", {})
    _coh_summary = _coherence.get("valeur", "")

    if _coh_status == "CONFORME":
        _bg, _border, _icon = "#d1fae5", "#10b981", "Statut :"
    elif _coh_status == "ALERTE":
        _bg, _border, _icon = "#fef3c7", "#f59e0b", "Statut :"
    else:
        _bg, _border, _icon = "#fee2e2", "#ef4444", "Statut :"

    st.markdown(
        f'<div style="background:{_bg};border-left:5px solid {_border};'
        f'padding:12px 18px;border-radius:8px;margin-bottom:1rem;">'
        f'<strong>{_icon} {_coh_status}</strong> '
        f'<span style="font-size:0.85rem;margin-left:10px;">Score global : {int(_coh_score*100)}%</span>'
        f'<br><span style="font-size:0.8rem;color:#555;">{_coh_summary}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Détail des alertes/erreurs groupées par champ
    all_issues = []
    for field, fdata in champs_consolidated.items():
        if isinstance(fdata, dict) and fdata.get("coherence_issues"):
            for iss in fdata["coherence_issues"]:
                all_issues.append(iss)

    if all_issues:
        with st.expander(f"Détail des incohérences détectées ({len(all_issues)} signal(aux))", expanded=(_coh_status == "REJET")):
            for iss in all_issues:
                lvl = iss.get("level", "INFO")
                src = iss.get("source", "")
                msg = iss.get("message", "")
                rule = iss.get("rule", "")
                icon_iss = "[ERREUR]" if lvl == "ERREUR" else ("[ALERTE]" if lvl == "ALERTE" else "[INFO]")
                src_label = {"rule_engine": "Règles métier", "pydantic": "Schéma", "ollama_llm": "Analyse texte", "knowledge_graph": "Base connaissance"}.get(src, src)
                st.markdown(
                    f'**{icon_iss} [{src_label}]** `{rule}` — {msg}',
                    unsafe_allow_html=False
                )

# ─── Interface Split-Screen ──────────────────────────────────────────────
col_form, col_view = st.columns([1, 2], gap="large")

# 1. Construction de l'image annotée (Droite)
with col_view:
    c_title, c_btn = st.columns([3, 1])
    with c_title:
        st.markdown('<div class="field-label">Aperçu du Document (Zoom Interactif)</div>', unsafe_allow_html=True)
    with c_btn:
        if "zoom_key" not in st.session_state:
            st.session_state.zoom_key = 0
        if st.button("Réinitialiser le Zoom", use_container_width=True):
            st.session_state.zoom_key += 1
            st.rerun()
    
    # Sélecteur de page si multi-pages
    all_pages = [p.get("page") for p in _json.get("pages", [])] if isinstance(_json, dict) else []
    if not all_pages: all_pages = [base_page_num]
    
    if len(all_pages) > 1:
        page_vue = st.radio("Page affichée :", options=all_pages, horizontal=True)
    else:
        page_vue = all_pages[0]
        
    img_base = load_img(base_name, page_vue, _json.get("fichier","") if isinstance(_json, dict) else "")
    
    if img_base:
        # Dessiner toutes les zones sur l'image
        draw = ImageDraw.Draw(img_base, "RGBA")
        w, h = img_base.size
        
        for jf, v in champs_consolidated.items():
            if isinstance(v, dict) and "zone" in v and v.get("page", 1) == page_vue:
                z = v["zone"]
                if len(z) == 4:
                    x0, y0, x1, y1 = z[0]*w, z[1]*h, z[2]*w, z[3]*h
                    color = COLORS.get(jf, (100,100,100))
                    # Remplissage semi-transparent
                    draw.rectangle([x0, y0, x1, y1], fill=(color[0], color[1], color[2], 40))
                    # Bordure
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
                    
        # O5: import déjà fait en tête de fichier
        fig = px.imshow(img_base)
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0, pad=0),
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            hovermode=False,
            dragmode='pan',
            autosize=True,
            # Hauteur ajustée (800) pour correspondre plus fidèlement à la largeur de la colonne, éliminant ainsi le grand espace blanc
            height=int(800 * (h / w)),  
        )
        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False}, key=f"img_viewer_{page_vue}_{st.session_state.zoom_key}")
        st.caption("Astuce : Utilisez la molette de la souris pour zoomer sur l'image.")
    else:
        st.info("Aperçu visuel non disponible pour cette page.")

# 2. Formulaire (Gauche)
LEARNABLE_FIELDS = ["Geometre", "Commune", "Section", "Indication", "Nature_Acte_Geofoncier", "Signataires"]

def _learn_correction(csv_col, old_val, new_val):
    if csv_col not in LEARNABLE_FIELDS: return
    old_str = str(old_val).strip()
    new_str = str(new_val).strip()
    if not old_str or not new_str or len(old_str) < 3: return
    
    import re
    old_norm = re.sub(r'[^a-z0-9]', '', old_str.lower())
    if len(old_norm) < 3: return
    
    db_path = os.path.join(OUTPUTS_DIR, "learned_corrections.json")
    db = {}
    if os.path.exists(db_path):
        try: db = json.load(open(db_path, encoding="utf-8"))
        except: pass
        
    jf = CSV_TO_JSON.get(csv_col, csv_col.lower())
    if jf not in db: db[jf] = {}
    
    new_norm = re.sub(r'[^a-z0-9]', '', new_str.lower())
    if old_norm != new_norm:
        db[jf][old_norm] = new_str
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)

def on_field_change(file_path, doc_id, col_name, wkey):
    new_val = st.session_state.get(wkey, "")
    try:
        temp_df = pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
    except:
        temp_df = pd.read_csv(file_path, sep=";", encoding="utf-8")
        
    if col_name not in temp_df.columns: temp_df[col_name] = ""
    if not pd.api.types.is_object_dtype(temp_df[col_name]):
        temp_df[col_name] = temp_df[col_name].astype(object)
        
    id_col_temp = "ID" if "ID" in temp_df.columns else "ID_Ligne"
    idxs = temp_df[temp_df[id_col_temp] == doc_id].index
    if len(idxs) > 0:
        idx = idxs[0]
        old_val = temp_df.at[idx, col_name]
        
        if pd.notna(old_val) and str(old_val).strip() != str(new_val).strip() and str(old_val).strip() != "":
            _learn_correction(col_name, old_val, new_val)
            
        final_val = new_val
        if col_name == "Section" and new_val:
            final_val = format_section(new_val)
        elif col_name == "Date" and new_val:
            final_val = format_date(new_val)
        elif col_name == "Echelle" and new_val:
            final_val = format_echelle(new_val)
        elif col_name in ["Parcelles", "Anciennes_Parcelles", "Nouvelles_Parcelles"] and new_val:
            final_val = clean_parcelles(new_val)
        elif col_name == "Commune" and new_val:
            m_nom, m_score, m_code = match_commune(new_val)
            if m_score >= 55:
                final_val = m_nom
                if "Code_INSEE" not in temp_df.columns:
                    temp_df["Code_INSEE"] = ""
                temp_df.at[idx, "Code_INSEE"] = m_code
                
        temp_df.at[idx, col_name] = final_val
        st.session_state[wkey] = final_val
        
        # Mettre à jour les champs Geofoncier (Répertoire)
        if col_name == "Commune":
            if f"lu_commune_{doc_id}" in st.session_state: st.session_state[f"lu_commune_{doc_id}"] = final_val
            if f"rc_commune_{doc_id}" in st.session_state: st.session_state[f"rc_commune_{doc_id}"] = final_val
        elif col_name == "Section":
            if f"lu_section_{doc_id}" in st.session_state: st.session_state[f"lu_section_{doc_id}"] = final_val
            if f"rc_section_{doc_id}" in st.session_state: st.session_state[f"rc_section_{doc_id}"] = final_val
        elif col_name in ["Parcelles", "Anciennes_Parcelles", "Nouvelles_Parcelles"]:
            _p_anc = str(temp_df.at[idx, "Anciennes_Parcelles"] if pd.notna(temp_df.at[idx, "Anciennes_Parcelles"]) else "").strip()
            _p_nou = str(temp_df.at[idx, "Nouvelles_Parcelles"] if pd.notna(temp_df.at[idx, "Nouvelles_Parcelles"]) else "").strip()
            _p_all = str(temp_df.at[idx, "Parcelles"] if pd.notna(temp_df.at[idx, "Parcelles"]) else "").strip()
            _parcs_concat = f"{_p_anc} {_p_nou} {_p_all}"
            try:
                from repertoire_lookup import clean_parcelles_for_lookup
                _p_propres = clean_parcelles_for_lookup(_parcs_concat)
                if f"lu_parcelle_{doc_id}" in st.session_state: st.session_state[f"lu_parcelle_{doc_id}"] = ", ".join(str(p) for p in _p_propres[:5]) if _p_propres else ""
            except: pass
            try:
                from repertoire_racat_ceyte import clean_parcelles_for_lookup_rc
                _p_propres_rc = clean_parcelles_for_lookup_rc(_parcs_concat)
                if f"rc_parcelles_{doc_id}" in st.session_state: st.session_state[f"rc_parcelles_{doc_id}"] = ", ".join(str(p) for p in _p_propres_rc[:5]) if _p_propres_rc else ""
            except: pass
        elif col_name == "Date":
            try:
                from repertoire_lookup import extract_year_from_date
                _annee = extract_year_from_date(final_val)
                if f"lu_annee_{doc_id}" in st.session_state: st.session_state[f"lu_annee_{doc_id}"] = str(_annee) if _annee else ""
            except: pass
            import re
            _m_date = re.search(r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})", str(final_val))
            if _m_date:
                _yr = int(_m_date.group(3))
                _rc_annee = 2000 + _yr if _yr <= 30 else (1900 + _yr if _yr < 100 else _yr)
                if f"rc_annee_{doc_id}" in st.session_state: st.session_state[f"rc_annee_{doc_id}"] = str(_rc_annee)
            
        temp_df.to_csv(file_path, sep=";", index=False, encoding="utf-8-sig")

with col_form:
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown('<div class="field-label" style="font-size:1.1rem; margin-bottom:1rem;">Saisie & Correction</div>', unsafe_allow_html=True)
    
    with st.container():
        edited = {}
        
        # Chargement de la base d'apprentissage pour correction en temps réel
        db_corrections = {}
        db_path = os.path.join(OUTPUTS_DIR, "learned_corrections.json")
        if os.path.exists(db_path):
            try: db_corrections = json.load(open(db_path, encoding="utf-8"))
            except: pass
            
        DMPC_HINTS = {
            "Commune": "Généralement en haut au centre ou dans le cartouche d'en-tête.",
            "Section": "En haut ou dans le tableau de filiation, souvent 1 ou 2 lettres (ex: AB).",
            "Feuille": "Souvent à côté de la section ou en haut à gauche.",
            "Parcelles": "Sur le plan lui-même.",
            "Anciennes_Parcelles": "Dans la colonne de gauche du tableau de modification du parcellaire.",
            "Nouvelles_Parcelles": "Dans la colonne de droite du tableau de modification du parcellaire.",
            "N_Ordre": "Numéro de DA (ex: 1234A ou simplement 123), souvent en haut à droite ou dans le cartouche.",
            "N_Dossier": "Référence interne du cabinet",
            "Date": "Dans le cartouche en bas à droite, ou près des signatures.",
            "Echelle": "Au-dessus du croquis ou dans le cartouche (ex: 1/500).",
            "Geometre" : "Souvent avec un logo ou en bas de page.",
            "Signataires": "Tout en bas du document.",
            "Indication": "L'information non raturée en bas du document.",
            "Nature_Acte_Geofoncier": "Déduit de l'objet du plan (DMPC, Document d'arpentage...)."
        }
        
        # On regroupe pour aérer l'interface
        ordered_csv_cols = [c for c in CSV_TO_JSON.keys() if c in df.columns]
        
        GROUPS = {
            "Localisation": ["Commune", "Section", "Feuille", "Parcelles"],
            "Références & Acte": ["N_Ordre", "N_Dossier", "Date", "Echelle", "Indication", "Nature_Acte_Geofoncier"],
            "Intervenants": ["Geometre", "Signataires"],
            "Parcelles (DMPC)": ["Anciennes_Parcelles", "Nouvelles_Parcelles"]
        }
        
        for group_name, cols_in_group in GROUPS.items():
            valid_cols = [c for c in cols_in_group if c in ordered_csv_cols]
            if not valid_cols: continue
            
            st.markdown(f"<h4 style='color:#1e293b; margin-top:1.5rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3rem;'>{group_name}</h4>", unsafe_allow_html=True)
            
            for csv_col in valid_cols:
                    jf = CSV_TO_JSON[csv_col]
                    val = row.get(csv_col,"")
                    if pd.isna(val) or str(val).strip() in ["nan","None"]: 
                        val = jval(champs_consolidated, jf)
                    else:
                        val = str(val).strip()
                        
                    # === APPRENTISSAGE EN DIRECT ===
                    auto_corrected = False
                    if db_corrections and jf in db_corrections and val:
                        import re
                        val_norm = re.sub(r'[^a-z0-9]', '', str(val).lower())
                        if val_norm in db_corrections[jf]:
                            val = db_corrections[jf][val_norm]
                            auto_corrected = True
                        
                    lbl = CHAMPS_LABELS.get(jf, jf)
                    color_hex = get_color_hex(jf)
                    
                    # ── Badge de vérification ──
                    field_meta = champs_consolidated.get(jf, {}) if isinstance(champs_consolidated.get(jf), dict) else {}
                    v_status = field_meta.get("verification_status", "")
                    v_conf   = field_meta.get("confidence", None)
                    v_notes  = field_meta.get("verification_notes", [])
                    
                    if auto_corrected:
                        badge = f'<span style="background:#e0e7ff;color:#3730a3;font-size:0.8rem;padding:3px 8px;border-radius:10px;font-weight:700;margin-left:8px;">Auto-corrigé</span>'
                    elif v_status == "OK":
                        badge = f'<span style="background:#d1fae5;color:#065f46;font-size:0.8rem;padding:3px 8px;border-radius:10px;font-weight:700;margin-left:8px;">OK {int(v_conf*100) if v_conf else ""}%</span>'
                    elif v_status == "INCERTAIN":
                        badge = f'<span style="background:#fef3c7;color:#92400e;font-size:0.8rem;padding:3px 8px;border-radius:10px;font-weight:700;margin-left:8px;">À vérifier {int(v_conf*100) if v_conf else ""}%</span>'
                    elif v_status == "SUSPECT":
                        badge = f'<span style="background:#fee2e2;color:#991b1b;font-size:0.8rem;padding:3px 8px;border-radius:10px;font-weight:700;margin-left:8px;">Rejeté</span>'
                    else:
                        badge = ""
                    
                    dmpc_hint = DMPC_HINTS.get(csv_col, "")
                    hint_html = f'<div style="font-size:0.75rem; color:#64748b; margin-bottom:5px; margin-left:20px; font-style:italic;">{dmpc_hint}</div>' if dmpc_hint else ""
                    
                    st.markdown(
                        f'<div class="field-label" style="margin-top:10px;">'
                        f'<span style="display:inline-block;width:12px;height:12px;background-color:{color_hex};border-radius:50%;margin-right:8px;"></span>'
                        f'{lbl}{badge}'
                        f'</div>'
                        f'{hint_html}',
                        unsafe_allow_html=True
                    )

                    
                    hint = ""
                    hint_class = "hint-ok"

                    if v_notes and v_status in ("INCERTAIN", "SUSPECT"):
                        hint = f"Contrôle : {' | '.join(v_notes[:2])}"
                        hint_class = "hint-warn" if v_status == "INCERTAIN" else "hint-err"

                    if jf == "commune" and val:
                        m_nom, m_score, m_code = match_commune(val)
                        if m_score >= 55:
                            val = m_nom
                            edited["Code_INSEE"] = m_code
                            hint = f"INSEE: {m_code} ({m_score}%)"
                            hint_class = "hint-ok"
                        else:
                            hint = "Commune non reconnue"
                            hint_class = "hint-err"
                    elif jf == "section" and val:
                        val = format_section(val)
                        if not re.match(r'^[A-Z]{1,2}$', val):
                            hint = "Inhabituel (A, AB...)"
                            hint_class = "hint-warn"
                    elif jf == "date" and val:
                        n_val_d = format_date(val)
                        if n_val_d != val:
                            hint = "Formaté JJ/MM/AAAA"
                            val = n_val_d
                    elif jf == "echelle" and val:
                        n_val_e = format_echelle(val)
                        if n_val_e != val:
                            hint = "Échelle formatée"
                            val = n_val_e
                    elif jf in ("parcelles", "anciennes_parcelles", "nouvelles_parcelles") and val:
                        n_val_p = clean_parcelles(val)
                        if n_val_p != val:
                            hint = "Liste nettoyée"
                            val = n_val_p

                    # Mini-crop affiché au-dessus du champ (Point 5)
                    f_page = field_meta.get("page", 1)
                    f_zone = field_meta.get("zone", field_meta.get("bbox", []))
                    if f_zone and len(f_zone) == 4 and sum(f_zone) > 0:
                        crop = get_crop(base_name, f_page, f_zone, _json.get("fichier","") if isinstance(_json, dict) else "")
                        if crop:
                            st.image(crop, use_container_width=True)

                    # Saisie
                    _wkey = f"inp_{base_name}_{page_id}_{jf}"
                    _args = (fichier_choisi, page_id, csv_col, _wkey)
                    if jf == "geometre":
                        _opts = ["", "HARROIS", "BARRIAL", "RACAT", "CEYTE", "DUPUY", "SERRET", "ROBERT"]
                        _val_clean = str(val).strip().upper()
                        if _val_clean not in _opts and _val_clean:
                            _opts.append(_val_clean)
                        _idx_sel = _opts.index(_val_clean) if _val_clean in _opts else 0
                        edited[csv_col] = st.selectbox("Correction", options=_opts, index=_idx_sel, key=_wkey, label_visibility="collapsed", on_change=on_field_change, args=_args)
                    elif jf in MULTI:
                        edited[csv_col] = st.text_area("Correction", value=val, height=80, key=_wkey, label_visibility="collapsed", on_change=on_field_change, args=_args)
                    else:
                        edited[csv_col] = st.text_input("Correction", value=val, key=_wkey, label_visibility="collapsed", on_change=on_field_change, args=_args)

                    if hint:
                        st.markdown(f'<div class="match-hint {hint_class}">{hint}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="margin-bottom:0.5rem;"></div>', unsafe_allow_html=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            ok = st.button("Enregistrer & Suivant", type="primary", use_container_width=True)
        with c2:
            vide = st.button("Ignorer le document", use_container_width=True)

        if ok or vide:
            idx = df[df[id_col]==page_id].index[0]
            df.at[idx,"Confirmation_Status"] = "Ignoré (Vide)" if vide else "Validé par l'humain"
            if ok:
                for csv_col, new_val in edited.items():
                    if csv_col not in df.columns: df[csv_col] = ""
                    # Forcer le type de la colonne en objet pour accepter des chaînes vides ("") si elle a été lue comme float64
                    if not pd.api.types.is_object_dtype(df[csv_col]):
                        df[csv_col] = df[csv_col].astype(object)
                    old_val = df.at[idx, csv_col]
                    if pd.notna(old_val) and str(old_val).strip() != str(new_val).strip() and str(old_val).strip() != "":
                        _learn_correction(csv_col, old_val, new_val)
                    df.at[idx, csv_col] = new_val
            # I5 : Vérification commune avant sauvegarde
            commune_saisie = edited.get("Commune", "")
            if commune_saisie and len(commune_saisie.strip()) >= 3:
                _, m_score, _ = match_commune(commune_saisie)
                if m_score < 60:
                    st.warning(f"Commune '{commune_saisie}' non reconnue (score {m_score}%). Vérifiez l'orthographe avant de sauvegarder.")
            df.to_csv(fichier_choisi, sep=";", index=False, encoding="utf-8-sig")
            
            # Nettoyer les clés de session Geofoncier pour forcer la mise à jour si on reste sur la même vue
            _keys_to_del = [
                f"lu_commune_{page_id}", f"lu_section_{page_id}", f"lu_parcelle_{page_id}", f"lu_annee_{page_id}",
                f"rc_commune_{page_id}", f"rc_section_{page_id}", f"rc_parcelles_{page_id}", f"rc_annee_{page_id}",
                f"_lookup_result_{base_name}_{page_id}", f"_lookup_confirmed_{base_name}_{page_id}",
                f"_rc_lookup_result_{base_name}_{page_id}", f"_rc_lookup_confirmed_{base_name}_{page_id}"
            ]
            for _k in _keys_to_del:
                if _k in st.session_state:
                    del st.session_state[_k]
                    
            st.cache_data.clear()
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Section Export CSV (conservée) ─────────────────────────────────────────
st.markdown("---")
with st.expander("Export CSV normé (Géofoncier)"):
    valid_df = df[df["Confirmation_Status"] == "Validé par l'humain"]
    col1, col2 = st.columns([2,1])
    with col1:
        if len(valid_df) > 0:
            export_cols = ["Code_INSEE", "Commune", "Section", "Parcelles", "Anciennes_Parcelles", "Nouvelles_Parcelles", "Date", "N_Ordre", "Geometre", "Signataires", "Nature_Acte_Geofoncier"]
            available_cols = [c for c in export_cols if c in valid_df.columns]
            st.dataframe(valid_df[available_cols], use_container_width=True, hide_index=True)
            st.download_button(
                label=f"Télécharger le CSV normé ({len(valid_df)} dossiers)",
                data=valid_df[available_cols].to_csv(sep=";", index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"geofoncier_export_{base_name}.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.warning("Aucun document validé. Validez d'abord dans l'interface au-dessus.")
    with col2:
        st.markdown("""
        **Normes appliquées :**
        - Uniquement les documents "Validés"
        - `Code_INSEE` inclus
        - `Section` en majuscules (A, AB)
        - `Date` formatée JJ/MM/AAAA
        - `Parcelles` séparées par des virgules
        """)

# ─── Section Versement Géofoncier ────────────────────────────────────────────
_geometre_raw = str(row.get("Geometre", "")).strip().upper() if "Geometre" in row.index else ""
_geometre_val = _geometre_raw.split()[0] if _geometre_raw else ""

_GEOMETRES_REPERTOIRE = {"HARROIS", "BARRIAL"}
_GEOMETRES_REPERTOIRE_DUPUY = {"DUPUY"}        # Répertoire spécifique archives DUPUY
_GEOMETRES_API_DIRECT = {"SERRET", "LACOUR", "ROBERT"}

if _geometre_val in _GEOMETRES_REPERTOIRE:
    st.markdown("---")
    _geo_label = {
        "HARROIS": "Archives Harrois",
        "BARRIAL": "Archives Barrial",
        "SERRET":  "Archives Serret",
        "DUPUY":   "Archives Dupuy",
        "LACOUR":  "Archives Lacour",
        "ROBERT":  "Archives Robert",
        "RACAT":   "Archives Racat",
        "CEYTE":   "Archives Ceyte",
    }.get(_geometre_val, f"Archives {_geometre_val}")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);color:#f8fafc;
    padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;
    border-left:5px solid #3b82f6;">
    <h3 style="margin:0;font-size:1.3rem;font-weight:700;">
        Versement Géofoncier — {_geo_label}
    </h3>
    <p style="margin:0.4rem 0 0;color:#94a3b8;font-size:0.9rem;">
       Recherche de la référence dans le répertoire, puis versement sur Géofoncier.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Import du module de lookup ──────────────────────────────────────────
    try:
        from repertoire_lookup import (
            find_dossier, extract_year_from_date, clean_parcelles_for_lookup,
            map_op_code, GEOMETRES_REPERTOIRE
        )
        from geofoncier_api import (
            create_geofoncier_dossier, upload_document_to_dossier,
            verify_parcel_ign, get_insee_from_commune, format_date_iso,
            get_parcel_geometry, geocode_commune
        )
        import folium
        from streamlit_folium import st_folium
        _lookup_ok = True
    except ImportError as e:
        st.error(f"Erreur d'import des modules Géofoncier : {e}")
        _lookup_ok = False

    if _lookup_ok:

        # ══════════════════════════════════════════════════════════════════
        # Section 1 — Résolution du répertoire
        # ══════════════════════════════════════════════════════════════════
        st.markdown("### Étape 1 — Résolution de la référence dans le répertoire")

        # Pré-remplissage depuis le CSV validé
        _commune_pre   = str(row.get("Commune", "")).strip()
        _section_pre   = str(row.get("Section", "")).strip().upper()
        _date_pre      = str(row.get("Date", "")).strip()
        _annee_pre     = extract_year_from_date(_date_pre)

        # Parcelles : on cherche la première parcelle numérique dans Anciennes ou Nouvelles
        _parcs_raw_anc = str(row.get("Anciennes_Parcelles", "")).strip()
        _parcs_raw_nouv = str(row.get("Nouvelles_Parcelles", "")).strip()
        _parcs_raw_all = str(row.get("Parcelles", "")).strip()
        # On concatène tout pour maximiser les chances
        _parcs_concat = f"{_parcs_raw_anc} {_parcs_raw_nouv} {_parcs_raw_all}"
        _parcs_propres = clean_parcelles_for_lookup(_parcs_concat)
        _parcs_pre_str = ", ".join(str(p) for p in _parcs_propres[:5]) if _parcs_propres else ""

        col_lookup_a, col_lookup_b = st.columns([1, 1])
        with col_lookup_a:
            _lu_commune = st.text_input("Commune", value=_commune_pre, key=f"lu_commune_{page_id}")
            _lu_section = st.text_input(
                "Section cadastrale (corrigeable si OCR erronée)",
                value=_section_pre,
                key=f"lu_section_{page_id}",
                help="Corrigez ici si la section extraite par OCR est incorrecte (ex: AC → AL)"
            )
        with col_lookup_b:
            _lu_parcelle_str = st.text_input(
                "Parcelle(s) (numéros séparés par virgules)",
                value=_parcs_pre_str,
                key=f"lu_parcelle_{page_id}",
                help="Saisissez au moins un numéro de parcelle ancienne visible sur le plan"
            )
            _lu_annee_str = st.text_input(
                "Année (2 chiffres, ex: 97 pour 1997)",
                value=str(_annee_pre) if _annee_pre else "",
                key=f"lu_annee_{page_id}",
                help="Laissez vide si inconnu — la recherche sera plus large"
            )

        # Conversion année
        _lu_annee = None
        if _lu_annee_str.strip():
            try:
                _lu_annee = int(_lu_annee_str.strip())
            except ValueError:
                st.warning("L'année doit être un entier (ex: 97 pour 1997).")

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            _btn_search = st.button("Rechercher dans le répertoire", type="primary",
                                    key=f"btn_search_{page_id}", use_container_width=True)

        # Stockage du résultat dans session_state (persiste entre reruns)
        _lookup_key = f"_lookup_result_{base_name}_{page_id}"
        _confirmed_key = f"_lookup_confirmed_{base_name}_{page_id}"

        if _btn_search:
            with st.spinner("Recherche dans le répertoire Excel..."):
                _result = find_dossier(
                    commune=_lu_commune,
                    section=_lu_section,
                    parcelles_raw=_lu_parcelle_str,
                    annee_2ch=_lu_annee,
                )
            st.session_state[_lookup_key] = _result
            # Réinitialiser la confirmation si nouvelle recherche
            if _confirmed_key in st.session_state:
                del st.session_state[_confirmed_key]

        _result = st.session_state.get(_lookup_key, None)
        _confirmed = st.session_state.get(_confirmed_key, None)

        if _result:
            status = _result.get("status", "")

            # ── Résultat MATCH UNIQUE ─────────────────────────────────────
            if status == "MATCH_UNIQUE":
                st.markdown(f"""
                <div style="background:#d1fae5;border-left:5px solid #10b981;
                padding:1rem 1.5rem;border-radius:10px;margin:1rem 0;">
                <strong style="color:#065f46;font-size:1rem;">Match unique trouvé</strong><br>
                <table style="margin-top:0.5rem;border-collapse:collapse;width:100%;">
                <tr><td style="color:#374151;padding:2px 8px;"><b>Référence</b></td>
                    <td style="color:#065f46;font-weight:700;font-size:1.1rem;">{_result['ref_dossier']}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Commune</b></td>
                    <td>{_result['commune_excel']}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Section</b></td>
                    <td>{_result['section_excel']}
                    {"&nbsp;<em style='color:#d97706'>différente de l'OCR (" + _lu_section + ")</em>" if _result['section_excel'] != _lu_section else "&nbsp;✓"}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Parcelle</b></td>
                    <td>{_result.get('parcelle_excel', '—')}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Type</b></td>
                    <td>{_result['op_code_excel']} → Code GF : <strong>{_result['op_code_gf'] or '(à préciser)'}</strong></td></tr>
                <tr><td style="padding:2px 8px;"><b>Propriétaires</b></td>
                    <td style="font-size:0.85rem;">{_result.get('prop_anciens','')} / {_result.get('prop_nouveaux','')}</td></tr>
                </table>
                </div>
                """, unsafe_allow_html=True)

                if _result['section_excel'] != _lu_section:
                    st.info(
                        f"La section dans le répertoire est **{_result['section_excel']}**, "
                        f"mais l'OCR a extrait **{_lu_section}**. "
                        "Corrigez la section dans le formulaire principal avant de verser.",

                    )

                _c1, _c2 = st.columns([1, 2])
                with _c1:
                    if st.button("Confirmer ce dossier", type="primary",
                                  key=f"btn_confirm_{page_id}", use_container_width=True):
                        st.session_state[_confirmed_key] = _result
                        st.rerun()
                with _c2:
                    if st.button("Nouvelle recherche", key=f"btn_reset_{page_id}", use_container_width=True):
                        if _lookup_key in st.session_state:
                            del st.session_state[_lookup_key]
                        if _confirmed_key in st.session_state:
                            del st.session_state[_confirmed_key]
                        st.rerun()

            # ── Résultat CANDIDATS ────────────────────────────────────────
            elif status == "CANDIDATS":
                candidats = _result.get("candidats", [])
                st.warning(
                    f"{len(candidats)} candidat(s) trouvé(s). "
                    "Cliquez sur la ligne correspondante dans le tableau ci-dessous."
                )
                st.info(
                    "💡 **Attention :** Prenez le temps d'identifier le bon dossier. "
                    "La date du répertoire peut différer de celle du plan, et un même propriétaire "
                    "peut posséder plusieurs parcelles dans la commune. Assurez-vous qu'il s'agit du bon acte."
                )
                candidat_df = pd.DataFrame(candidats)
                display_cols = [c for c in ["ref_dossier","annee_full","commune","section","parcelle","op_code_excel","prop_anciens"] if c in candidat_df.columns]
                candidat_df_display = candidat_df[display_cols].copy()
                candidat_df_display.columns = ["Référence","Année","Commune","Section","Parcelle","Type","Propriétaires"][: len(display_cols)]

                selection = st.dataframe(
                    candidat_df_display,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"candidat_table_{page_id}"
                )
                sel_rows = selection.selection.get("rows", []) if hasattr(selection, "selection") else []
                if sel_rows:
                    chosen = candidats[sel_rows[0]]
                    st.success(f"Sélection : **{chosen['ref_dossier']}** — {chosen['commune']} {chosen['section']} {chosen.get('parcelle','?')} ({chosen['op_code_excel']})")
                    if st.button("Confirmer la sélection", type="primary", key=f"btn_confirm_cand_{page_id}"):
                        st.session_state[_confirmed_key] = chosen
                        st.rerun()

            # ── Résultat NO_MATCH ─────────────────────────────────────────
            elif status == "NO_MATCH":
                st.error(
                    f"{_result.get('message', 'Aucune correspondance trouvée.')} "
                    "Vérifiez la commune et la parcelle, ou saisissez la référence manuellement ci-dessous.",

                )
                _manual_ref = st.text_input(
                    "Référence dossier manuelle (ex: 97050)",
                    key=f"manual_ref_{page_id}",
                    help="Saisissez la référence trouvée manuellement dans le répertoire papier"
                )
                _manual_op = st.text_input(
                    "Code opération Géofoncier (ex: Da pour Document d'Arpentage)",
                    key=f"manual_op_{page_id}",
                )
                if _manual_ref and st.button("Utiliser cette référence manuelle", key=f"btn_manual_{page_id}"):
                    st.session_state[_confirmed_key] = {
                        "ref_dossier": _manual_ref.strip(),
                        "op_code_gf": _manual_op.strip(),
                        "op_code_excel": "",
                        "annee_full": None,
                        "section_excel": _lu_section,
                        "parcelle_excel": None,
                        "commune_excel": _lu_commune,
                    }
                    st.rerun()

            elif status == "ERREUR":
                st.error(f"Erreur technique : {_result.get('message', '')}")

        # Le passage vers l'étape 2 (Carte) se fait plus bas dans le flux unifié.


elif _geometre_val in {"RACAT", "CEYTE"}:
    # ══════════════════════════════════════════════════════════════════
    # Section Racat & Ceyte — Répertoire 1959–fév. 1964
    # ══════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:#f8fafc;
    padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;
    border-left:5px solid #f59e0b;">
    <h3 style="margin:0;font-size:1.3rem;font-weight:700;">
        Résolution répertoire — Archives Racat &amp; Ceyte
    </h3>
    <p style="margin:0.4rem 0 0;color:#94a3b8;font-size:0.9rem;">
        Répertoire couvrant janvier 1959 à février 1964.
    </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        from repertoire_racat_ceyte import (
            find_dossier_rc, clean_parcelles_for_lookup_rc,
            check_hors_repertoire, GEOMETRES_REPERTOIRE_RC
        )
        _rc_lookup_ok = True
    except ImportError as _rc_err:
        st.error(f"Module repertoire_racat_ceyte introuvable : {_rc_err}")
        _rc_lookup_ok = False

    if _rc_lookup_ok:
        # Pré-remplissage
        _rc_commune_pre = str(row.get("Commune", "")).strip()
        _rc_section_pre = str(row.get("Section", "")).strip().upper()
        _rc_date_pre    = str(row.get("Date", "")).strip()

        # Extraire année et mois depuis la date
        _rc_annee_plan  = None
        _rc_mois_plan   = None
        import re as _re_rc
        _m_date = _re_rc.search(r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})", _rc_date_pre)
        if _m_date:
            _yr = int(_m_date.group(3))
            _rc_annee_plan = 2000 + _yr if _yr <= 30 else (1900 + _yr if _yr < 100 else _yr)
            _rc_mois_plan  = int(_m_date.group(2))

        # Avertissement hors-répertoire (si date > fév. 1964)
        if check_hors_repertoire(_rc_annee_plan, _rc_mois_plan):
            st.warning(
                f"**Attention — hors répertoire :** Le répertoire Racat & Ceyte ne couvre que "
                f"jusqu'à **février 1964**. Ce dossier (date estimée : {_rc_date_pre or 'inconnue'}) "
                "est postérieur à cette date — aucune correspondance possible dans ce répertoire. "
                "Une recherche manuelle dans les archives papier est nécessaire."
            )
        else:
            # Parcelles
            _rc_parcs_concat = " ".join([
                str(row.get("Anciennes_Parcelles", "")),
                str(row.get("Nouvelles_Parcelles", "")),
                str(row.get("Parcelles", ""))
            ])
            _rc_parcs_propres = clean_parcelles_for_lookup_rc(_rc_parcs_concat)
            _rc_parcs_pre_str = ", ".join(str(p) for p in _rc_parcs_propres[:5]) if _rc_parcs_propres else ""

            _rc_col_a, _rc_col_b = st.columns([1, 1])
            with _rc_col_a:
                _rc_lu_commune = st.text_input("Commune", value=_rc_commune_pre,
                                               key=f"rc_commune_{page_id}")
                _rc_lu_section = st.text_input("Section", value=_rc_section_pre,
                                               key=f"rc_section_{page_id}")
            with _rc_col_b:
                _rc_lu_parcs = st.text_input(
                    "Parcelle(s) — acquéreur en priorité",
                    value=_rc_parcs_pre_str,
                    key=f"rc_parcelles_{page_id}",
                    help="Le numéro à l'acquéreur est prioritaire, mais les 3 colonnes sont cherchées."
                )
                _rc_lu_annee_str = st.text_input(
                    "Année (4 chiffres, ex: 1961)",
                    value=str(_rc_annee_plan) if _rc_annee_plan else "",
                    key=f"rc_annee_{page_id}",
                    help="Laissez vide pour une recherche sans filtre d'année."
                )

            _rc_annee_input = None
            if _rc_lu_annee_str.strip():
                try:
                    _rc_annee_input = int(_rc_lu_annee_str.strip())
                except ValueError:
                    st.warning("L'année doit être un entier (ex: 1961).")

            _rc_btn_key    = f"rc_btn_search_{page_id}"
            _rc_lookup_key = f"_rc_lookup_result_{base_name}_{page_id}"
            _rc_conf_key   = f"_rc_lookup_confirmed_{base_name}_{page_id}"

            _col_rc_btn, _ = st.columns([1, 3])
            with _col_rc_btn:
                if st.button("Rechercher dans le répertoire Racat/Ceyte",
                             type="primary", key=_rc_btn_key, use_container_width=True):
                    with st.spinner("Recherche en cours..."):
                        _rc_result = find_dossier_rc(
                            commune=_rc_lu_commune,
                            section=_rc_lu_section,
                            parcelles_raw=_rc_lu_parcs,
                            annee_plan=_rc_annee_input,
                        )
                    st.session_state[_rc_lookup_key] = _rc_result
                    if _rc_conf_key in st.session_state:
                        del st.session_state[_rc_conf_key]

            _rc_result   = st.session_state.get(_rc_lookup_key, None)
            _rc_confirmed = st.session_state.get(_rc_conf_key, None)

            if _rc_result:
                _rc_status = _rc_result.get("status", "")

                if _rc_status == "MATCH_UNIQUE":
                    st.markdown(f"""
                    <div style="background:#d1fae5;border-left:5px solid #10b981;
                    padding:1rem 1.5rem;border-radius:10px;margin:1rem 0;">
                    <strong style="color:#065f46;font-size:1rem;">Match unique trouvé</strong><br>
                    <table style="margin-top:0.5rem;border-collapse:collapse;width:100%;">
                    <tr><td style="padding:2px 8px;"><b>Référence</b></td>
                        <td style="color:#065f46;font-weight:700;font-size:1.1rem;">{_rc_result['ref_dossier']}</td></tr>
                    <tr><td style="padding:2px 8px;"><b>Commune</b></td>
                        <td>{_rc_result['commune_excel']}</td></tr>
                    <tr><td style="padding:2px 8px;"><b>Section</b></td>
                        <td>{_rc_result['section_excel']}</td></tr>
                    <tr><td style="padding:2px 8px;"><b>Parcelle acquéreur</b></td>
                        <td>{', '.join(str(p) for p in _rc_result.get('parcelle_acq', []))}</td></tr>
                    <tr><td style="padding:2px 8px;"><b>Parcelle origine</b></td>
                        <td>{', '.join(str(p) for p in _rc_result.get('parcelle_orig', []))}</td></tr>
                    <tr><td style="padding:2px 8px;"><b>Vendeur</b></td>
                        <td style="font-size:0.85rem;">{str(_rc_result.get('vendeur',''))[:80]}</td></tr>
                    <tr><td style="padding:2px 8px;"><b>Acquéreur</b></td>
                        <td style="font-size:0.85rem;">{str(_rc_result.get('acquereur',''))[:80]}</td></tr>
                    </table>
                    </div>
                    """, unsafe_allow_html=True)

                    _rc_c1, _rc_c2 = st.columns([1, 2])
                    with _rc_c1:
                        if st.button("Confirmer ce dossier", type="primary",
                                     key=f"rc_btn_confirm_{page_id}", use_container_width=True):
                            st.session_state[_rc_conf_key] = _rc_result
                            st.rerun()
                    with _rc_c2:
                        if st.button("Nouvelle recherche", key=f"rc_btn_reset_{page_id}",
                                     use_container_width=True):
                            for _k in [_rc_lookup_key, _rc_conf_key]:
                                if _k in st.session_state: del st.session_state[_k]
                            st.rerun()

                elif _rc_status == "CANDIDATS":
                    _rc_cands = _rc_result.get("candidats", [])
                    st.warning(f"{len(_rc_cands)} candidat(s) trouvé(s). Sélectionnez le bon dossier.")
                    st.info(
                        "💡 **Attention :** Prenez le temps d'identifier le bon dossier. "
                        "La date du répertoire peut différer de celle du plan, et un même propriétaire "
                        "peut posséder plusieurs parcelles dans la commune. Assurez-vous qu'il s'agit du bon acte."
                    )
                    _rc_cand_df = pd.DataFrame(_rc_cands)
                    _rc_display_cols = [c for c in ["ref_dossier", "annee", "commune", "section",
                                                     "parcelle_acq", "vendeur"] if c in _rc_cand_df.columns]
                    _rc_sel = st.dataframe(
                        _rc_cand_df[_rc_display_cols],
                        use_container_width=True, hide_index=True,
                        on_select="rerun", selection_mode="single-row",
                        key=f"rc_cand_table_{page_id}"
                    )
                    _rc_sel_rows = _rc_sel.selection.get("rows", []) if hasattr(_rc_sel, "selection") else []
                    if _rc_sel_rows:
                        _rc_chosen = _rc_cands[_rc_sel_rows[0]]
                        st.success(f"Sélection : **{_rc_chosen['ref_dossier']}** — {_rc_chosen['commune']} {_rc_chosen['section']}")
                        if st.button("Confirmer la sélection", type="primary",
                                     key=f"rc_btn_confirm_cand_{page_id}"):
                            st.session_state[_rc_conf_key] = _rc_chosen
                            st.rerun()

                elif _rc_status == "NO_MATCH":
                    st.error(_rc_result.get("message", "Aucune correspondance trouvée."))
                    _rc_manual = st.text_input(
                        "Référence manuelle (ex: 59/488)",
                        key=f"rc_manual_{page_id}"
                    )
                    if _rc_manual and st.button("Utiliser cette référence manuelle",
                                                key=f"rc_btn_manual_{page_id}"):
                        st.session_state[_rc_conf_key] = {
                            "ref_dossier": _rc_manual.strip(), "annee": _rc_annee_input,
                            "section_excel": _rc_lu_section, "commune_excel": _rc_lu_commune,
                        }
                        st.rerun()

            # Dossier confirmé — affichage récapitulatif
            if _rc_confirmed:
                st.success(
                    f"Dossier confirmé : **{_rc_confirmed.get('ref_dossier', '—')}** "
                    f"({_rc_confirmed.get('commune_excel', '')} {_rc_confirmed.get('section_excel', '')})"
                )
                st.info(
                    "Pour verser sur Géofoncier, utilisez l'export CSV normé ci-dessus avec cette référence. "
                    "Le versement API automatique pour Racat/Ceyte sera disponible dans une prochaine version.",
                )

# ── Archives DUPUY — Répertoire Excel spécifique ─────────────────────────────────
elif _geometre_val in _GEOMETRES_REPERTOIRE_DUPUY:
    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1c1108,#3d2100);color:#f8fafc;
    padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;
    border-left:5px solid #f59e0b;">
    <h3 style="margin:0;font-size:1.3rem;font-weight:700;">
        Résolution répertoire — Archives Roger DUPUY
    </h3>
    <p style="margin:0.4rem 0 0;color:#fcd34d;font-size:0.9rem;">
        Recherche par commune et année dans le répertoire numérique des archives DUPUY.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Avertissement permanent sur les registres partiels ───────────────────────
    st.warning(
        "⚠️ **Registres partiellement numérisés** — "
        "Les archives de Roger DUPUY ne sont pas toutes présentes dans la base. "
        "Si aucun résultat n'est trouvé, le dossier peut exister dans les registres papier. "
        "Consultez les archives physiques en cas de doute."
    )

    # ── Import du module de lookup DUPUY ─────────────────────────────────────
    try:
        from repertoire_dupuy_lookup import (
            find_dossier_dupuy, build_ref_dupuy,
            GEOMETRES_REPERTOIRE_DUPUY as _DUPUY_GEO_SET
        )
        _dupuy_ok = True
    except ImportError as _dupuy_err:
        st.error(f"Module repertoire_dupuy_lookup introuvable : {_dupuy_err}")
        _dupuy_ok = False

    if _dupuy_ok:
        # Pré-remplissage depuis le CSV
        _dp_commune_pre = str(row.get("Commune", "")).strip()
        _dp_date_pre    = str(row.get("Date", "")).strip()

        # Extraire l'année (4 chiffres) depuis la date du plan
        _dp_annee_pre = None
        import re as _re_dp
        _m_dp = _re_dp.search(r"(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})", _dp_date_pre)
        if _m_dp:
            _yr = int(_m_dp.group(3))
            _dp_annee_pre = 2000 + _yr if _yr <= 30 else (1900 + _yr if _yr < 100 else _yr)
        else:
            # Cherche une année seule à 4 chiffres
            _m_dp4 = _re_dp.search(r"\b(19\d{2}|20[0-2]\d)\b", _dp_date_pre)
            if _m_dp4:
                _dp_annee_pre = int(_m_dp4.group(1))

        # Champs de saisie
        _dp_col_a, _dp_col_b = st.columns([1, 1])
        with _dp_col_a:
            _dp_commune = st.text_input(
                "Commune",
                value=_dp_commune_pre,
                key=f"dp_commune_{page_id}",
                help="Le nom de la commune tel qu'inscrit sur le plan."
            )
        with _dp_col_b:
            _dp_annee_str = st.text_input(
                "Année (4 chiffres, ex: 1975)",
                value=str(_dp_annee_pre) if _dp_annee_pre else "",
                key=f"dp_annee_{page_id}",
                help="Année du document. Laissez vide pour chercher sur toutes les années de la commune."
            )

        # Champs optionnels pour améliorer le tri (non obligatoires)
        with st.expander(
            "🔍 Aide au tri (optionnel) — Noms de propriétaires visibles sur le plan",
            expanded=False
        ):
            st.caption(
                "Si vous voyez les noms des propriétaires sur le plan, "
                "saisissez-les ici pour que le dossier le plus probable apparaisse en tête de liste."
            )
            _dp_col_c, _dp_col_d = st.columns(2)
            with _dp_col_c:
                _dp_hint_anc = st.text_area(
                    "Anciens propriétaires (texte libre)",
                    height=80, key=f"dp_hint_anc_{page_id}",
                    label_visibility="visible"
                )
            with _dp_col_d:
                _dp_hint_nou = st.text_area(
                    "Nouveaux propriétaires (texte libre)",
                    height=80, key=f"dp_hint_nou_{page_id}",
                    label_visibility="visible"
                )

        # Parsing année
        _dp_annee = None
        if _dp_annee_str.strip():
            try:
                _dp_annee = int(_dp_annee_str.strip())
                if not (1900 <= _dp_annee <= 2050):
                    st.warning("L'année doit être un entier à 4 chiffres (ex: 1975).")
                    _dp_annee = None
            except ValueError:
                st.warning("L'année doit être un entier (ex: 1975).")

        _dp_lookup_key  = f"_dp_lookup_result_{base_name}_{page_id}"
        _dp_conf_key    = f"_dp_lookup_confirmed_{base_name}_{page_id}"

        _dp_col_btn, _ = st.columns([1, 3])
        with _dp_col_btn:
            if st.button(
                "Rechercher dans les archives DUPUY",
                type="primary", key=f"dp_btn_search_{page_id}",
                use_container_width=True
            ):
                with st.spinner("Recherche dans le répertoire DUPUY..."):
                    _dp_result = find_dossier_dupuy(
                        commune=_dp_commune,
                        annee=_dp_annee,
                        hint_anciens=st.session_state.get(f"dp_hint_anc_{page_id}", ""),
                        hint_nouveaux=st.session_state.get(f"dp_hint_nou_{page_id}", ""),
                    )
                st.session_state[_dp_lookup_key] = _dp_result
                if _dp_conf_key in st.session_state:
                    del st.session_state[_dp_conf_key]

        _dp_result    = st.session_state.get(_dp_lookup_key, None)
        _dp_confirmed = st.session_state.get(_dp_conf_key, None)

        if _dp_result:
            _dp_status = _dp_result.get("status", "")

            if _dp_status == "CANDIDATS":
                _dp_cands = _dp_result.get("candidats", [])
                _dp_nb    = _dp_result.get("nb_candidats", len(_dp_cands))

                # Message récapitulatif
                st.info(_dp_result.get("message", ""))

                # Tableau interactif des candidats
                # Colonnes affichées : Ref, Année, Commune, Anciens prop, Nouveaux prop, Score sim., Notes
                _dp_rows_display = []
                for _cand in _dp_cands:
                    _score_txt = f"{_cand['score_prop']}%" if _cand['score_prop'] > 0 else "—"
                    _dp_rows_display.append({
                        "Référence":       _cand["ref_dossier"],
                        "Année":           _cand["annee"],
                        "Commune":         _cand["commune"],
                        "Anciens Propriétaires": _cand["prop_anciens"][:80] + ("..." if len(_cand["prop_anciens"]) > 80 else ""),
                        "Nouveaux Propriétaires": _cand["prop_nouveaux"][:80] + ("..." if len(_cand["prop_nouveaux"]) > 80 else ""),
                        "Similarité Prop.": _score_txt,
                        "Notes":           _cand.get("notes", ""),
                    })

                _dp_df_display = pd.DataFrame(_dp_rows_display)

                _dp_sel = st.dataframe(
                    _dp_df_display,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=f"dp_cand_table_{page_id}",
                    column_config={
                        "Référence":       st.column_config.TextColumn(width="small"),
                        "Année":           st.column_config.NumberColumn(width="small", format="%d"),
                        "Commune":         st.column_config.TextColumn(width="small"),
                        "Anciens Propriétaires":  st.column_config.TextColumn(width="large"),
                        "Nouveaux Propriétaires": st.column_config.TextColumn(width="large"),
                        "Similarité Prop.": st.column_config.TextColumn(width="small"),
                        "Notes":           st.column_config.TextColumn(width="medium"),
                    }
                )

                # Guide de lecture
                st.caption(
                    "💡 Cliquez sur une ligne pour la sélectionner, puis confirmez. "
                    "Le premier résultat est le plus probable (meilleure similarité de propriétaires). "
                    "Vérifiez les noms avant de confirmer."
                )

                _dp_sel_rows = _dp_sel.selection.get("rows", []) if hasattr(_dp_sel, "selection") else []
                if _dp_sel_rows:
                    _dp_chosen = _dp_cands[_dp_sel_rows[0]]
                    st.success(
                        f"✅ Sélection : **{_dp_chosen['ref_dossier']}** — "
                        f"{_dp_chosen['commune']} ({_dp_chosen['annee']}) — "
                        f"Anciens : {str(_dp_chosen['prop_anciens'])[:60]}"
                    )
                    _dp_c1, _dp_c2 = st.columns([1, 2])
                    with _dp_c1:
                        if st.button(
                            "Confirmer ce dossier DUPUY",
                            type="primary",
                            key=f"dp_btn_confirm_{page_id}",
                            use_container_width=True
                        ):
                            st.session_state[_dp_conf_key] = _dp_chosen
                            st.rerun()
                    with _dp_c2:
                        if st.button(
                            "Nouvelle recherche",
                            key=f"dp_btn_reset_{page_id}",
                            use_container_width=True
                        ):
                            for _k in [_dp_lookup_key, _dp_conf_key]:
                                if _k in st.session_state: del st.session_state[_k]
                            st.rerun()

            elif _dp_status == "NO_MATCH":
                st.error(_dp_result.get("message", "Aucune correspondance trouvée."))
                st.markdown(
                    "> ⚠️ **Rappel** : Ce résultat ne signifie pas que le dossier est inexistant. "
                    "Il peut se trouver dans les registres papier non encore numérisés."
                )
                # Saisie manuelle
                _dp_manual_ref = st.text_input(
                    "Référence manuelle (ex: 70123)",
                    key=f"dp_manual_ref_{page_id}",
                    help="Saisissez la référence trouvée manuellement dans le registre papier"
                )
                if _dp_manual_ref and st.button(
                    "Utiliser cette référence manuelle",
                    key=f"dp_btn_manual_{page_id}"
                ):
                    st.session_state[_dp_conf_key] = {
                        "ref_dossier":   _dp_manual_ref.strip(),
                        "annee":         _dp_annee,
                        "n_dossier":     _dp_manual_ref.strip(),
                        "commune":       _dp_commune,
                        "prop_anciens":  "",
                        "prop_nouveaux": "",
                        "notes":         "Saisie manuelle",
                        "score_commune": 0,
                        "score_prop":    0,
                    }
                    st.rerun()

            elif _dp_status == "ERREUR":
                st.error(f"Erreur technique : {_dp_result.get('message', '')}")
                st.info(
                    "Vérifiez que le fichier `Repertoire_Archives_DUPUY.xlsx` est accessible "
                    f"sur `Z:\\_ArchivesDUPUY\\` ou dans le dossier `outputs/` de l'extracteur Dupuy."
                )

        # Dossier confirmé — récapitulatif
        if _dp_confirmed:
            st.markdown(
                f"""
                <div style="background:#d1fae5;border-left:5px solid #10b981;
                padding:1rem 1.5rem;border-radius:10px;margin:1rem 0;">
                <strong style="color:#065f46;font-size:1rem;">Dossier DUPUY confirmé</strong><br>
                <table style="margin-top:0.5rem;border-collapse:collapse;width:100%;">
                <tr><td style="padding:2px 8px;"><b>Référence</b></td>
                    <td style="color:#065f46;font-weight:700;font-size:1.1rem;">{_dp_confirmed.get('ref_dossier', '—')}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Commune</b></td>
                    <td>{_dp_confirmed.get('commune', '')}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Année</b></td>
                    <td>{_dp_confirmed.get('annee', '')}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Anciens Propriétaires</b></td>
                    <td style="font-size:0.85rem;">{str(_dp_confirmed.get('prop_anciens', ''))[:100]}</td></tr>
                <tr><td style="padding:2px 8px;"><b>Nouveaux Propriétaires</b></td>
                    <td style="font-size:0.85rem;">{str(_dp_confirmed.get('prop_nouveaux', ''))[:100]}</td></tr>
                </table>
                </div>
                """,
                unsafe_allow_html=True
            )

# Géomètres avec flux API direct (sans répertoire Excel dédié)
elif _geometre_val in _GEOMETRES_API_DIRECT:
    st.markdown("---")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);color:#f8fafc;
    padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;
    border-left:5px solid #a78bfa;">
    <h3 style="margin:0;font-size:1.3rem;font-weight:700;">
        Versement Géofoncier — Archives {_geometre_val.capitalize()}
    </h3>
    <p style="margin:0.4rem 0 0;color:#94a3b8;font-size:0.9rem;">
        Ce géomètre ne possède pas encore de répertoire Excel numérique.
        Saisissez la référence manuellement pouvoir verser.
    </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        from geofoncier_api import (
            create_geofoncier_dossier, upload_document_to_dossier,
            get_insee_from_commune, format_date_iso
        )
        _ad_ok = True
    except ImportError as _ad_err:
        st.error(f"Module géofoncier introuvable : {_ad_err}")
        _ad_ok = False

    if _ad_ok:
        _ad_commune = str(row.get("Commune", "")).strip()
        _ad_section  = str(row.get("Section", "")).strip().upper()
        _ad_date     = str(row.get("Date", "")).strip()
        _ad_parcelle = str(row.get("Parcelles", "")).strip()
        _, _ad_insee = get_insee_from_commune(_ad_commune)

        st.markdown("#### Référence manuelle")
        _col_ad1, _col_ad2 = st.columns(2)
        with _col_ad1:
            _ad_ref = st.text_input("Référence dossier", key=f"ad_ref_{page_id}",
                                    help="Trouvez la référence dans le registre papier du géomètre.")
            _ad_op  = st.text_input("Code opération Géofoncier", value="Da", key=f"ad_op_{page_id}")
        with _col_ad2:
            _ad_date_iso = st.text_input("Date (AAAA-MM-JJ)", value=format_date_iso(_ad_date),
                                         key=f"ad_date_{page_id}")
            _ad_statut = st.selectbox("Statut", ["Achevé", "Indéterminé", "En cours"],
                                      key=f"ad_statut_{page_id}")

        _ad_can = bool(_ad_ref and _ad_op)

        if not _ad_insee:
            st.warning(f"Code INSEE introuvable pour '{_ad_commune}' — vérifiez la commune.")

        if st.button("Confirmer cette référence", type="primary", disabled=not _ad_ref, key=f"ad_btn_conf_{page_id}"):
            st.session_state[f"_ad_lookup_confirmed_{base_name}_{page_id}"] = {
                "ref_dossier": _ad_ref.strip(),
                "commune_excel": _ad_commune,
                "section_excel": _ad_section,
                "parcelle_excel": _ad_parcelle,
                "date_cadastre": _ad_date_iso,
                "op_code_gf": _ad_op.strip(),
                "op_code_excel": _ad_op.strip(),
                "annee_full": int(_ad_date_iso[:4]) if _ad_date_iso and len(_ad_date_iso)>=4 else None,
                "enr_statut": _ad_statut
            }
            st.rerun()
else:
    # Géomètre non pris en charge
    if _geometre_val and _geometre_val not in {"", "NAN", "NONE"}:
        st.markdown("---")
        st.info(
            f"Le géomètre **{_geometre_val}** n'est pas encore pris en charge pour la résolution automatique. "
            "Utilisez l'export CSV ci-dessus.",
        )


# ══════════════════════════════════════════════════════════════════
# UNIFICATION DU FLUX GEOFONCIER (Étapes 2 et 3) POUR TOUS GEOMETRES
# ══════════════════════════════════════════════════════════════════
_unified_dossier = None
if 'page_id' in locals() and 'base_name' in locals():
    if _geometre_val in _GEOMETRES_REPERTOIRE and st.session_state.get(f"_lookup_confirmed_{base_name}_{page_id}"):
        _unified_dossier = st.session_state.get(f"_lookup_confirmed_{base_name}_{page_id}")
    elif _geometre_val in {"RACAT", "CEYTE"} and st.session_state.get(f"_rc_lookup_confirmed_{base_name}_{page_id}"):
        _unified_dossier = st.session_state.get(f"_rc_lookup_confirmed_{base_name}_{page_id}")
        _unified_dossier["parcelle_excel"] = _unified_dossier.get("parcelle_acq", [None])[0] if _unified_dossier.get("parcelle_acq") else None
        _unified_dossier["op_code_gf"] = "Em"  # Em = Modification du parcellaire cadastral
        if _unified_dossier.get("annee"):
            _unified_dossier["date_cadastre"] = f"{_unified_dossier['annee']}-01-01"
    elif _geometre_val in _GEOMETRES_REPERTOIRE_DUPUY and st.session_state.get(f"_dp_lookup_confirmed_{base_name}_{page_id}"):
        _dp_conf_data = st.session_state.get(f"_dp_lookup_confirmed_{base_name}_{page_id}")
        _unified_dossier = {
            "ref_dossier":   _dp_conf_data.get("ref_dossier", ""),
            "commune_excel": _dp_conf_data.get("commune", ""),
            "section_excel": str(row.get("Section", "")).strip().upper(),
            "parcelle_excel": None,  # pas de parcelle dans les registres Dupuy
            "op_code_gf":    "Em",   # Modification du parcellaire cadastral par défaut
            "op_code_excel": "DA",
            "annee_full":    _dp_conf_data.get("annee"),
            "date_cadastre": f"{_dp_conf_data['annee']}-01-01" if _dp_conf_data.get("annee") else None,
            "enr_statut":    "Achevé",
            "prop_anciens":  _dp_conf_data.get("prop_anciens", ""),
            "prop_nouveaux": _dp_conf_data.get("prop_nouveaux", ""),
        }
    elif _geometre_val in _GEOMETRES_API_DIRECT and st.session_state.get(f"_ad_lookup_confirmed_{base_name}_{page_id}"):
        _unified_dossier = st.session_state.get(f"_ad_lookup_confirmed_{base_name}_{page_id}")

_confirmed = None  # Valeur par défaut : aucun dossier confirmé
if _unified_dossier:
    _confirmed = _unified_dossier
if _confirmed:
    _sec_clean = ""
    _num_clean = ""
    
    # Clés de session globales pour l'étape 2 (Carte)
    _map_confirmed_key = f"_map_confirmed_{base_name}_{page_id}"
    _map_coords_key    = f"_map_coords_{base_name}_{page_id}"
    
    # Clé de sauvegarde du dossier (diffère selon le géomètre)
    if _geometre_val in {"RACAT", "CEYTE"}:
        _unified_confirmed_key = f"_rc_lookup_confirmed_{base_name}_{page_id}"
    elif _geometre_val in _GEOMETRES_REPERTOIRE_DUPUY:
        _unified_confirmed_key = f"_dp_lookup_confirmed_{base_name}_{page_id}"
    elif _geometre_val in _GEOMETRES_API_DIRECT:
        _unified_confirmed_key = f"_ad_lookup_confirmed_{base_name}_{page_id}"
    else: # Harrois/Barrial
        _unified_confirmed_key = f"_lookup_confirmed_{base_name}_{page_id}"

    
    st.markdown("---")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);color:#f8fafc;
    padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;
    border-left:5px solid #10b981;">
    <h3 style="margin:0;font-size:1.3rem;font-weight:700;">
        Étape 2 — Localisation cartographique
    </h3>
    <p style="margin:0.4rem 0 0;color:#94a3b8;font-size:0.9rem;">
        Vérifiez que la pastille sera créée au bon endroit. Déplacez le marqueur si nécessaire.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ **Attention aux doublons :** Si une pastille est déjà visible à proximité sur la carte, vérifiez directement sur [**Géofoncier**](https://expert.geofoncier.fr) de quoi il s'agit. Il ne faut pas créer deux pastilles pour le même dossier.")
    st.info("**Parcelle introuvable ?** Si la parcelle ne s'affiche pas sur la carte (même après avoir cliqué sur 'Actualiser la carte' et sans la filiation), essayez de chercher un autre numéro de parcelle présent sur le plan. Soyez vigilant avec le domaine public (routes, places) qui ne possède généralement pas de numéro cadastral propre.")

    # Récupération préliminaire des infos pour la carte
    # Fallback universel sur les champs du CSV si les variables locales du bloc
    # géomètre (ex: _lu_commune pour Harrois) ne sont pas définies dans ce scope.
    # Fallback universel sur les champs du CSV si les variables locales du bloc
    # géomètre (ex: _lu_commune pour Harrois) ne sont pas définies dans ce scope.
    # NOTE : on n'utilise PAS locals() qui est fragile si le code est refactorisé dans une fonction.
    _csv_commune   = str(row.get("Commune",  "")).strip()
    _csv_section   = str(row.get("Section",  "")).strip().upper()
    _csv_parcelles = " ".join(filter(None, [
        str(row.get("Nouvelles_Parcelles", "")),
        str(row.get("Anciennes_Parcelles", "")),
        str(row.get("Parcelles", "")),
    ])).strip()

    # Utilise les variables Harrois/Barrial si disponibles (dans le scope global du script),
    # sinon repli sur les valeurs CSV.
    _fb_commune  = st.session_state.get(f"lu_commune_{page_id}",  _csv_commune)  or _csv_commune
    _fb_section  = st.session_state.get(f"lu_section_{page_id}",  _csv_section)  or _csv_section
    _fb_parcelle = st.session_state.get(f"lu_parcelle_{page_id}", _csv_parcelles) or _csv_parcelles

    _map_commune  = _confirmed.get("commune_excel") or _confirmed.get("commune") or _fb_commune
    _map_section  = _confirmed.get("section_excel", _confirmed.get("section",  _fb_section))

    _map_parcelle = _confirmed.get("parcelle_excel", _confirmed.get("parcelle", None))
    if pd.isna(_map_parcelle) or not _map_parcelle:
        # Extraction robuste du premier numéro de parcelle (pas de dépendance externe)
        import re as _re_parc
        _parc_nums = _re_parc.findall(r"\b\d+\b", str(_fb_parcelle))
        _map_parcelle = int(_parc_nums[0]) if _parc_nums else None
    # ── IMPORTS UNIFIÉS ──────────────────────────────────────────────────────────
    # Garantit que les bibliothèques et API sont disponibles quel que soit le
    # chemin du géomètre (Dupuy, Racat, API Direct n'ont pas forcément ces imports)
    import folium
    from streamlit_folium import st_folium
    from geofoncier_api import (
        get_insee_from_commune, get_parcel_geometry, geocode_commune,
        verify_parcel_ign, create_geofoncier_dossier, upload_document_to_dossier,
        format_date_iso, refresh_geofoncier_token
    )
    
    _, _map_insee = get_insee_from_commune(_map_commune)

    # Clés de session pour correction manuelle section/parcelle
    _map_section_key  = f"map_section_{page_id}"
    _map_parcelle_key = f"map_parcelle_{page_id}"

    # ── Contrôles de correction section/parcelle ───────────────────
    _col_mc1, _col_mc2, _col_mc3 = st.columns([1, 1, 1])
    with _col_mc1:
        _map_section_input = st.text_input(
            "Section (corrigeable)",
            value=st.session_state.get(_map_section_key, _map_section or ""),
            key=_map_section_key,
            help="Modifiez si l'OCR a mal lu la section cadastrale."
        )
    with _col_mc2:
        _map_parcelle_input = st.text_input(
            "N° Parcelle (corrigeable)",
            value=st.session_state.get(_map_parcelle_key, str(_map_parcelle) if _map_parcelle else ""),
            key=_map_parcelle_key,
            help="Modifiez si le numéro de parcelle est incorrect."
        )
    with _col_mc3:
        _btn_refresh_map = st.button(
            "Actualiser la carte",
            key=f"btn_refresh_map_{page_id}",
            use_container_width=True,
            help="Recherche la parcelle corrigée sur la carte."
        )

    # Cache de géométrie dans session_state (évite les appels IGN répétés)
    # ─── Localisation en 3 niveaux : parcelle exacte → section → commune ───
    # L'API IGN apicarto cherche la parcelle avec le numero actuel.
    # Format attendu : section 2 cars (ex "AL"), numero 4 chiffres (ex "0160")
    # Si la parcelle n'est plus dans IGN (fusionnee, renumerotee),
    # on cherche n'importe quelle parcelle de la meme section pour
    # centrer la carte dans le bon quartier de la commune.
    # --- Formatage robuste section / numero (HORS CACHE pour reruns) ---
    _sec_clean = str(_map_section_input or "").strip().upper().zfill(2)
    _num_raw   = str(_map_parcelle_input or "").strip()
    _num_digits = "".join(c for c in _num_raw if c.isdigit())
    _num_clean  = _num_digits.zfill(4) if _num_digits else ""

    _geom_cache_key = f"_geom_v2_{page_id}_{_map_section_input}_{_map_parcelle_input}_{_map_insee}"
    if _geom_cache_key not in st.session_state or _btn_refresh_map:
        _geom = None
        _geo_status = "commune"  # "parcel" | "section" | "commune"

        # Etape 1 : parcelle exacte (API IGN apicarto)
        if _sec_clean and _num_clean and _map_insee:
            with st.spinner(f"Recherche parcelle {_sec_clean}-{_num_clean} sur IGN..."):
                _geom = get_parcel_geometry(_map_insee, _sec_clean, _num_clean)
            if _geom and _geom.get("found"):
                _geo_status = "parcel"
            else:
                # Etape 1.5 : Recherche dans l'historique DFI (Filiation)
                with st.spinner("Parcelle introuvable. Recherche dans la filiation (DFI)..."):
                    import cadastre_filiation
                    dfi_path = "dfi_07.json" # Fichier JSON généré à partir du TXT DFI
                    filiation_engine = cadastre_filiation.get_filiation_engine(dfi_path if __import__('os').path.exists(dfi_path) else None)
                    filles = filiation_engine.trouver_parcelles_actuelles(_map_insee, _sec_clean, _num_clean)
                    
                    if filles:
                        # Filiation trouvée ! On géocode TOUTES les parcelles filles
                        _geom_filles_all = []
                        for p_fille in filles:
                            g = get_parcel_geometry(_map_insee, p_fille['section'], p_fille['numero'])
                            if g and g.get("found"):
                                _geom_filles_all.append(g)
                                
                        if _geom_filles_all:
                            # On utilise le centroid de la première pour centrer la carte
                            _geom = _geom_filles_all[0].copy()
                            _geom["filles_filiation"] = filles
                            
                            # On fusionne tous les polygones dans une FeatureCollection
                            features = [g["geojson"] for g in _geom_filles_all if g.get("geojson")]
                            if features:
                                _geom["geojson"] = {
                                    "type": "FeatureCollection",
                                    "features": features
                                }
                            _geo_status = "filiation"

        # Etape 2 : centroide de n'importe quelle parcelle de la section
        if _geo_status not in ["parcel", "filiation"] and _sec_clean and _map_insee:
            with st.spinner(f"Parcelle non trouvee — recherche du centre section {_sec_clean}..."):
                import requests as _rq_sec
                try:
                    _sec_url = (
                        f"https://apicarto.ign.fr/api/cadastre/parcelle"
                        f"?code_insee={_map_insee}&section={_sec_clean}&_limit=5"
                    )
                    _sec_r = _rq_sec.get(_sec_url, timeout=7)
                    if _sec_r.status_code == 200:
                        _sec_feats = _sec_r.json().get("features", [])
                        if _sec_feats:
                            # Calculer le centroide moyen de toutes les parcelles trouvees
                            _all_lats, _all_lons = [], []
                            for _sf in _sec_feats:
                                _sg = _sf.get("geometry", {})
                                _sc = _sg.get("coordinates", [])
                                if _sc:
                                    # Handle both Polygon (1 ring) and MultiPolygon (list of polygons)
                                    _ring = _sc[0][0] if _sg.get("type") == "MultiPolygon" else _sc[0]
                                    _all_lats.append(sum(p[1] for p in _ring) / len(_ring))
                                    _all_lons.append(sum(p[0] for p in _ring) / len(_ring))
                            if _all_lats:
                                _sec_ctr = [sum(_all_lats)/len(_all_lats), sum(_all_lons)/len(_all_lons)]
                                _geom = {"centroid": _sec_ctr, "geojson": None, "found": False, "section_found": True}
                                _geo_status = "section"
                except Exception:
                    pass

        # Etape 3 : centroide de la commune (dernier recours)
        if _geo_status == "commune":
            _comm_ctr = geocode_commune(_map_commune, _map_insee)
            _geom = {"centroid": _comm_ctr, "geojson": None, "found": False, "section_found": False}

        st.session_state[_geom_cache_key] = _geom
        st.session_state[f"_geo_status_{page_id}"] = _geo_status

    _geom      = st.session_state.get(_geom_cache_key) or {"centroid": [44.7356, 4.5990], "found": False}
    _geo_status = st.session_state.get(f"_geo_status_{page_id}", "commune")

    _parcel_found   = (_geo_status == "parcel")
    _filiation_found = (_geo_status == "filiation")
    _section_approx = (_geo_status == "section")
    _map_center     = _geom.get("centroid") or [44.7356, 4.5990]

    # Position du marqueur :
    # Priorite : clic utilisateur sur cette carte > centroide IGN de la parcelle/section/commune
    _click_key   = f"_map_click_{base_name}_{page_id}"
    _saved_click = st.session_state.get(_click_key, None)
    _marker_pos  = _saved_click if _saved_click else _map_center

    # ── Bandeau de statut ──────────────────────────────────────────────────
    if _parcel_found:
        st.success(
            f"Parcelle **{_sec_clean}-{_num_clean}** localisee sur IGN "
            f"(commune {_map_commune}). Polygone orange = parcelle exacte. "
            "Cliquez sur la carte pour ajuster la position si besoin."
        )
    elif _filiation_found:
        filles_info = ", ".join([f"{f['section']}-{f['numero']}" for f in _geom.get("filles_filiation", [])])
        st.success(
            f"⚠️ Ancienne parcelle **{_map_section_input or _sec_clean}-{_map_parcelle_input or _num_clean}** introuvable, mais **filiation identifiée** ! "
            f"Elle correspond aujourd'hui aux parcelles : **{filles_info}**. "
            "Les emprises des nouvelles parcelles sont affichées (polygones violets) et le marqueur est placé au centre."
        )
        
        _filles = _geom.get("filles_filiation", [])
        if _filles:
            _new_sec = _filles[0]['section']
            _new_nums = ", ".join([f['numero'] for f in _filles if f['section'] == _new_sec])
            st.info(f"💡 Pour le versement Géofoncier (Étape 3), les nouvelles parcelles **{_new_sec} - {_new_nums}** seront automatiquement utilisées.")
    elif _section_approx:
        st.info(
            f"Parcelle **{_sec_clean}-{_num_clean}** introuvable dans IGN et sans historique (DFI). "
            f"Carte centree sur la **section {_sec_clean}** au zoom cadastral. "
            "**Cliquez sur la parcelle correcte** pour placer la pastille."
        )
    else:
        st.warning(
            f"Section **{_sec_clean}** non trouvee. Carte centree sur **{_map_commune}**. "
            "Naviguez dans le cadastre et **cliquez** pour positionner la pastille."
        )
    if _saved_click:
        st.success(
            f"Position choisie : **{_saved_click[0]:.5f}N, {_saved_click[1]:.5f}E**. "
            "Recliquez pour corriger."
        )

    # ── Construction de la carte Folium ────────────────────────────
    _zoom = 18 if (_parcel_found or _filiation_found) else 14
    # Zoom forcé à 18 si parcelle trouvée (sinon numéros illisibles au-dessous)
    _zoom_final = max(_zoom, 18) if (_parcel_found or _filiation_found) else _zoom
    _fmap = folium.Map(
        location=_marker_pos,
        zoom_start=_zoom_final,
        control_scale=True,
        tiles=None,  # On ne charge PAS le fond OSM par défaut
    )

    # ── FOND 1 : Plan IGN v2 (défaut — affiche routes, batiments, contexte)
    folium.TileLayer(
        tiles="https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
              "&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2&STYLE=normal&FORMAT=image/png"
              "&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}",
        attr="IGN-Géoportail — Plan IGN",
        name="Plan IGN (défaut)",
        max_zoom=19,
        show=True,
    ).add_to(_fmap)

    # ── FOND 2 : Orthophoto IGN haute résolution
    folium.TileLayer(
        tiles="https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
              "&LAYER=HR.ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg"
              "&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}",
        attr="IGN-Géoportail — Orthophoto",
        name="Orthophoto IGN",
        max_zoom=21,
        show=False,
    ).add_to(_fmap)

    # ── FOND 3 : Plan OSM (secours)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Plan OSM",
        show=False,
    ).add_to(_fmap)

    # ── OVERLAY : Cadastre IGN (numéros de parcelles) — ACTIF PAR DÉFAUT
    folium.WmsTileLayer(
        url="https://data.geopf.fr/wms-r/wms?",
        layers="CADASTRALPARCELS.PARCELLAIRE_EXPRESS",
        fmt="image/png",
        transparent=True,
        name="Cadastre (numéros parcelles)",
        attr="IGN-Géoportail — Cadastre",
        overlay=True,
        show=True,       # Activé par défaut
        opacity=0.85,    # Bien visible
    ).add_to(_fmap)

    # Polygone de la parcelle (si trouvée via API IGN ou Filiation)
    if (_parcel_found or _filiation_found) and _geom.get("geojson"):
        _props = _geom["geojson"].get("properties") or {}
        folium.GeoJson(
            _geom["geojson"],
            name="Parcelle identifiée" if _parcel_found else "Nouvelle Parcelle Fille",
            style_function=lambda x: {
                "fillColor":   "#f97316" if _parcel_found else "#8b5cf6", # Violet si issue de filiation
                "color":       "#ea580c" if _parcel_found else "#7c3aed",
                "weight":      4,
                "fillOpacity": 0.30,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[k for k in ["section", "numero", "contenance"] if k in _props],
                aliases=["Section", "Parcelle", "Contenance"][: sum(1 for k in ["section", "numero", "contenance"] if k in _props)],
            )
        ).add_to(_fmap)

    # Marqueur draggable (= future pastille Géofoncier)
    _marker_popup = folium.Popup(
        f"<b>Dossier : {_confirmed.get('ref_dossier', '?')}</b><br>"
        f"Commune : {_map_commune}<br>"
        f"Section : {_map_section_input} — Parcelle : {_map_parcelle_input}<br>"
        f"<em>Cliquez sur la carte pour déplacer la pastille.</em>",
        max_width=280,
    )
    folium.Marker(
        location=_marker_pos,
        popup=_marker_popup,
        tooltip="📍 Future pastille Géofoncier",
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(_fmap)

    # Cercle de contexte (rayon 30m)
    folium.Circle(
        location=_marker_pos,
        radius=30,
        color="#f59e0b",
        fill=True,
        fill_color="#fef3c7",
        fill_opacity=0.15,
        weight=2,
        dash_array="6",
        tooltip="Rayon 30m",
    ).add_to(_fmap)

    # Contrôle des couches (déplié pour visibilité)
    folium.LayerControl(position="topright", collapsed=False).add_to(_fmap)

    # ─── Affichage de la carte ──────────────────────────────────────────────
    # Zoom adaptatif : 18 si parcelle trouvee, 16 si section, 14 si commune
    _zoom_map = 18 if (_parcel_found or _filiation_found) else (16 if _section_approx else 14)

    _fmap = folium.Map(
        location=_map_center,
        zoom_start=_zoom_map,
        control_scale=True,
        tiles=None,
    )

    # FOND 1 : Plan IGN v2
    folium.TileLayer(
        tiles="https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
              "&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2&STYLE=normal&FORMAT=image/png"
              "&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}",
        attr="IGN-Geoportail Plan IGN", name="Plan IGN",
        max_zoom=19, show=True,
    ).add_to(_fmap)

    # FOND 2 : Orthophoto IGN
    folium.TileLayer(
        tiles="https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
              "&LAYER=HR.ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg"
              "&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}",
        attr="IGN-Geoportail Orthophoto", name="Orthophoto IGN",
        max_zoom=21, show=False,
    ).add_to(_fmap)

    folium.TileLayer(tiles="OpenStreetMap", name="Plan OSM", show=False).add_to(_fmap)

    # ─── COUCHES GEOFONCIER (Publiques via API OGE) ────────────────────
    # Emprises des dossiers (polygones)
    folium.WmsTileLayer(
        url="https://api2.geofoncier.fr/api/referentielsoge/wxs?",
        layers="DOSSIERS_EMPRISES",
        fmt="image/png", transparent=True,
        name="Géofoncier (Emprises)",
        attr="Ordre des Géomètres-Experts",
        overlay=True, show=False, opacity=0.6,
        maxNativeZoom=19, maxZoom=22,
    ).add_to(_fmap)

    # Localisants des dossiers (pastilles) - Actif par defaut
    folium.WmsTileLayer(
        url="https://api2.geofoncier.fr/api/referentielsoge/wxs?",
        layers="DOSSIERS_LOCALISANTS",
        fmt="image/png", transparent=True,
        name="Géofoncier (Pastilles)",
        attr="Ordre des Géomètres-Experts",
        overlay=True, show=True, opacity=0.9,
        maxNativeZoom=19, maxZoom=22,
    ).add_to(_fmap)

    # OVERLAY CADASTRE : numeros de parcelles actifs par defaut
    folium.WmsTileLayer(
        url="https://data.geopf.fr/wms-r/wms?",
        layers="CADASTRALPARCELS.PARCELLAIRE_EXPRESS",
        fmt="image/png", transparent=True,
        name="Cadastre IGN (numeros parcelles)",
        attr="IGN-Geoportail Cadastre",
        overlay=True, show=True, opacity=0.85,
        maxNativeZoom=18, maxZoom=22,
    ).add_to(_fmap)

    # Polygone orange : parcelle exacte si trouvee dans IGN (violet si filiation)
    if (_parcel_found or _filiation_found) and _geom.get("geojson"):
        _props = _geom["geojson"].get("properties") or {}
        _tt_fields  = [k for k in ["section", "numero", "contenance"] if k in _props]
        _tt_aliases = ["Section", "Parcelle", "Surface m2"][:len(_tt_fields)]
        folium.GeoJson(
            _geom["geojson"],
            name="Parcelle cadastrale identifiee",
            style_function=lambda x: {
                "fillColor": "#f97316" if _parcel_found else "#8b5cf6",
                "color": "#ea580c" if _parcel_found else "#7c3aed",
                "weight": 4, "fillOpacity": 0.35,
            },
            tooltip=folium.GeoJsonTooltip(fields=_tt_fields, aliases=_tt_aliases)
        ).add_to(_fmap)

    # Marqueur rouge = future pastille Geofoncier
    # Position : clic utilisateur si deja clique, sinon centroide IGN
    folium.Marker(
        location=_marker_pos,
        popup=folium.Popup(
            f"<b>Dossier {_confirmed.get('ref_dossier', '?')}</b><br>"
            f"Commune : {_map_commune}<br>"
            f"Section : {_sec_clean} - Parcelle : {_num_clean}<br>"
            f"<i>Cliquez sur la carte pour repositionner.</i>",
            max_width=260,
        ),
        tooltip="Pastille Geofoncier — cliquez sur la carte pour deplacer",
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(_fmap)

    folium.Circle(
        location=_marker_pos, radius=25,
        color="#f59e0b", fill=True, fill_color="#fef3c7",
        fill_opacity=0.2, weight=2, dash_array="5",
    ).add_to(_fmap)

    folium.LayerControl(position="topright", collapsed=False).add_to(_fmap)

    # ─── Rendu st_folium ──────────────────────────────────────────────────────
    # La cle inclut _marker_pos pour forcer le rerendu quand la position change.
    _map_key = f"fmap_{page_id}_{_sec_clean}_{_num_clean}_{str(_marker_pos)[:25]}"

    if "show_plan_comparatif" not in st.session_state:
        st.session_state.show_plan_comparatif = False

    _btn_label = "Masquer le plan d'époque" if st.session_state.show_plan_comparatif else "Comparer avec le plan d'époque"
    if st.button(_btn_label):
        st.session_state.show_plan_comparatif = not st.session_state.show_plan_comparatif
        st.rerun()

    if st.session_state.show_plan_comparatif and 'img_base' in locals() and img_base:
        col_map, col_img = st.columns([1, 1], gap="medium")
    else:
        col_map = st.container()
        col_img = None

    with col_map:
        _map_output = st_folium(
            _fmap,
            width="100%",
            height=520,
            returned_objects=["last_clicked"],
            key=_map_key,
        )
        
    if col_img is not None:
        with col_img:
            st.image(img_base, use_container_width=True)

    # ─── Mise a jour position apres clic utilisateur ─────────────────────────
    # Quand l'utilisateur clique sur une parcelle, last_clicked est mis a jour.
    # On sauvegarde le clic → rerun → marqueur se deplace a la nouvelle position.
    if _map_output and _map_output.get("last_clicked"):
        _lc = _map_output["last_clicked"]
        _lat_c, _lng_c = _lc.get("lat"), _lc.get("lng")
        if _lat_c is not None and _lng_c is not None:
            _new_pos = [_lat_c, _lng_c]
            if st.session_state.get(_click_key) != _new_pos:
                st.session_state[_click_key] = _new_pos
                st.rerun()

    # ─── Boutons de confirmation / reinitialisation ───────────────────────────
    st.caption(
        "Cliquez sur la carte pour positionner la pastille sur la bonne parcelle. "
        "Puis cliquez Confirmer pour passer au versement."
    )
    def _on_confirm_map():
        _pos_finale = st.session_state.get(_click_key, _marker_pos)
        st.session_state[_map_confirmed_key] = True
        st.session_state[_map_coords_key]    = _pos_finale
        
        _final_sec = _sec_clean
        _final_num = _num_clean
        
        # Récupérer la parcelle exacte sous le marqueur placé par l'opérateur
        import geofoncier_api
        _clicked_parcel = geofoncier_api.get_parcel_by_coordinates(_pos_finale[0], _pos_finale[1])
        
        if _clicked_parcel and _clicked_parcel.get("section") and _clicked_parcel.get("numero"):
            _final_sec = _clicked_parcel["section"].zfill(2)
            _final_num = _clicked_parcel["numero"].zfill(4)
            if _clicked_parcel.get("code_insee"):
                _confirmed["code_insee_ign"] = _clicked_parcel["code_insee"]
        elif _filiation_found and _geom and _geom.get("filles_filiation"):
            # Fallback sur la première fille de la filiation
            _fille = _geom["filles_filiation"][0]
            _final_sec = _fille["section"].zfill(2)
            _final_num = _fille["numero"]
        
        st.session_state[f"lu_section_{page_id}"] = _final_sec
        _confirmed["section_excel"] = _final_sec
        _confirmed["parcelle_excel"] = _final_num
        
        st.session_state[_unified_confirmed_key] = _confirmed

    _col_btn1, _col_btn2 = st.columns([1, 1])
    with _col_btn1:
        st.button(
            "Confirmer cette localisation",
            type="primary",
            key=f"btn_map_confirm_{page_id}",
            use_container_width=True,
            on_click=_on_confirm_map
        )

    with _col_btn2:
        if st.button(
            "Reinitialiser la localisation",
            key=f"btn_map_reset_{page_id}",
            use_container_width=True,
        ):
            for _k in [_map_confirmed_key, _map_coords_key, _geom_cache_key,
                        _click_key, f"_geo_status_{page_id}"]:
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()

    if st.session_state.get(_map_confirmed_key):
        _pos_conf = st.session_state.get(_map_coords_key, _marker_pos)
        st.success(
            f"Localisation confirmee : {_pos_conf[0]:.5f}N, {_pos_conf[1]:.5f}E "
            f"- Section {_sec_clean}, Parcelle {_num_clean}"
        )
    else:
        st.markdown(
            "<div style='background:#fef3c7;border-left:4px solid #f59e0b;"
            "padding:10px 16px;border-radius:8px;font-size:0.9rem;'>"
            "<b>Action requise</b> : Verifiez la pastille rouge, cliquez sur la"
            " bonne parcelle si besoin, puis confirmez ci-dessus.</div>",
            unsafe_allow_html=True
        )

    _map_validated = st.session_state.get(_map_confirmed_key, False)
    # La carte est optionnelle si la parcelle a été trouvée et validée via IGN
    _map_optional = _parcel_found or _filiation_found
    if _confirmed and (_map_validated or _map_optional):
        st.markdown("---")
        st.markdown("### Étape 3 — Versement sur Géofoncier")

        # Récupération des informations finales
        _ref_dossier  = _confirmed.get("ref_dossier", "")
        _op_gf        = _confirmed.get("op_code_gf", "")
        _op_excel     = _confirmed.get("op_code_excel", "")
        _annee_full   = _confirmed.get("annee_full", None)
        _section_final = _map_section_input.strip().upper() if _map_section_input.strip() else _confirmed.get("section_excel", _fb_section)
        _parcelle_final = _map_parcelle_input.strip() if _map_parcelle_input.strip() else _confirmed.get("parcelle_excel", None)

        # ── AUTO-REMPLACEMENT PAR LA FILIATION ──────────────────────
        if _filiation_found and _geom.get("filles_filiation"):
            _filles = _geom["filles_filiation"]
            _section_final = _filles[0]['section']
            _parcelle_final = ", ".join([f['numero'] for f in _filles if f['section'] == _section_final])

        # Code INSEE : priorité à la localisation IGN (clic carte), sinon résolution textuelle
        _code_insee_auto = _confirmed.get("code_insee_ign")
        if not _code_insee_auto:
            _, _code_insee_auto = get_insee_from_commune(_map_commune)

        # Récap des identifiants (Adaptation dynamique selon le géomètre validé)
        import geofoncier_api
        _cab_createur_label = _geometre_val
        _cab_createur_code  = geofoncier_api.CABINETS_CREATEURS.get(_geometre_val) or geofoncier_api.ENR_CAB_DETENTEUR

        # Affichage récapitulatif
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1.5rem;margin-bottom:1rem;">
        <h4 style="margin:0 0 1rem;color:#1e293b;border-bottom:2px solid #e2e8f0;padding-bottom:0.5rem;">
            Récapitulatif du dossier à créer
        </h4>
        """, unsafe_allow_html=True)

        _cols_recap = st.columns(2)
        with _cols_recap[0]:
            st.markdown(f"""
            | Champ | Valeur |
            |---|---|
            | **Cabinet créateur** | {_cab_createur_label} (`{_cab_createur_code}`) |
            | **Cabinet détenteur** | GEO-SIAPP (`1992C100001`) |
            | **GE créateur** | Lionnel Robert (`05141`) |
            | **Référence dossier** | `{_ref_dossier}` |
            """)
        with _cols_recap[1]:
            st.markdown(f"""
            | Champ | Valeur |
            |---|---|
            | **Code INSEE** | `{_code_insee_auto or 'Non trouvé'}` |
            | **Commune** | {_map_commune} |
            | **Section** | {_section_final} |
            | **Parcelle** | {_parcelle_final or '—'} |
            | **Opération** | {_op_excel} → `{_op_gf}` |
            """)

        st.markdown("</div>", unsafe_allow_html=True)

        # Date dossier (modifiable)
        _date_csv = str(row.get("Date","")).strip()
        _date_iso_auto = format_date_iso(_date_csv, _annee_full)
    
        # Forcer la cohérence avec le répertoire certifié
        _date_repertoire = _confirmed.get("date_cadastre")
        if _date_repertoire and str(_date_repertoire).strip() not in ("nan", "None", ""):
            # Si le répertoire contient une date complète (ex: Racat/Ceyte)
            _date_iso_auto = format_date_iso(str(_date_repertoire), _annee_full)
        elif _annee_full and not _date_iso_auto.startswith(str(_annee_full)):
            # Si l'OCR a sorti une date abracadabrante (mauvaise année) on force le 1er janvier de l'année certifiée
            _date_iso_auto = f"{_annee_full:04d}-01-01"
        
        _final_date_iso = st.text_input(
            "Date du dossier (format AAAA-MM-JJ)",
            value=_date_iso_auto,
            key=f"lu_date_iso_{page_id}",
            help="Priorité : Date exacte du répertoire > 1er Janvier de l'année du répertoire > Date lue par l'OCR."
        )

        # Code opération (modifiable si vide)
        if not _op_gf:
            _final_op_gf = st.text_input(
                "Code opération Géofoncier (obligatoire)",
                value="Em",  # Valeur par défaut (Modification du parcellaire cadastral)
                key=f"lu_op_gf_{page_id}",
                help="Ex: Em=Modification du parcellaire cadastral, Ec=Division, Eb=Bornage..."
            )
        else:
            _final_op_gf = _op_gf
            st.info(f"Code opération Géofoncier : **{_op_excel}** → `{_op_gf}`")
        
        _final_statut_gf = st.selectbox(
            "Statut du dossier sur Géofoncier",
            ["Achevé", "Indéterminé", "En cours", "Annulé", "Archivé"],
            index=0,
            key=f"lu_statut_gf_{page_id}",
            help="Le statut 'Achevé' est attendu par défaut pour ce type d'archive."
        )

        # Vérification IGN
        _ign_ok = False
        if _code_insee_auto and _section_final and _parcelle_final:
            with st.spinner("Vérification IGN de la parcelle..."):
                _ign_ok = verify_parcel_ign(_code_insee_auto, _section_final, str(_parcelle_final))
            if _ign_ok:
                st.success(f"Parcelle **{_section_final}-{_parcelle_final}** trouvée sur l'IGN (commune {_code_insee_auto}).")
            else:
                st.warning(
                    f"Parcelle **{_section_final}-{_parcelle_final}** non trouvée sur l'IGN (commune {_code_insee_auto}). "
                    "Vérifiez la section et la parcelle. Vous pouvez quand même verser si vous êtes certain(e).",

                )
        else:
            if not _code_insee_auto:
                st.error("Code INSEE introuvable — le versement ne peut pas continuer sans code INSEE valide.")
            else:
                st.warning("Parcelle ou section manquante — la vérification IGN est ignorée.")

        # Bouton désactivé si code INSEE manquant ou code opération manquant
        _can_submit = bool(_code_insee_auto and _final_op_gf and _ref_dossier)
        if not _can_submit:
            _missing = []
            if not _code_insee_auto: _missing.append("Code INSEE")
            if not _final_op_gf:        _missing.append("Code opération GF")
            if not _ref_dossier:     _missing.append("Référence dossier")
            st.error(f"Impossible de verser — informations manquantes : {', '.join(_missing)}")

        _col_vers, _col_cancel = st.columns([1, 1])
        with _col_vers:
            _btn_vers = st.button(
                "Créer le dossier sur Géofoncier",
                type="primary",
                disabled=not _can_submit,
                key=f"btn_vers_{page_id}",
                use_container_width=True
            )
        with _col_cancel:
            if st.button("Annuler / Changer de dossier", key=f"btn_cancel_vers_{page_id}", use_container_width=True):
                # Nettoyer toutes les clés de confirmation pour forcer le retour à l'étape 1
                _keys_to_clear_cancel = [
                    _unified_confirmed_key,
                    f"_lookup_result_{base_name}_{page_id}",
                    f"_lookup_confirmed_{base_name}_{page_id}",
                    f"_rc_lookup_result_{base_name}_{page_id}",
                    f"_rc_lookup_confirmed_{base_name}_{page_id}",
                    f"_dp_lookup_result_{base_name}_{page_id}",
                    f"_dp_lookup_confirmed_{base_name}_{page_id}",
                    f"_ad_lookup_confirmed_{base_name}_{page_id}",
                    _map_confirmed_key,
                    _map_coords_key,
                    _click_key,
                ]
                for _k in _keys_to_clear_cancel:
                    if _k in st.session_state:
                        del st.session_state[_k]
                st.rerun()

        # ── Exécution du versement ────────────────────────────────────
        if _btn_vers and _can_submit:
            # Construction du payload dict
            _metadata_vers = {
                "geometre":     _geometre_val,
                "ref_dossier":  _ref_dossier,
                "n_ordre":      str(row.get("N_Ordre", "")).strip(),
                "commune":      _map_commune,
                "code_insee":   _code_insee_auto,
                "section":      _section_final,
                "parcelles":    [_parcelle_final] if _parcelle_final else [],
                "annee_full":   _annee_full,
                "date_dossier": _final_date_iso,
                "op_codes_gf":  [_final_op_gf] if _final_op_gf else [],
                "enr_statut":   _final_statut_gf,
            }

            with st.spinner("Création du dossier en cours..."):
                # Passer les coordonnées GPS du clic carte pour le localisant Lambert93
                _pos_finale = st.session_state.get(_map_coords_key, None) or st.session_state.get(_click_key, None)
                _metadata_vers["lat_lon"] = _pos_finale  # [lat, lon] ou None
                # Passer la nature de l'acte pour résolution du doc_code
                _metadata_vers["nature_acte"] = str(row.get("Nature_Acte_Geofoncier", "AUTRE")).strip()
                try:
                    _vers_result = create_geofoncier_dossier(_metadata_vers)
                except RuntimeError as _token_err:
                    # Le token a expiré ou les identifiants .env sont invalides
                    _vers_result = {
                        "success": False,
                        "error_code": "TOKEN",
                        "error_msg": (
                            f"Impossible de s'authentifier sur Géofoncier : {_token_err}. "
                            "Vérifiez GEOFONCIER_LOGIN et GEOFONCIER_PASSWORD dans le fichier .env."
                        )
                    }
                except Exception as _api_err:
                    _vers_result = {
                        "success": False,
                        "error_code": "EXCEPTION",
                        "error_msg": f"Erreur inattendue lors du versement : {_api_err}"
                    }

            if _vers_result.get("success"):
                _id_doss = _vers_result.get("id_dossier", "")
                st.success(f"Dossier créé avec succès sur Géofoncier ! ID : **{_id_doss}**")

                # Mise à jour du statut dans le CSV
                _idx_row = df[df[id_col] == page_id].index[0]
                import datetime as _dt
                df.at[_idx_row, "Confirmation_Status"] = f"Versé sur Géofoncier ({_dt.datetime.now().strftime('%d/%m/%Y %H:%M')})"
                df.to_csv(fichier_choisi, sep=";", index=False, encoding="utf-8-sig")
                st.cache_data.clear()

                # Upload du document source
                # Recherche insensible à la casse (Windows/WSL) pour éviter les échecs
                # liés aux extensions en majuscules (ex: .PDF au lieu de .pdf)
                _doc_path = None
                for _ext in [".pdf", ".PDF", ".jpg", ".JPG", ".png", ".PNG", ".tif", ".TIF", ".tiff", ".TIFF"]:
                    _p = os.path.join(INPUTS_DIR, f"{base_name}{_ext}")
                    if os.path.exists(_p):
                        _doc_path = _p
                        break
                # Fallback glob insensible à la casse (cherche n'importe quelle extension)
                if not _doc_path:
                    _glob_candidates = glob.glob(os.path.join(INPUTS_DIR, f"{base_name}.*"))
                    if _glob_candidates:
                        # Prioriser les PDF
                        _glob_pdfs = [g for g in _glob_candidates if g.lower().endswith(".pdf")]
                        _doc_path = _glob_pdfs[0] if _glob_pdfs else _glob_candidates[0]

                if _doc_path and _id_doss:
                    with st.spinner("Upload du document PDF..."):
                        _nature_acte = str(row.get("Nature_Acte_Geofoncier", "AUTRE")).strip()
                        _up_result = upload_document_to_dossier(
                            _id_doss, _doc_path,
                            doc_description=f"Archive {_ref_dossier} — {_map_commune}",
                            nature_acte=_nature_acte
                        )
                    if _up_result.get("success"):
                        st.success("Document PDF uploadé avec succès.")
                    else:
                        st.warning(f"Le dossier est créé, mais l'upload du PDF a échoué : {_up_result.get('error_msg','')}")
                elif not _doc_path:
                    st.warning(f"Document PDF source introuvable dans {INPUTS_DIR}/ pour {base_name}.*")
            else:
                st.error(
                    f"Échec de la création du dossier : "
                    f"({_vers_result.get('error_code','?')}) {_vers_result.get('error_msg','Erreur inconnue.')}"
                )
                if _vers_result.get("payload"):
                    with st.expander("Détail du payload envoyé (diagnostic)"):
                        st.json(_vers_result["payload"])

