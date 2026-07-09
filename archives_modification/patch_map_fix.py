"""
Patch complet pour corriger la carte interactive de localisation (app_validation.py).

Problemes corriges :
1. L'API IGN fonctionne (AL+0160 trouve bien la parcelle). Le bug vient du
   fait que parcelle_excel peut etre None/vide dans _confirmed, et que la
   section peut mal se formater.
2. Le marqueur se positionnait toujours au meme endroit (commune) car la
   logique de fallback s'appliquait a tous les docs.
3. Ajout du mecanisme clic-sur-carte pour repositionner la pastille.
4. Suppression du _saved_coords (les coords confirmes ne doivent pas
   pre-positionner le marqueur d un autre document).
"""

TARGET_START = '_geom_cache_key = f"_geom_{page_id}_{_map_section_input}_{_map_parcelle_input}_{_map_insee}"'
TARGET_END   = '                    "La carte est centr\u00e9e sur la commune. Placez le marqueur manuellement."\n                )\n'

NEW_BLOCK = '''\
            # ─── Localisation en 3 niveaux : parcelle exacte → section → commune ───
            # L'API IGN apicarto cherche la parcelle avec le numero actuel.
            # Format attendu : section 2 cars (ex "AL"), numero 4 chiffres (ex "0160")
            # Si la parcelle n'est plus dans IGN (fusionnee, renumerotee),
            # on cherche n'importe quelle parcelle de la meme section pour
            # centrer la carte dans le bon quartier de la commune.
            _geom_cache_key = f"_geom_v2_{page_id}_{_map_section_input}_{_map_parcelle_input}_{_map_insee}"
            if _geom_cache_key not in st.session_state or _btn_refresh_map:
                _geom = None
                _geo_status = "commune"  # "parcel" | "section" | "commune"

                # --- Formatage robuste section / numero ---
                _sec_clean = str(_map_section_input or "").strip().upper()
                _num_raw   = str(_map_parcelle_input or "").strip()
                # Garder uniquement les chiffres pour le numero
                _num_digits = "".join(c for c in _num_raw if c.isdigit())
                _num_clean  = _num_digits.zfill(4) if _num_digits else ""

                # Etape 1 : parcelle exacte (API IGN apicarto)
                if _sec_clean and _num_clean and _map_insee:
                    with st.spinner(f"Recherche parcelle {_sec_clean}-{_num_clean} sur IGN..."):
                        _geom = get_parcel_geometry(_map_insee, _sec_clean, _num_clean)
                    if _geom and _geom.get("found"):
                        _geo_status = "parcel"

                # Etape 2 : centroide de n'importe quelle parcelle de la section
                if _geo_status != "parcel" and _sec_clean and _map_insee:
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
                                        if _sg.get("type") == "Polygon" and _sc:
                                            _ring = _sc[0]
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
            elif _section_approx:
                st.info(
                    f"Parcelle **{_sec_clean}-{_num_clean}** introuvable dans IGN "
                    f"(peut etre renumerotee depuis l'archive). "
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
'''

with open('app_validation.py', encoding='utf-8') as f:
    content = f.read()

# Trouver le debut et la fin du bloc a remplacer
idx_start = content.find('            _geom_cache_key = f"_geom_{page_id}_{_map_section_input}_{_map_parcelle_input}_{_map_insee}"')
if idx_start == -1:
    # Peut etre que c'est la version avec v2
    idx_start = content.find('            _geom_cache_key = f"_geom_v2_{page_id}')
    print(f"[INFO] Bloc v2 trouve a idx={idx_start}")

if idx_start == -1:
    print("ERREUR: debut du bloc non trouve")
    exit(1)

# Trouver la fin : la ligne qui contient la fin du bandeau de statut
# On cherche la ligne avec "Bandeau de statut" puis le prochain bloc apres
# La fin du bloc est avant "# Construction de la carte"
search_end = '            # \u2500\u2500 Construction de la carte Folium \u2500\u2500'
idx_end = content.find(search_end, idx_start)
if idx_end == -1:
    search_end = '            # \u2500\u2500 Construction de la carte'
    idx_end = content.find(search_end, idx_start)

if idx_end == -1:
    print("ERREUR: fin du bloc non trouvee")
    print(f"Contenu autour de idx_start ({idx_start}):")
    print(content[idx_start:idx_start+500])
    exit(1)

print(f"[OK] Remplacement lignes {content[:idx_start].count(chr(10))+1} "
      f"a {content[:idx_end].count(chr(10))+1}")

# Remplacement
content_new = content[:idx_start] + NEW_BLOCK + '\n' + content[idx_end:]

with open('app_validation.py', 'w', encoding='utf-8') as f:
    f.write(content_new)
print("PATCH APPLIQUE avec succes")
