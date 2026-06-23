"""
Patch 2 v2 : Corrige le bloc d'affichage de la carte en utilisant
une ancre robuste basee sur du code ASCII pur.
"""

TARGET_START_ANCHOR = 'st.markdown(\n                "<div style=\'border:2px solid #e2e8f0;border-radius:12px;overflow:hidden;margin-bottom:1rem;\'>",'
TARGET_END_ANCHOR   = '        _map_validated = st.session_state.get(_map_confirmed_key, False)'

NEW_DISPLAY_BLOCK = '''\
            # ─── Affichage de la carte ──────────────────────────────────────────────
            # Zoom adaptatif : 18 si parcelle trouvee, 16 si section, 14 si commune
            _zoom_map = 18 if _parcel_found else (16 if _section_approx else 14)

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

            # OVERLAY CADASTRE : numeros de parcelles actifs par defaut
            folium.WmsTileLayer(
                url="https://data.geopf.fr/wms-r/wms?",
                layers="CADASTRALPARCELS.PARCELLAIRE_EXPRESS",
                fmt="image/png", transparent=True,
                name="Cadastre IGN (numeros parcelles)",
                attr="IGN-Geoportail Cadastre",
                overlay=True, show=True, opacity=0.85,
            ).add_to(_fmap)

            # Polygone orange : parcelle exacte si trouvee dans IGN
            if _parcel_found and _geom.get("geojson"):
                _props = _geom["geojson"].get("properties") or {}
                _tt_fields  = [k for k in ["section", "numero", "contenance"] if k in _props]
                _tt_aliases = ["Section", "Parcelle", "Surface m2"][:len(_tt_fields)]
                folium.GeoJson(
                    _geom["geojson"],
                    name="Parcelle cadastrale identifiee",
                    style_function=lambda x: {
                        "fillColor": "#f97316", "color": "#ea580c",
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
                draggable=True,
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

            st.markdown(
                "<div style='border:2px solid #3b82f6;border-radius:12px;"
                "overflow:hidden;margin-bottom:0.5rem;'>",
                unsafe_allow_html=True
            )
            _map_output = st_folium(
                _fmap,
                width="100%",
                height=520,
                returned_objects=["last_clicked"],
                key=_map_key,
            )
            st.markdown("</div>", unsafe_allow_html=True)

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
            _col_btn1, _col_btn2 = st.columns([1, 1])
            with _col_btn1:
                if st.button(
                    "Confirmer cette localisation",
                    type="primary",
                    key=f"btn_map_confirm_{page_id}",
                    use_container_width=True,
                ):
                    _pos_finale = st.session_state.get(_click_key, _marker_pos)
                    st.session_state[_map_confirmed_key] = True
                    st.session_state[_map_coords_key]    = _pos_finale
                    if _sec_clean != (_map_section or ""):
                        st.session_state[f"lu_section_{page_id}"] = _sec_clean
                    st.rerun()

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

'''

with open('app_validation.py', encoding='utf-8') as f:
    content = f.read()

# Trouver le debut : le st.markdown("<div style='border:2px...") de la carte
# Chercher en remontant depuis "Affichage Streamlit"
idx_aff = content.find('Affichage Streamlit')
# Trouver le debut de la ligne precedente (le commentaire)
idx_start = content.rfind('\n', 0, idx_aff) + 1  # Debut de la ligne du commentaire

# Trouver la fin : la ligne _map_validated
idx_end = content.find('        _map_validated = st.session_state.get(_map_confirmed_key, False)')

if idx_start <= 0 or idx_end == -1:
    print(f"ERREUR: idx_start={idx_start}, idx_end={idx_end}")
    exit(1)

line_start = content[:idx_start].count('\n') + 1
line_end   = content[:idx_end].count('\n') + 1
print(f"[OK] Remplacement lignes {line_start} a {line_end}")

content_new = content[:idx_start] + NEW_DISPLAY_BLOCK + content[idx_end:]

with open('app_validation.py', 'w', encoding='utf-8') as f:
    f.write(content_new)
print("PATCH 2 APPLIQUE")
