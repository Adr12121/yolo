"""
Patch 4 : Deplacer _sec_clean et _num_clean hors du if de cache
"""

with open('app_validation.py', encoding='utf-8') as f:
    content = f.read()

OLD_BLOCK = '''\
            _geom_cache_key = f"_geom_v2_{page_id}_{_map_section_input}_{_map_parcelle_input}_{_map_insee}"
            if _geom_cache_key not in st.session_state or _btn_refresh_map:
                _geom = None
                _geo_status = "commune"  # "parcel" | "section" | "commune"

                # --- Formatage robuste section / numero ---
                _sec_clean = str(_map_section_input or "").strip().upper().zfill(2)
                _num_raw   = str(_map_parcelle_input or "").strip()
                # Garder uniquement les chiffres pour le numero
                _num_digits = "".join(c for c in _num_raw if c.isdigit())
                _num_clean  = _num_digits.zfill(4) if _num_digits else ""
'''

NEW_BLOCK = '''\
            # --- Formatage robuste section / numero (HORS CACHE pour reruns) ---
            _sec_clean = str(_map_section_input or "").strip().upper().zfill(2)
            _num_raw   = str(_map_parcelle_input or "").strip()
            _num_digits = "".join(c for c in _num_raw if c.isdigit())
            _num_clean  = _num_digits.zfill(4) if _num_digits else ""

            _geom_cache_key = f"_geom_v2_{page_id}_{_map_section_input}_{_map_parcelle_input}_{_map_insee}"
            if _geom_cache_key not in st.session_state or _btn_refresh_map:
                _geom = None
                _geo_status = "commune"  # "parcel" | "section" | "commune"
'''

if OLD_BLOCK in content:
    content_new = content.replace(OLD_BLOCK, NEW_BLOCK)
    with open('app_validation.py', 'w', encoding='utf-8') as f:
        f.write(content_new)
    print("PATCH 4 OK : Variables deplacees avec succes.")
else:
    print("ERREUR : Ancre non trouvee pour patch 4.")
