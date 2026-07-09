import re
import sys

def main():
    with open('app_validation.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find start of "if _confirmed:" block inside HARROIS block
    start_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('        if _confirmed:') and i > 1100:
            start_idx = i
            break
    
    if start_idx == -1:
        print("Could not find 'if _confirmed:' block")
        return

    # Find end of block
    end_idx = -1
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith('        '):
            if line.startswith('    elif _geometre_val in {"RACAT", "CEYTE"}:') or line.startswith('elif _geometre_val in {"RACAT", "CEYTE"}:'):
                end_idx = i
                break

    if end_idx == -1:
        print("Could not find end of 'if _confirmed:' block")
        return

    print(f"Extracting lines {start_idx} to {end_idx-1}")
    
    map_block = lines[start_idx:end_idx]
    
    # We remove the old map block from the HARROIS section
    new_lines = lines[:start_idx] + ["\n"] + lines[end_idx:]
    
    # Replace API_DIRECT versement with confirmation
    ad_button_str = '        if st.button("Créer le dossier sur Géofoncier" if not _ad_dry else "Simuler (Dry Run)",'
    for i in range(len(new_lines)):
        if new_lines[i].startswith(ad_button_str):
            # We replace lines i to i+39 (approx) with the confirmation logic
            end_ad_idx = i
            for j in range(i, min(i+50, len(new_lines))):
                if new_lines[j].startswith('else:'):
                    end_ad_idx = j
                    break
            
            replacement = [
                '        if st.button("Confirmer cette référence", type="primary", disabled=not _ad_ref, key=f"ad_btn_conf_{page_id}"):\n',
                '            st.session_state[f"_ad_lookup_confirmed_{base_name}_{page_id}"] = {\n',
                '                "ref_dossier": _ad_ref.strip(),\n',
                '                "commune_excel": _ad_commune,\n',
                '                "section_excel": _ad_section,\n',
                '                "parcelle_excel": _ad_parcelle,\n',
                '                "date_cadastre": _ad_date_iso,\n',
                '                "op_code_gf": _ad_op.strip(),\n',
                '                "op_code_excel": _ad_op.strip(),\n',
                '                "annee_full": int(_ad_date_iso[:4]) if _ad_date_iso and len(_ad_date_iso)>=4 else None,\n',
                '                "enr_statut": _ad_statut\n',
                '            }\n',
                '            st.rerun()\n'
            ]
            new_lines = new_lines[:i] + replacement + new_lines[end_ad_idx:]
            print("Replaced API DIRECT block.")
            break

    # Dedent the map block by 8 spaces
    dedented_block = []
    for line in map_block:
        if line.startswith('        '):
            dedented_block.append(line[8:])
        elif line == '\n':
            dedented_block.append(line)
        else:
            dedented_block.append(line.lstrip())

    # Create the unification header
    unification_header = """
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
        _unified_dossier["op_code_gf"] = "Da"
        if _unified_dossier.get("annee"):
            _unified_dossier["date_cadastre"] = f"{_unified_dossier['annee']}-01-01"
    elif _geometre_val in _GEOMETRES_API_DIRECT and st.session_state.get(f"_ad_lookup_confirmed_{base_name}_{page_id}"):
        _unified_dossier = st.session_state.get(f"_ad_lookup_confirmed_{base_name}_{page_id}")

if _unified_dossier:
    _confirmed = _unified_dossier
"""

    # We append the unification block and the dedented map block to the end of the file
    new_lines.append("\n" + unification_header)
    new_lines.extend(dedented_block)

    with open('app_validation.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("Refactoring completed successfully.")

if __name__ == '__main__':
    main()
