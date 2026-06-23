"""
Patch 2: Dans _extract_with_vlm, utiliser display_zone (si disponible) 
comme zone d'affichage dans le résultat, à la place de la grande zone de crop.
"""
path = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py'

with open(path, encoding='utf-8') as f:
    content = f.read()

# La zone z est la zone de crop (grande), on veut utiliser display_zone pour l'affichage
old = '''                        if final_val:
                            res_dict[field] = {
                                "valeur": final_val, "zone": z,
                                "brut": crop_info.get("brut", "") + " -> " + raw,
                                "methode": f"vlm_crop_{model_name.split(':')[0]}",
                                "confidence": 0.92 if model_name == "llama3.2-vision" else 0.90,
                            }'''

new = '''                        if final_val:
                            # Utiliser display_zone si disponible (zone plus petite, centrée sur le label)
                            # Sinon, fallback sur la zone de crop
                            display_z = crop_info.get("display_zone", z)
                            res_dict[field] = {
                                "valeur": final_val, "zone": display_z,
                                "brut": crop_info.get("brut", "") + " -> " + raw,
                                "methode": f"vlm_crop_{model_name.split(':')[0]}",
                                "confidence": 0.92 if model_name == "llama3.2-vision" else 0.90,
                            }'''

if old in content:
    content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patch 2 appliqué : display_zone utilisé pour l'affichage VLM")
else:
    print("ERREUR: Cible non trouvée. Vérifier le contenu du fichier.")
    idx = content.find('res_dict[field] = {')
    print(f"Contexte autour de res_dict: {repr(content[idx-100:idx+300])}")
