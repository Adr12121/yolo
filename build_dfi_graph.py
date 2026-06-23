import os
import json
from collections import defaultdict
import time

def build_graph(txt_path, out_path):
    print(f"Lecture du fichier {txt_path}...")
    start_time = time.time()
    
    # Structure pour regrouper les lignes par document et sequence
    # {(commune, document, sequence): {"meres": [], "filles": []}}
    operations = defaultdict(lambda: {"meres": [], "filles": []})
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 11:
                continue
                
            dept = parts[0]
            commune_code = parts[1]
            doc_num = parts[3]
            seq_num = parts[8]
            type_ligne = parts[9] # '1' = meres, '2' = filles
            
            # Reconstruction code INSEE
            if dept.endswith('0') and not dept.startswith('97'):
                insee = dept[:2] + commune_code
            else:
                insee = dept + commune_code
            
            # Lecture des parcelles
            parcelles = []
            for p in parts[10:]:
                if not p.strip(): continue
                # Format: " A0092" -> section (lettres) + numero (chiffres)
                # Le dernier caractère non-chiffre sépare la section du numéro
                p_clean = p.strip()
                import re
                match = re.match(r"^([A-Z]+)\s*(\d+)$", p_clean, re.IGNORECASE)
                if match:
                    section = match.group(1).upper()
                    numero = str(match.group(2)).zfill(4)
                else:
                    # Fallback au cas où
                    digits = "".join([c for c in p_clean if c.isdigit()])
                    letters = "".join([c for c in p_clean if c.isalpha()]).upper()
                    if digits:
                        section = letters
                        numero = digits.zfill(4)
                    else:
                        continue
                
                parcelles.append({
                    "code_commune": insee,
                    "section": section,
                    "numero": numero
                })
                    
            key = (insee, doc_num, seq_num)
            if type_ligne == '1':
                operations[key]["meres"].extend(parcelles)
            elif type_ligne == '2':
                operations[key]["filles"].extend(parcelles)

    print(f"{len(operations)} opérations trouvées. Construction du graphe...")
    
    # Graphe final: {"insee_section_numero": [{"code_commune":..., "section":..., "numero":...}, ...]}
    graph = defaultdict(list)
    
    for key, data in operations.items():
        meres = data["meres"]
        filles = data["filles"]
        
        for mere in meres:
            mere_id = f"{mere['code_commune']}_{mere['section']}_{mere['numero']}"
            for fille in filles:
                if fille not in graph[mere_id]:
                    graph[mere_id].append(fille)

    print(f"Graphe construit avec {len(graph)} parcelles mères uniques. Sauvegarde dans {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(graph, f, ensure_ascii=False)
        
    end_time = time.time()
    print(f"Terminé en {end_time - start_time:.2f} secondes. Taille du fichier: {os.path.getsize(out_path)/1024/1024:.2f} MB")

if __name__ == "__main__":
    txt_file = "Filiation_parcelles/dfiano-dep070-03012026.txt"
    out_file = "dfi_07.json"
    
    if os.path.exists(txt_file):
        build_graph(txt_file, out_file)
    else:
        print(f"Fichier introuvable: {txt_file}")
