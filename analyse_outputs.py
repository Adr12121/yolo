import json, os, glob

files = sorted(glob.glob('outputs/*_plan_resultats.csv'))
sep = ";"

print("=" * 90)
print(f"{'FICHIER':<50} {'COMMUNE':<20} {'N_ORDRE':<10} {'GEO':<20} {'SECTION'}")
print("=" * 90)

issues = []

for f in files:
    with open(f, encoding='utf-8-sig', errors='replace') as fp:
        lines = fp.read().splitlines()
    if len(lines) < 2:
        continue
    headers = lines[0].split(sep)
    def col(row, name):
        try:
            return row[headers.index(name)] if name in headers else ''
        except:
            return ''
    for line in lines[1:]:
        row = line.split(sep)
        fname = os.path.basename(f).replace('_plan_resultats.csv','')
        commune = col(row,'Commune')
        n_ordre = col(row,'N_Ordre')
        geometre = col(row,'Geometre')
        section = col(row,'Section')
        nature = col(row,'Nature_Acte_Geofoncier')
        n_dossier = col(row,'N_Dossier')
        print(f"{fname:<50} {commune:<20} {n_ordre:<10} {geometre:<20} {section}")
        # Detect problems
        if not commune or commune in ('None',''):
            issues.append(f"[ERREUR] {fname}: commune vide")
        if not n_ordre or n_ordre in ('None','') and 'PLa' in fname or 'dmpc' in fname.lower():
            if not n_dossier or n_dossier in ('None',''):
                issues.append(f"[ALERTE] {fname}: n_ordre ET n_dossier vides")
        if not geometre or geometre in ('None',''):
            issues.append(f"[ALERTE] {fname}: géomètre vide")
        elif any(x in geometre.lower() for x in ['expert', 'g\u00e9om', 'geom', 'ing\u00e9nierie', 'ingenier', 'r\u00e9f\u00e9rence']):
            issues.append(f"[ALERTE] {fname}: géomètre suspect = '{geometre}'")
        if not section or section in ('None',''):
            issues.append(f"[INFO]   {fname}: section vide")
        if nature == 'AUTRE' or not nature:
            issues.append(f"[INFO]   {fname}: nature acte = '{nature}'")

print()
print("=" * 90)
print("PROBLÈMES DÉTECTÉS")
print("=" * 90)
for iss in issues:
    print(iss)

print()
print("=" * 90)
print("RÉSUMÉ JSON — méthodes d'extraction")
print("=" * 90)

json_files = sorted(glob.glob('outputs/*_plan_*.json'))
for jf in json_files:
    if 'payload' in jf or 'coherence' in jf:
        continue
    with open(jf, encoding='utf-8', errors='replace') as fp:
        try:
            d = json.load(fp)
        except:
            continue
    pages = d.get('pages', [])
    champs = pages[0].get('champs', {}) if pages else d.get('champs', {})
    bname = os.path.basename(jf)
    print(f"\n--- {bname} ---")
    for k, v in champs.items():
        if isinstance(v, dict) and not k.startswith('_'):
            valeur = v.get('valeur', '')
            methode = v.get('methode', 'ocr')
            conf = v.get('confidence', '?')
            print(f"  {k:<28}: {str(valeur):<30} [methode={methode}, conf={conf}]")
