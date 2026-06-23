import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

results = glob.glob('**/*.xlsx', recursive=True)
xlsx_path = None
for r in results:
    if '~$' not in r and 'BARRIAL' in r:
        xlsx_path = os.path.abspath(r)
        break

df = pd.read_excel(xlsx_path, header=None)

# === CAS CONCRET: plan geofoncier_dmpc_07289_000_608.pdf ===
# Info extraite: Commune=Saint-Privat, Section=AC, Geometre=HARROIS
# La ref_dossier attendue = 97050

print("=== TEST 1: recherche 'Saint-Privat', section 'AC', geometre HARROIS ===")
# Le plan a section AC mais le répertoire a AL -> OCR a mal lu AC au lieu de AL ?
mask_commune = df[3].str.contains('SAINT.PRIVAT', case=False, na=False, regex=True)
mask_section_AC = df[4].str.contains('AC', case=False, na=False)
mask_section_AL = df[4].str.contains('AL', case=False, na=False)
print("Avec section AC:")
print(df[mask_commune & mask_section_AC].iloc[:, :11].to_string())
print()
print("Avec section AL:")
print(df[mask_commune & mask_section_AL].iloc[:, :11].to_string())

print()
print("=== TEST 2: parcelle 69 de Saint-Privat ===")
mask_p69 = df[5] == 69
print(df[mask_commune & mask_p69].iloc[:, :11].to_string())

print()
print("=== Structure colonnes sur quelques lignes type DA ===")
# Voir tout le détail des colonnes pour une ligne DA
sample = df[df[8] == 'DA'].head(3)
for idx, row in sample.iterrows():
    print(f"Ligne Excel {idx}:")
    for col_idx, val in enumerate(row):
        if pd.notna(val) and str(val).strip():
            col_letter = chr(65 + col_idx) if col_idx < 26 else f"AA+{col_idx-26}"
            print(f"  Col {col_letter} ({col_idx}): {val}")
    print()

print()
print("=== Référence dossier: format attendu ===")
# Ligne 2915: annee=97, n_dossier=50 -> ref = "97050" (année 2 chiffres + dossier 3 chiffres zéro-padded)
row = df.iloc[2915]
annee = int(row[0])  # 97
ndos = int(row[1])   # 50
ref = f"{annee:02d}{ndos:03d}"
print(f"Ligne 2915: Année={annee}, N°={ndos} -> Référence = '{ref}'")
print(f"Référence attendue selon toi: '97050' -> MATCH: {ref == '97050'}")
