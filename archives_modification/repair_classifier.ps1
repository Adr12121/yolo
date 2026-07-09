# repair_classifier.ps1
# Répare plan_classifier.py en remplaçant la section corrompue

$file = "plan_classifier.py"
$backup = "plan_classifier.py.bak"

# Backup
Copy-Item $file $backup -Force
Write-Host "Backup créé : $backup"

# Lire le contenu en UTF-8
$content = Get-Content $file -Raw -Encoding UTF8

# Compter les occurrences du bug
$bugPattern1 = 'val_norm.startswith\(interdit.lower\(\)    elif'
$bugPattern2 = 'return Noneier'
$matches1 = [regex]::Matches($content, $bugPattern1).Count
$matches2 = [regex]::Matches($content, $bugPattern2).Count
Write-Host "Occurrences bug1 (startswith corrompu): $matches1"
Write-Host "Occurrences bug2 (return Noneier): $matches2"

# RÉPARATION 1 : Extraire les lignes avant et après la corruption
# La ligne 335 corrompue doit devenir deux lignes séparées
$content = $content -replace 'val_norm\.startswith\(interdit\.lower\(\)\s+elif field_type in \("n_ordre", "n_dossier"\):', `
'val_norm.startswith(interdit.lower() + " "):
            return None

    if field_type == "section":
        m = re.search(r''\''\\b([A-Za-z]{1,2}|\\d{1,3})\\b'\'', val)
        if m:
            clean = m.group(1).upper()
            if re.match(r'\''^[018]$'\'', clean) and len(val) <= 3:
                clean = clean.replace("0", "O").replace("1", "I").replace("8", "B")
            return clean
        return None

    elif field_type == "feuille":
        m = re.search(r'\''\b(\d{1,4}[A-Za-z]?|[A-Za-z]\d{1,3})\b'\'', val)
        if m:
            return m.group(1).upper()
        return None

    elif field_type == "echelle":
        if any(u in val_norm for u in ["ca", "ha", "m2", " a ", "m" + [char]0x00B2 + ""]):
            return None
        m = re.search(r'\''(1\s*[/:]\s*\d{3,5}|\d{3,5})'\'', val)
        if m:
            return m.group(1).replace(" ", "")
        return None

    elif field_type in ("n_ordre", "n_dossier"):'

Write-Host "Réparation 1 appliquée : $([regex]::Matches($content, 'field_type == "section"').Count) occurrences de section trouvées"

# RÉPARATION 2 : Supprimer le doublon du bloc n_ordre (les lignes 367-388 corrompues)
# Chercher "return Noneier") jusqu'à "return None" suivi de "elif field_type == "date""
$content = $content -replace 'return Noneier"\"\):\r?\n\s+# Doit contenir.*?return None\r?\n\r?\n    elif field_type == "date"', `
'return None

    elif field_type == "date"' 

# Vérification
$matches2_after = [regex]::Matches($content, 'Noneier').Count
Write-Host "Après réparation 2 - occurrences Noneier: $matches2_after"

# Écrire le fichier
Set-Content $file -Value $content -Encoding UTF8 -NoNewline
Write-Host "Fichier écrit."

# Vérifier la syntaxe Python
$result = & python -m py_compile $file 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Syntaxe Python OK"
} else {
    Write-Host "❌ Erreur syntaxe: $result"
    Write-Host "Restauration backup..."
    Copy-Item $backup $file -Force
}
