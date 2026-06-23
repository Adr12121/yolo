import re

file_path = 'plan_classifier.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''    if field == "indication":
        val_clean = val.strip().upper()
        if "A" in val_clean and "B" not in val_clean and "C" not in val_clean:
            return "d'après les indications qu'ils ont fournies au bureau"
        elif "B" in val_clean and "A" not in val_clean and "C" not in val_clean:
            return "en conformité d'un piquetage qu'ils ont effectué sur le terrain"
        elif "C" in val_clean and "A" not in val_clean and "B" not in val_clean:
            return "d'après un plan d'arpentage ou de bornage, dont copie ci-jointe"'''

new_logic = '''    if field == "indication":
        val_clean = val.strip().upper()
        # On s'assure que si c'est A, B ou C, c'est vraiment juste la lettre (ex: "B", "RÉPONSE: B", "LETTRE B")
        # On évite de remplacer "BORNAGE" par la phrase B !
        if re.fullmatch(r'(?:LA LETTRE |R[EÉ]PONSE\s*:\s*|^)[ABC]', val_clean) or len(val_clean) == 1:
            if "A" in val_clean:
                return "d'après les indications qu'ils ont fournies au bureau"
            elif "B" in val_clean:
                return "en conformité d'un piquetage qu'ils ont effectué sur le terrain"
            elif "C" in val_clean:
                return "d'après un plan d'arpentage ou de bornage, dont copie ci-jointe"'''

content = content.replace(old_logic, new_logic)

# if mojibake is present, fix it
content = content.replace("d'aprs", "d'après")
content = content.replace("conformitǸ", "conformité")
content = content.replace("effectuǸ", "effectué")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
