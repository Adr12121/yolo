import glob, re

original_geometres = '["BARRE", "DUPUY", "HARROIS", "SERRET", "GUIGUE", "BARRIAL", "ROBERT", "GALLO-BALMA"]'

for filepath in glob.glob("*.py"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # We look for GEOMETRES_CONNUS = [...]
        if "GEOMETRES_CONNUS =" in content:
            # replace it
            new_content = re.sub(
                r'GEOMETRES_CONNUS\s*=\s*\[[^\]]+\]', 
                f'GEOMETRES_CONNUS = {original_geometres}', 
                content
            )
            if new_content != content:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Reverted {filepath}")
    except Exception as e:
        pass
