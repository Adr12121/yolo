import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open("app_validation.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    # Emojis or words with 4+ uppercase letters
    if re.search(r'[^\x00-\x7F\xC0-\xFF\sœ]', l) or re.search(r'[A-ZÀ-Ÿ]{4,}', l):
        print(f"{i+1}: {l.strip()}")
