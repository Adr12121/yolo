import codecs

with codecs.open("tmp_rebuild/plan_classifier.py", "r", "utf-8", errors="ignore") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "m_com = re.search(r'(?i)commune" in line:
        new_lines.append("    m_com = re.search(r'(?i)commune\\s+(?:de\\s+|d\\'|d’)([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\\s\\-]{2,30}?)(?:\\s+section|\\s*,|\\s*\\n|$)', full_text)\n")
    elif "|$)', full_text)" in line:
        pass # Skip the broken newline part
    elif "m_fait = re.search(r'(?i)fait" in line:
        new_lines.append("    m_fait = re.search(r'(?i)fait\\s+[aà]\\s+([A-Za-zÀ-ÿ\\s\\-]{2,30}?)\\s*,\\s*le\\s+([0-9]{1,2}(?:er)?\\s+[a-zéû]+(?:\\s+[0-9]{4})?|[0-9]{1,2}\\s*[/\-\.]\\s*[0-9]{1,2}\\s*[/\-\.]\\s*[0-9]{2,4})', full_text)\n")
    else:
        new_lines.append(line)

with codecs.open("tmp_rebuild/plan_classifier.py", "w", "utf-8") as f:
    f.writelines(new_lines)
