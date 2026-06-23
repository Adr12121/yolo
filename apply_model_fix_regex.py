import re

file_path = 'plan_classifier.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(r'("DMPC"\s*:\s*)"llava"', r'\1"llama3.2-vision"', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
