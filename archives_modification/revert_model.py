import re

file_path = 'plan_classifier.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace llama3.2-vision with llava everywhere in _OLLAMA_MODEL_FOR_TYPE just to be absolutely sure!
# Or at least for DMPC!
# I will change ALL of them to llava to fix the crash.

new_model_dict = '''_OLLAMA_MODEL_FOR_TYPE = {
    "DMPC":    "llava",
    "PVa":     "llava",
    "PLa":     "llava",
    "CROQUIS": "llava",
    "GENERIC": "llava",
    "DEFAULT": "llava"
}'''

# Replace the block
pattern = re.compile(r'_OLLAMA_MODEL_FOR_TYPE\s*=\s*\{[^}]+\}', re.DOTALL)
content = pattern.sub(new_model_dict, content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Reverted to llava successfully!")
