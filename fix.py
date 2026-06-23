import re

with open('plan_classifier.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace llama3.2-vision with llava in _OLLAMA_MODEL_FOR_TYPE
content = content.replace('\"PVa\":     \"llama3.2-vision\"', '\"PVa\":     \"llava\"')
content = content.replace('\"CROQUIS\": \"llama3.2-vision\"', '\"CROQUIS\": \"llava\"')
content = content.replace('\"GENERIC\": \"llama3.2-vision\"', '\"GENERIC\": \"llava\"')

# 2. Add return {} at the beginning of _extract_with_vlm_full_page
content = content.replace('def _extract_with_vlm_full_page(img_bgr, fields_to_extract, commune_db=None, type_plan: str = \"GENERIC\"):', 
'def _extract_with_vlm_full_page(img_bgr, fields_to_extract, commune_db=None, type_plan: str = \"GENERIC\"):\n    return {}  # Desactive pour des raisons de performances')

with open('plan_classifier.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('File updated successfully.')
