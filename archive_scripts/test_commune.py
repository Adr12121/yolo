import json

try:
    with open('communes_france.json', 'r', encoding='utf-8') as f:
        communes = json.load(f)
    for c in communes:
        if c.get('code') in ['07116', '17116'] or 'Aigrefeuille' in c.get('nom', ''):
            print(c)
except Exception as e:
    print(f"Error reading communes_france.json: {e}")

try:
    with open('ardeche.json', 'r', encoding='utf-8') as f:
        communes = json.load(f)
    for c in communes:
        if c.get('code') in ['07116', '17116'] or 'Aigrefeuille' in c.get('nom', ''):
            print("ardeche:", c)
except Exception as e:
    print(f"Error reading ardeche.json: {e}")
