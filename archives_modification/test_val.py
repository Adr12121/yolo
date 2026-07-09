import plan_classifier as pc
import json

pc._commune_db = []
for e in json.load(open("ardeche.json", encoding="utf-8")):
    n = e.get("nom","").strip()
    if n:
        pc._commune_db.append({"officiel": n, "code": e.get("code","")})

print("Testing d_loyezse:")
v1 = pc._validate_field("commune", "d_loyezse", pc._commune_db)
print("v1:", v1)

print("Testing Sectioj:")
v2 = pc._validate_field("commune", "Sectioj", pc._commune_db)
print("v2:", v2)
