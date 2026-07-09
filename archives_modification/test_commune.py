from plan_classifier import _validate_field, load_commune_db
db = load_commune_db()
print("Test 1:", _validate_field("commune", "Saint Privat.......", db))
print("Test 2:", _validate_field("commune", "Saint Privat", db))
print("Test 3:", _validate_field("commune", "Sant Privat", db))
