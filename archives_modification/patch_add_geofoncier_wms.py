"""
Patch 3 : Ajoute la couche publique Géofoncier (pastilles + emprises)
au fond de carte Folium.
"""

with open('app_validation.py', encoding='utf-8') as f:
    content = f.read()

anchor = '            # OVERLAY CADASTRE : numeros de parcelles actifs par defaut'

new_layers = '''\
            # ─── COUCHES GEOFONCIER (Publiques via API OGE) ────────────────────
            # Emprises des dossiers (polygones)
            folium.WmsTileLayer(
                url="https://api2.geofoncier.fr/api/referentielsoge/wxs?",
                layers="DOSSIERS_EMPRISES",
                fmt="image/png", transparent=True,
                name="Géofoncier (Emprises)",
                attr="Ordre des Géomètres-Experts",
                overlay=True, show=False, opacity=0.6,
            ).add_to(_fmap)

            # Localisants des dossiers (pastilles) - Actif par defaut
            folium.WmsTileLayer(
                url="https://api2.geofoncier.fr/api/referentielsoge/wxs?",
                layers="DOSSIERS_LOCALISANTS",
                fmt="image/png", transparent=True,
                name="Géofoncier (Pastilles)",
                attr="Ordre des Géomètres-Experts",
                overlay=True, show=True, opacity=0.9,
            ).add_to(_fmap)

'''

if anchor in content and new_layers not in content:
    content_new = content.replace(anchor, new_layers + anchor)
    with open('app_validation.py', 'w', encoding='utf-8') as f:
        f.write(content_new)
    print("COUCHES GEOFONCIER AJOUTEES AVEC SUCCES")
else:
    print("Echec : ancre non trouvee ou couches deja presentes")
