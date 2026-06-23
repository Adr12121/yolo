import os
import glob
import json
from geofoncier_api import create_geofoncier_dossier, upload_document_to_dossier

# Mode Simulation (True par défaut tant que vous n'avez pas de clé d'API)
DRY_RUN = True

def main():
    print("==================================================")
    print(" Lancement de l'export automatique vers Géofoncier")
    print("==================================================")

    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print(f"Erreur : Le dossier '{outputs_dir}' n'existe pas.")
        return

    json_files = glob.glob(os.path.join(outputs_dir, "*_plan_moderne.json"))
    if not json_files:
        print("Aucun fichier JSON trouvé dans le dossier 'outputs'.")
        return

    print(f"{len(json_files)} fichier(s) JSON à traiter trouvés.\n")

    for json_path in json_files:
        print(f"--- Traitement de : {os.path.basename(json_path)} ---")
        
        # 1. Création de la pastille / dossier
        result_dossier = create_geofoncier_dossier(json_path, dry_run=DRY_RUN)
        
        if result_dossier.get("success"):
            id_dossier = result_dossier.get("id_dossier")
            
            # Déduire le chemin du PDF/Image original à uploader
            # Note: il faut adapter selon la convention de nommage exacte de vos fichiers originaux
            base_name = os.path.basename(json_path).replace("_plan_moderne.json", "")
            
            # On cherche le document correspondant dans inputs/
            # Peut être pdf, jpg, png ou tif
            pdf_path = None
            for ext in ['.pdf', '.jpg', '.png', '.tif']:
                possible_path = os.path.join("inputs", f"{base_name}{ext}")
                if os.path.exists(possible_path):
                    pdf_path = possible_path
                    break
            
            if pdf_path:
                # 2. Versement du document dans la pastille créée
                upload_result = upload_document_to_dossier(id_dossier, pdf_path, dry_run=DRY_RUN)
                if upload_result.get("success"):
                    print("--> Traitement terminé avec SUCCÈS pour ce document.")
                else:
                    print("--> Création dossier OK, mais ÉCHEC de l'upload du document.")
            else:
                print(f"⚠️ Document source introuvable pour upload (cherché dans inputs/ avec nom {base_name}.*)")
        else:
            print("--> ÉCHEC de la création du dossier sur Géofoncier.")
            
        print("\n")

if __name__ == "__main__":
    # Charge le .env s'il existe (installation de python-dotenv recommandée si non présente)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    main()
