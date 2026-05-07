from ultralytics import YOLO

def train_yolo_custom():
    """
    Script pour ré-entraîner YOLOv8 sur VOS documents cadastraux.
    L'objectif n'est plus de lire le texte, mais de détecter OÙ se trouvent 
    les zones importantes (Cartouche, Bloc Commune, Numéro de Dossier).
    """
    print("Initialisation du modèle YOLOv8 pré-entraîné...")
    # On part d'un modèle léger ('n' pour nano) qui s'entraîne vite sur un simple PC
    model = YOLO('yolov8n.pt') 

    print("Lancement de l'entraînement...")
    # NOTE: Ce script nécessite un fichier 'cadastre_data.yaml' et un dossier d'images étiquetées.
    # Exécutez ce script uniquement lorsque votre dataset est prêt.
    try:
        results = model.train(
            data='cadastre_data.yaml', # Le fichier qui décrit où sont vos images et vos classes (ex: 0: commune, 1: geometre)
            epochs=50,                 # 50 passages complets sur vos exemples (commencez par 50, puis montez à 100 si besoin)
            imgsz=1024,                # Taille de l'image. Les plans sont grands, on utilise 1024 au lieu de 640.
            batch=4,                   # Nombre d'images traitées en même temps (dépend de la mémoire de votre carte graphique)
            name='detection_cadastre', # Le nom du dossier de sauvegarde
            device='cpu'               # Mettez '0' si vous avez une carte graphique NVIDIA, sinon 'cpu'
        )
        print("Entraînement terminé avec succès !")
        print("Le nouveau modèle est sauvegardé dans : runs/detect/detection_cadastre/weights/best.pt")
        print("Vous pourrez l'utiliser dans main.py avec : model = YOLO('runs/detect/detection_cadastre/weights/best.pt')")
        
    except Exception as e:
        print(f"Erreur lors de l'entraînement : {e}")
        print("\nAvez-vous créé le fichier 'cadastre_data.yaml' et le dossier de données ?")

if __name__ == "__main__":
    train_yolo_custom()
