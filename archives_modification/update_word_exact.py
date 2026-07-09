import docx

doc_path = r'c:\Users\Topo_4\Documents\AT_PFE\Plan Détaillé du Mémoire de PFE_backup.docx'
out_path = r'c:\Users\Topo_4\Documents\AT_PFE\Plan Détaillé du Mémoire de PFE.docx'
doc = docx.Document(doc_path)

updates = {
    8: "   La difficulté d'exploitation des registres anciens (manuscrits) et la diversité structurelle des plans modernes numérisés.",
    11: "   Créer un processus hybride et autonome de détection, de lecture sémantique et de validation interactive.",
    15: "   Spécifications techniques des données attendues par l'API de l'Ordre (format, tri de pertinence…).",
    17: "   Analyse de la bivalence des flux documentaires : registres anciens (format papier) et plans modernes (DGFIP, DMPC, PVa).",
    18: "   Analyse de la complexité des écritures et des variations de mise en page.",
    26: "   L'apport décisif des modèles multimodaux (ex: Qwen2-VL, LLaVA) pour la compréhension sémantique.",
    28: " 4.1. Étape 1 : Classification et routage des documents",
    29: "   Identification automatique de la nature de l'archive (plan moderne vs registre ancien).",
    30: "   Recherche de correspondances géométriques et par ancrage de mots-clés pour le recadrage des plans modernes.",
    31: " 4.2. Étape 2 : Extraction géométrique et segmentation",
    32: "   Détection des colonnes et cellules via YOLO pour les livrets anciens (algorithme \"Ghost Lines\").",
    33: " 4.3. Étape 3 : Moteur d'extraction hybride (HTR & VLM)",
    34: "   Stratégie multi-hypothèses : Génération de plusieurs prédictions par TrOCR pour les écritures très dégradées.",
    35: "   Utilisation des modèles Vision-Langage (VLM) en tant que moteur d'extraction sémantique direct (N° DA, Section).",
    40: " 5.2. Matching intelligent et croisement d'archives",
    42: "   Matching Fuzzy, similarité visuelle et croisement avec des répertoires historiques externes (ex: Racat & Ceyte).",
    43: " 5.3. Interface \"Human-in-the-Loop\" : l'application de validation",
    44: "   Développement d'une interface de contrôle réactive (Streamlit) garantissant l'intégrité de la donnée.",
    45: "   Validation via les référentiels officiels et auto-sauvegarde des corrections en temps réel.",
    47: " 6.1. Le connecteur API Geofoncier",
    48: "   Développement d'une communication directe avec Geofoncier (mode Dry Run de simulation, requêtes multipart).",
    49: "   Structuration, contrôle et formatage automatiques des paquets de données (JSON/CSV) pour l'API.",
}

for i, p in enumerate(doc.paragraphs):
    if i in updates:
        new_text = updates[i]
        if len(p.runs) > 0:
            # Preserve the styling of the first run
            p.runs[0].text = new_text
            for run in p.runs[1:]:
                run.text = ""
        else:
            p.text = new_text

doc.save(out_path)
print("Updated successfully while keeping exact formatting.")
