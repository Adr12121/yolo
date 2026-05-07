# 🗺️ Cadastral OCR & Extraction Pipeline (PFE)

Projet de fin d'études dédié à l'automatisation de l'extraction de données à partir de documents cadastraux (Plans modernes DA/DMPC et Livrets historiques).

## 🚀 Fonctionnalités principales

- **Classification Intelligente** : Distinction automatique entre plans modernes et livrets manuscrits.
- **Détection YOLOv8** : Localisation précise des zones d'intérêt (cartouches, tableaux, mentions manuscrites).
- **Pipeline OCR Hybride** : 
  - Extraction de texte structuré pour les documents imprimés.
  - Reconnaissance de l'écriture manuscrite pour les archives historiques.
- **Validation Human-in-the-loop** : Interface Streamlit dédiée pour la vérification visuelle et la correction des données extraites.
- **Référentiel National** : Comparaison systématique avec la base de données officielle des communes françaises pour éliminer les hallucinations OCR.

## 🛠️ Stack Technique

- **Langage** : Python 3.x
- **Vision** : YOLOv8 (Ultralytics), PyMuPDF (fitz)
- **OCR** : Tesseract / TrOCR / Kraken (selon configuration)
- **Interface** : Streamlit
- **Données** : JSON / CSV / YAML

## 📁 Structure du Projet

- `main.py` : Point d'entrée principal du pipeline de traitement.
- `app_validation.py` : Interface de validation utilisateur.
- `modern_plan_extractor.py` : Logique d'extraction spécifique aux plans DA/DMPC.
- `plan_classifier.py` : Moteur de classification des types de documents.
- `models/` : Modèles YOLO et poids entraînés.

## ⚙️ Installation & Utilisation

1. **Installation des dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

2. **Lancement du traitement** :
   ```bash
   python main.py
   ```

3. **Lancement de l'interface de validation** :
   ```bash
   streamlit run app_validation.py
   ```

---
*Développé dans le cadre d'un Projet de Fin d'Études (PFE).*
