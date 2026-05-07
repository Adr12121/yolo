import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

"""
Script pour fine-tuner (ré-entraîner) le modèle de reconnaissance TrOCR 
(celui qui lit les anciennes écritures) sur VOTRE propre style d'écriture.
"""

class CustomCadastreDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. On récupère le nom de l'image (ex: parcelle_123.jpg) et son texte ("305")
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        
        # 2. On charge l'image en RGB
        image = Image.open(f"{self.root_dir}/{file_name}").convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # 3. On tokenize le texte (le convertit en nombres)
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
                                          
        # PyTorch Trainer s'attend à ce que les caractères vides (padding) valent -100
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

def train_trocr_custom():
    print("Initialisation du modèle TrOCR agomberto/trocr-large-handwritten-fr...")
    MODEL_NAME = "agomberto/trocr-large-handwritten-fr"
    
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # Paramétrage spécial TrOCR
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id

    # -- SIMULATION DE DONNÉES --
    # Dans la vraie vie, vous devez créer un fichier 'labels.csv' listant vos images 
    # et le texte exact qui est dessus.
    df = pd.DataFrame({
        'file_name': ['image1.jpg', 'image2.jpg'],
        'text': ['Commune de Vals', 'Section AB']
    })
    
    print("\nAttention: Le dossier 'dataset_images' n'existe pas. C'est juste un exemple de code !")
    print("Pour que cela fonctionne, vous devez :")
    print("1. Découper des bouts d'images de votre plan avec du texte.")
    print("2. Créer un fichier labels.csv qui contient : 'nom_image.jpg,texte exact'.\n")
    
    try:
        # train_dataset = CustomCadastreDataset(root_dir="dataset_images", df=df, processor=processor)
        
        # Paramètres d'entraînement
        # training_args = Seq2SeqTrainingArguments(
        #     predict_with_generate=True,
        #     evaluation_strategy="steps",
        #     per_device_train_batch_size=4,  # Si carte graphique puissante, mettre 8
        #     fp16=True,                      # Accélère via la carte graphique
        #     output_dir="./trocr_pfe_cadastre",
        #     logging_steps=10,
        #     save_steps=1000,
        #     eval_steps=100,
        # )

        # trainer = Seq2SeqTrainer(
        #     model=model,
        #     tokenizer=processor.feature_extractor,
        #     args=training_args,
        #     train_dataset=train_dataset,
        # )
        
        # print("Début de l'entraînement (peut prendre plusieurs heures)...")
        # trainer.train()
        print("Scellette prêtre à être utilisée quand vos données seront prêtes.")
        
    except Exception as e:
        print(f"Erreur (Normal si vous n'avez pas préparé les données) : {e}")

if __name__ == "__main__":
    train_trocr_custom()
