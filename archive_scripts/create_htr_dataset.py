import os
import cv2
import pandas as pd
from PIL import Image
from kraken import blla
import numpy as np
import fitz # PyMuPDF

def extract_training_crops(input_dir='inputs', output_dir='dataset_images'):
    """Extrait des segments de lignes manuscrites pour créer un dataset d'entraînement."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    labels = []
    
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.pdf', '.jpg', '.png'))]
    
    for f in files:
        print(f"Traitement de {f}...")
        path = os.path.join(input_dir, f)
        
        # Lecture PDF ou Image
        images = []
        if f.lower().endswith('.pdf'):
            doc = fitz.open(path)
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                images.append(img)
        else:
            images = [cv2.imread(path)]
            
        for p_idx, img in enumerate(images):
            # Segmentation Kraken
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            bounds = blla.segment(pil_img)
            
            lines = bounds.lines if hasattr(bounds, 'lines') else []
            for l_idx, line in enumerate(lines):
                poly = np.array(line.boundary, np.int32)
                x, y, w, h = cv2.boundingRect(poly)
                
                # Crop avec un peu de marge
                pad = 10
                crop = img[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
                
                if crop.size == 0 or w < 50 or h < 10:
                    continue
                
                crop_name = f"{os.path.splitext(f)[0]}_p{p_idx}_l{l_idx}.jpg"
                cv2.imwrite(os.path.join(output_dir, crop_name), crop)
                
                labels.append({
                    "file_name": crop_name,
                    "text": "À COMPLÉTER" # L'utilisateur devra remplir ceci
                })
                
    df = pd.DataFrame(labels)
    df.to_csv('labels_skeleton.csv', index=False, sep=',')
    print(f"\nTerminé ! {len(labels)} images extraites dans '{output_dir}'.")
    print("Veuillez remplir 'labels_skeleton.csv' avec les transcriptions exactes avant l'entraînement.")

if __name__ == "__main__":
    extract_training_crops()
