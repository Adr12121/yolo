import os
import json
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from plan_classifier import process_plan

def test():
    file_path = "inputs/geofoncier_dmpc_07289_000_608.pdf"
    if not os.path.exists(file_path):
        print("File not found")
        return

    print("Loading models...")
    reader = easyocr.Reader(['fr'], gpu=torch.cuda.is_available())
    
    try:
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        if torch.cuda.is_available():
            model = model.to('cuda')
    except Exception as e:
        print(f"Error loading trocr: {e}")
        processor, model = None, None

    print(f"Testing {file_path}...")
    res = process_plan(file_path, reader, processor, model, "cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test()
