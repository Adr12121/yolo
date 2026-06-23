import fitz
import cv2
import numpy as np
import easyocr
import sys
from plan_classifier import CONTEXTUAL_PATTERNS, _find_field_contextual

file_path = 'inputs/geofoncier_dmpc_07116_000_262 (1).pdf'
doc = fitz.open(file_path)
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

reader = easyocr.Reader(['fr'])
res = reader.readtext(img)

# Shift format for _find_field_contextual
h, w = img.shape[:2]
ocr_res = []
for bbox, text, prob in res:
    ocr_res.append((bbox, text, prob))

print(f"File: {file_path}")
for field in ["commune", "n_ordre", "geometre", "date", "section", "feuille"]:
    val = _find_field_contextual(field, ocr_res, (h, w))
    print(f"{field}: {val}")

print("\n--- OCR RAW RESULTS ---")
for r in res:
    print(r[1])
