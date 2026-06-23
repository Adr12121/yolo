import cv2
import fitz
import numpy as np
import easyocr
import re

pdf_path = "inputs/1992C100001_a094147_1PVa.pdf"
doc = fitz.open(pdf_path)
page = doc[0]
mat = fitz.Matrix(2.0, 2.0)
pix = page.get_pixmap(matrix=mat, alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
h, w = img_bgr.shape[:2]

reader = easyocr.Reader(['fr'])

# Zone section
z = [0.0, 0.0, 0.60, 0.40]
x0, y0, x1, y1 = int(z[0]*w), int(z[1]*h), int(z[2]*w), int(z[3]*h)
crop = img_bgr[y0:y1, x0:x1]
res = reader.readtext(crop)

full = "\n".join(r[1] for r in res)
print("FULL TEXT SECTION ZONE:\n", full)
print("----------------")
m = re.search(r"\bsection\s+([A-Z]{1,2})\b", full, re.IGNORECASE)
if m:
    print("MATCHED PATTERN 1:", m.groups())
else:
    print("NO MATCH PATTERN 1")

m2 = re.search(r"(?:section\s+(?:cadastrale\s+)?n[o\xb0]?|section\s*:)\s*([A-Z]{1,2}\d{0,2})\b", full, re.IGNORECASE)
if m2:
    print("MATCHED PATTERN 2:", m2.groups())
else:
    print("NO MATCH PATTERN 2")
