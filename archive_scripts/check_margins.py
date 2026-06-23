import cv2
import numpy as np
import fitz

doc = fitz.open('inputs/geofoncier_dmpc_07116_000_262 (1).pdf')
pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
coords = cv2.findNonZero(thresh)
x, y, w, h = cv2.boundingRect(coords)

print(f"Original size: {img.shape}")
print(f"Content bounding box: x={x}, y={y}, w={w}, h={h}")
print(f"White margins: Left={x}, Right={img.shape[1]-(x+w)}, Top={y}, Bottom={img.shape[0]-(y+h)}")
