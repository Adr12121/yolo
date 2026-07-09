import sys, os, cv2, fitz, numpy as np
sys.path.insert(0, os.getcwd())
from color_ocr_engine import extract_color_parcels

def get_page_image(pdf_path, page_num=0, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

print("Testing modern plan:")
img1 = get_page_image('inputs/289-891-A (1).pdf', 0)
res1 = extract_color_parcels(img1)
print(" Modern Nouvelles:", res1['nouvelles_parcelles'])
print(" Modern Anciennes:", res1['anciennes_parcelles'])

print("\nTesting old plan:")
img2 = get_page_image('inputs/geofoncier_dmpc_07116_000_262 (1).pdf', 0)
res2 = extract_color_parcels(img2)
print(" Old Nouvelles:", res2['nouvelles_parcelles'])
print(" Old Anciennes:", res2['anciennes_parcelles'])
