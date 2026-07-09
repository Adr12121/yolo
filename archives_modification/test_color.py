import sys, os, cv2, fitz, numpy as np
import easyocr

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

img = get_page_image('inputs/289-891-A (1).pdf', 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Red mask (red wraps around 0 and 180 in HSV)
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Green mask
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

cv2.imwrite('outputs/test_mask_red.jpg', mask_red)
cv2.imwrite('outputs/test_mask_green.jpg', mask_green)

print("Masks saved.")
