import cv2
import base64
import json
import subprocess
import os

img_bgr = cv2.imread('inputs/EXEMPLE_SERRET.pdf_page_1.png')
if img_bgr is None:
    # Try pdf to img
    import fitz, numpy as np
    doc = fitz.open('inputs/EXEMPLE_SERRET.pdf')
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

h_img, w_img = img_bgr.shape[:2]
z = [0.0, 0.0, 0.40, 0.30]
x0, y0 = int(z[0]*w_img), int(z[1]*h_img)
x1, y1 = int(z[2]*w_img), int(z[3]*h_img)
crop_img = img_bgr[y0:y1, x0:x1]

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
_, buffer = cv2.imencode('.jpg', crop_img, encode_param)
img_base64 = base64.b64encode(buffer).decode('utf-8')

prompt = "Quel est le nom de la ville ou commune écrit sur cette image ? Réponds uniquement par le nom de la commune, sans aucune phrase."
payload = {
    'model': 'llava',
    'prompt': prompt,
    'images': [img_base64],
    'stream': False,
    'options': {'temperature': 0.0, 'num_predict': 64, 'seed': 42}
}

with open('payload.json', 'w') as f:
    json.dump(payload, f)

res = subprocess.run(['curl.exe', '-s', '-X', 'POST', 'http://127.0.0.1:11434/api/generate', '-H', 'Content-Type: application/json', '-d', '@payload.json'], capture_output=True, text=True)

print(res.stdout)
