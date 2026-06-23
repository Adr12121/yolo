import fitz, cv2, base64, json, urllib.request
import numpy as np

doc = fitz.open(r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\inputs\289-891-A (1).pdf')
pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Resize to prevent excessive VRAM usage, making it run faster
max_dim = 1024
h_img, w_img = img_bgr.shape[:2]
if max(h_img, w_img) > max_dim:
    scale = max_dim / max(h_img, w_img)
    img_bgr = cv2.resize(img_bgr, (int(w_img * scale), int(h_img * scale)))

_, buffer = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
img_base64 = base64.b64encode(buffer).decode('utf-8')

prompt = """Extract the text in this image. Format as JSON.
{
  "commune": "nom de la commune",
  "section": "lettre de la section",
  "geometre": "nom du geometre"
}"""

payload = {'model': 'llava', 'prompt': prompt, 'images': [img_base64], 'stream': False, 'options': {'temperature': 0.0}}

req = urllib.request.Request('http://127.0.0.1:11434/api/generate', data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
try:
    res = urllib.request.urlopen(req)
    print(json.loads(res.read().decode('utf-8'))['response'])
except Exception as e:
    print("CRASH:", e)
