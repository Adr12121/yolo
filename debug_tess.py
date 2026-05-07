import cv2, numpy as np, subprocess, os, pandas as pd
import fitz

# Lire la 1ère page du PDF avec une résolution beaucoup plus haute (300 DPI)
doc = fitz.open("inputs/1992C100001_a094147_1PVa.pdf")
page = doc.load_page(0)
# Augmenter la résolution x2 (matrix = zoom 2x = ~150 DPI -> 300 DPI)
mat = fitz.Matrix(3.0, 3.0)  # 3x = environ 216 DPI natif -> 648 DPI effectif
pix = page.get_pixmap(matrix=mat)
img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
print(f"Image shape (3x zoom): {img.shape}")

# Binarisation
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("outputs/debug_tess_hires.png", thresh)

temp_img = os.path.abspath("outputs/debug_tess_hires.png")
temp_prefix = os.path.abspath("outputs/debug_tess_hires_out")

win_img = subprocess.check_output(['wslpath', '-w', temp_img]).decode().strip()
win_prefix = subprocess.check_output(['wslpath', '-w', temp_prefix]).decode().strip()

tess_exe = "/mnt/c/Users/Topo_4/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
tessdata_dir = r'C:\Users\Topo_4\AppData\Local\Programs\Tesseract-OCR\tessdata'

cmd_tsv = [tess_exe, win_img, win_prefix, '--tessdata-dir', tessdata_dir, '--oem', '3', '--psm', '3', '-l', 'fra', 'tsv']
r = subprocess.run(cmd_tsv, capture_output=True, text=True)
print(f"Code: {r.returncode}, Stderr: {r.stderr[:200]}")

tsv_file = temp_prefix + ".tsv" 
if os.path.exists(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t', quoting=3)
    print(f"Lignes TSV: {len(df)}")
    
    # Voir les valeurs de confidence brutes
    print("Valeurs conf uniques:", sorted(df['conf'].unique()))
    
    # Filtrer != -1 (lignes qui ont une vraie valeur)
    df2 = df[df['conf'] != -1]
    print(f"Mots avec conf != -1: {len(df2)}")
    print(df2[['text','conf']].to_string())
