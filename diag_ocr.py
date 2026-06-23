"""
Script de diagnostic OCR : extrait le texte brut OCR de chaque plan
pour identifier ce que l'OCR lit réellement dans la zone n_ordre.
"""
import fitz, cv2, numpy as np, easyocr, re, sys

reader = easyocr.Reader(['fr'], gpu=False, verbose=False)

plans = [
    'inputs/geofoncier_dmpc_07116_000_262 (1).pdf',
    'inputs/geofoncier_dmpc_07289_000_608.pdf',
    'inputs/geofoncier_dmpc_07289_000_677.pdf',
]

for pdf_path in plans:
    print(f'\n{"="*60}')
    print(f'Plan: {pdf_path}')
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        doc.close()
    except Exception as e:
        print(f'  ERREUR: {e}')
        continue

    # Zone n_ordre sur DMPC : haut de page [0.0, 0.0, 1.0, 0.40]
    ocr_all = reader.readtext(img_bgr)
    
    print(f'\n-- Tous les tokens OCR dans le haut de page (y < 40%) contenant des chiffres --')
    keywords_ordre = ['ordre', 'da', 'arpentage', 'document', 'n°', 'n\'', 'numero']
    
    for (bbox, text, prob) in ocr_all:
        cy = (min(p[1] for p in bbox) + max(p[1] for p in bbox)) / 2 / h
        if cy > 0.50:
            continue
        # Montrer tous les tokens avec chiffres OU mots-clés
        has_digit = bool(re.search(r'\d', text))
        has_kw = any(k in text.lower() for k in keywords_ordre)
        if has_digit or has_kw:
            x0 = min(p[0] for p in bbox) / w
            y0 = min(p[1] for p in bbox) / h
            x1 = max(p[0] for p in bbox) / w
            print(f'  [{x0:.2f},{y0:.2f}]-[{x1:.2f}] prob={prob:.2f} | {repr(text)}')
    
    print(f'\n-- Tous les tokens contenant "ordre" (zone élargie) --')
    for (bbox, text, prob) in ocr_all:
        if 'ordre' in text.lower() or 'arpentage' in text.lower():
            cy = (min(p[1] for p in bbox) + max(p[1] for p in bbox)) / 2 / h
            x0 = min(p[0] for p in bbox) / w
            y0 = min(p[1] for p in bbox) / h
            print(f'  [{x0:.2f},{y0:.2f}] cy={cy:.2f} prob={prob:.2f} | {repr(text)}')
