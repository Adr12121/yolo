import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import cv2
import fitz
import numpy as np
import os

pdf_path = r"inputs\geofoncier_dmpc_07289_000_677.pdf"
output_path = r"outputs\illustration_rapport_pfe.jpg"

def load_clean_image_from_pdf(pdf_path, page_num=0, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def get_single_roi_interactively(img_bgr, title="Selection"):
    roi = None
    root = tk.Tk()
    root.title(title)
    
    h, w = img_bgr.shape[:2]
    screen_w = root.winfo_screenwidth() - 100
    screen_h = root.winfo_screenheight() - 100
    scale = min(screen_w/max(w, 1), screen_h/max(h, 1))
    
    if scale < 1.0:
        img_resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        img_resized = img_bgr.copy()
        
    tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
    tk.Label(root, text="Tracez UN SEUL grand rectangle (la zone à zoomer) puis appuyez sur ENTREE pour valider.", font=("Arial", 12), fg="red").pack(pady=5)
    canvas = tk.Canvas(root, width=img_resized.shape[1], height=img_resized.shape[0], cursor="cross")
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    
    rect_id = None
    start_x, start_y = None, None
    
    def on_press(e):
        nonlocal start_x, start_y, rect_id
        start_x, start_y = canvas.canvasx(e.x), canvas.canvasy(e.y)
        if rect_id: canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=3)
        
    def on_drag(e):
        if rect_id: canvas.coords(rect_id, start_x, start_y, canvas.canvasx(e.x), canvas.canvasy(e.y))
            
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    
    def on_enter(e):
        nonlocal roi
        if rect_id:
            x1, y1, x2, y2 = canvas.coords(rect_id)
            roi = (int(min(x1,x2)/scale), int(min(y1,y2)/scale), int(abs(x2-x1)/scale), int(abs(y2-y1)/scale))
            root.quit()
            root.destroy()
            
    root.bind('<Return>', on_enter)
    root.mainloop()
    return roi

def get_rois_and_texts_interactively(img_bgr, title="Selection des Detections"):
    rois = []
    root = tk.Tk()
    root.title(title)
    
    h, w = img_bgr.shape[:2]
    screen_w = root.winfo_screenwidth() - 100
    screen_h = root.winfo_screenheight() - 100
    scale = min(screen_w/max(w, 1), screen_h/max(h, 1))
    
    if scale < 1.0:
        img_resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        img_resized = img_bgr.copy()
        
    tk_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
    tk.Label(root, text="Tracez un rectangle, appuyez sur ENTREE, saisissez le texte lu par l'OCR. ECHAP pour terminer l'étape.", font=("Arial", 12), fg="red").pack(pady=5)
    canvas = tk.Canvas(root, width=img_resized.shape[1], height=img_resized.shape[0], cursor="cross")
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
    
    rect_id = None
    start_x, start_y = None, None
    
    def on_press(e):
        nonlocal start_x, start_y, rect_id
        start_x, start_y = canvas.canvasx(e.x), canvas.canvasy(e.y)
        if rect_id: canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red', width=3)
        
    def on_drag(e):
        if rect_id: canvas.coords(rect_id, start_x, start_y, canvas.canvasx(e.x), canvas.canvasy(e.y))
            
    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    
    def on_enter(e):
        nonlocal rect_id
        if rect_id:
            x1, y1, x2, y2 = canvas.coords(rect_id)
            x, y, bw, bh = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
            real_x, real_y, real_w, real_h = int(x/scale), int(y/scale), int(bw/scale), int(bh/scale)
            if real_w > 5 and real_h > 5:
                # Ouvre une popup pour demander le texte
                text = simpledialog.askstring("Résultat OCR", "Quel texte la machine a-t-elle lu ici ? (ex: St Privot)", parent=root)
                if text:
                    rois.append((real_x, real_y, real_w, real_h, text))
                    canvas.itemconfig(rect_id, outline='green')
                    canvas.create_text(x, y-10, text=text, fill="blue", font=("Arial", 14, "bold"), anchor="sw")
                    rect_id = None
                else:
                    canvas.delete(rect_id)
                    rect_id = None
                    
    def on_esc(e):
        root.quit()
        root.destroy()
        
    root.bind('<Return>', on_enter)
    root.bind('<Escape>', on_esc)
    root.mainloop()
    return rois

def draw_boxes_with_labels(img, rois):
    res = img.copy()
    for (x, y, w, h, text) in rois:
        # Dessiner le rectangle de détection
        cv2.rectangle(res, (x, y), (x+w, y+h), (0, 200, 0), 4)
        
        # Préparer le texte
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Dessiner le fond (label) juste au-dessus du rectangle
        # On s'assure de ne pas sortir de l'image vers le haut
        box_y_top = max(y - text_h - 15, 0)
        box_y_bottom = box_y_top + text_h + 15
        
        cv2.rectangle(res, (x, box_y_top), (x + text_w + 20, box_y_bottom), (0, 200, 0), -1)
        
        # Écrire le texte en blanc par-dessus le fond vert
        cv2.putText(res, text, (x + 10, box_y_bottom - 8), font, font_scale, (255, 255, 255), thickness)
    return res

def main():
    print("Chargement de l'image depuis le PDF...")
    clean_img = load_clean_image_from_pdf(pdf_path)
    
    print("\n[ETAPE 1] Sélectionnez la ZONE globale HAUT-GAUCHE (Commune, etc)")
    roi_hg = get_single_roi_interactively(clean_img, "1. Zone Haut-Gauche")
    if not roi_hg: return
    x, y, w, h = roi_hg
    zone_hg = clean_img[y:y+h, x:x+w].copy()
    
    print("\n[ETAPE 2] Tracez les boîtes sur le texte. Après chaque boîte, tapez le texte lu.")
    detections_hg = get_rois_and_texts_interactively(zone_hg, "2. Détections Haut-Gauche")
    
    print("\n[ETAPE 3] Sélectionnez la ZONE globale CENTRE (Parcelles)")
    roi_c = get_single_roi_interactively(clean_img, "3. Zone Centre")
    if not roi_c: return
    x, y, w, h = roi_c
    zone_c = clean_img[y:y+h, x:x+w].copy()
    
    print("\n[ETAPE 4] Tracez les boîtes sur le texte. Après chaque boîte, tapez le texte lu.")
    detections_c = get_rois_and_texts_interactively(zone_c, "4. Détections Centre")
    
    # Rendu final
    res_hg = draw_boxes_with_labels(zone_hg, detections_hg)
    res_c = draw_boxes_with_labels(zone_c, detections_c)
    
    # Assemblage
    def add_title(img, title):
        header_h = 70
        header = np.full((header_h, img.shape[1], 3), 245, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(header, title, (30, 45), font, 1.2, (30,30,30), 2)
        cv2.line(header, (0, header_h-1), (img.shape[1], header_h-1), (150, 150, 150), 2)
        return np.vstack((header, img))
        
    res_hg = add_title(res_hg, "Extraction des métadonnées (Commune, Feuille, Section)")
    res_c = add_title(res_c, "Extraction des numéros de parcelles")
    
    target_w = max(res_hg.shape[1], res_c.shape[1])
    def pad(img, tw):
        if img.shape[1] < tw:
            return np.hstack((img, np.full((img.shape[0], tw-img.shape[1], 3), 255, dtype=np.uint8)))
        return img
        
    final = np.vstack((pad(res_hg, target_w), np.full((20, target_w, 3), 200, dtype=np.uint8), pad(res_c, target_w)))
    
    cv2.imwrite(output_path, final)
    print(f"\n[SUCCES] L'image est prête : {output_path}")

if __name__ == "__main__":
    main()
