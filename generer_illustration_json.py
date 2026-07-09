import tkinter as tk
import cv2
import fitz
import numpy as np
import os
import json
from PIL import Image, ImageTk

# Chemins des fichiers
pdf_path = r"inputs\geofoncier_dmpc_07289_000_608.pdf"
json_path = r"outputs\geofoncier_dmpc_07289_000_608_plan_DMPC.json"
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

from tkinter import simpledialog

def get_roi_for_field(img_bgr, field_name, field_text):
    """Demande à l'utilisateur de tracer le rectangle précis pour un champ donné et permet de modifier le texte"""
    roi = None
    final_text = field_text
    
    root = tk.Tk()
    root.title(f"Sélection précise pour : {field_name}")
    
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
    
    # Instructions claires en haut de la fenêtre
    header = tk.Frame(root)
    header.pack(fill=tk.X, pady=10)
    tk.Label(header, text=f"Champ actuel : {field_name.upper()}", font=("Arial", 14, "bold"), fg="blue").pack()
    tk.Label(header, text=f"Texte lu par défaut : {field_text}", font=("Arial", 12, "italic")).pack()
    tk.Label(header, text="Tracez le rectangle PARFAIT sur l'image, puis appuyez sur ENTREE. (ECHAP pour ignorer)", font=("Arial", 11), fg="red").pack()
    
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
        nonlocal roi, final_text
        if rect_id:
            x1, y1, x2, y2 = canvas.coords(rect_id)
            bw, bh = abs(x2-x1), abs(y2-y1)
            if bw > 5 and bh > 5:
                roi = (int(min(x1,x2)/scale), int(min(y1,y2)/scale), int(bw/scale), int(bh/scale))
                # Popup pour modifier le texte avec la valeur initiale du JSON
                new_text = simpledialog.askstring(
                    "Validation du texte", 
                    f"Texte à afficher pour {field_name} :", 
                    initialvalue=field_text,
                    parent=root
                )
                if new_text is not None:
                    final_text = new_text
                else:
                    roi = None # Annulé
            root.quit()
            root.destroy()
            
    def on_escape(e):
        root.quit()
        root.destroy()
        
    root.bind('<Return>', on_enter)
    root.bind('<Escape>', on_escape)
    root.mainloop()
    return roi, final_text

def draw_boxes_with_labels(img, detections):
    res = img.copy()
    for (x, y, w, h, text) in detections:
        # 1. Rectangle de détection
        cv2.rectangle(res, (x, y), (x+w, y+h), (0, 200, 0), 3)
        
        # 2. Préparation du texte
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.9
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 3. Dessiner le fond (label) juste au-dessus du rectangle
        box_y_top = max(y - text_h - 15, 0)
        box_y_bottom = box_y_top + text_h + 15
        
        # Fond vert
        cv2.rectangle(res, (x, box_y_top), (x + text_w + 20, box_y_bottom), (0, 200, 0), -1)
        
        # Texte en blanc
        cv2.putText(res, text, (x + 10, box_y_bottom - 8), font, font_scale, (255, 255, 255), thickness)
    return res

def main():
    print("1. Chargement des données JSON...")
    if not os.path.exists(json_path):
        print(f"Erreur : Le fichier JSON {json_path} est introuvable.")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    champs = data.get("champs", {})
    
    print("2. Chargement de l'image PDF...")
    clean_img = load_clean_image_from_pdf(pdf_path)
    
    # On va découper le processus en deux zones, comme vous l'avez demandé
    # --- ZONE HAUT GAUCHE ---
    print("\n--- ZONE HAUT-GAUCHE (Commune, Section, Feuille) ---")
    
    h_img, w_img = clean_img.shape[:2]
    zone_hg = clean_img[0:int(h_img*0.4), 0:int(w_img*0.5)].copy()
    
    champs_hg = ["commune", "section", "feuille", "echelle"]
    detections_hg = []
    
    import ast

    def process_champ(champ_name, zone_img, detections_list):
        if champ_name not in champs:
            return
            
        # On utilise "valeur" comme demandé (la bonne information normalisée)
        texte_lu = champs[champ_name].get("valeur", "")
        
        # Si c'est une liste sous forme de texte (ex: "['611', '106']")
        items = []
        if isinstance(texte_lu, str) and texte_lu.strip().startswith("[") and texte_lu.strip().endswith("]"):
            try:
                items = ast.literal_eval(texte_lu)
            except:
                items = [texte_lu]
        elif isinstance(texte_lu, list):
            items = texte_lu
        else:
            items = [str(texte_lu)]
            
        # On filtre les valeurs vides
        items = [i for i in items if str(i).strip()]
        
        # --- NOUVEAUTÉ : Forcer la saisie des parcelles si le JSON est vide ---
        if not items:
            if champ_name == "parcelles":
                print(f"-> Aucune parcelle dans le JSON. Ajout de 3 emplacements manuels.")
                items = ["(A définir)", "(A définir)", "(A définir)"]
            else:
                print(f"-> Aucune donnée trouvée pour {champ_name}.")
                return
            
        # Pour chaque élément (ex: chaque parcelle), on demande une boîte
        for idx, item in enumerate(items):
            display_name = champ_name
            if len(items) > 1:
                display_name = f"{champ_name} ({idx+1}/{len(items)})"
                
            res = get_roi_for_field(zone_img, display_name, str(item))
            if res[0] is not None:
                roi, final_text = res
                x, y, w, h = roi
                detections_list.append((x, y, w, h, final_text))

    for champ in champs_hg:
        process_champ(champ, zone_hg, detections_hg)

    # --- ZONE CENTRE ---
    print("\n--- ZONE CENTRE (Parcelles) ---")
    
    zone_c_y_start = int(h_img*0.2)
    zone_c_y_end = int(h_img*0.9)
    zone_c_x_start = int(w_img*0.1)
    zone_c_x_end = int(w_img*0.9)
    zone_c = clean_img[zone_c_y_start:zone_c_y_end, zone_c_x_start:zone_c_x_end].copy()
    
    champs_c = ["parcelles", "nouvelles_parcelles"]
    detections_c = []
    
    for champ in champs_c:
        process_champ(champ, zone_c, detections_c)
                
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
        
    res_hg = add_title(res_hg, "Informations Administratives (Haut-Gauche)")
    res_c = add_title(res_c, "Numéros de Parcelles (Centre)")
    
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
