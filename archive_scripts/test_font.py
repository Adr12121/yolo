from PIL import Image, ImageDraw, ImageFont
import os

img = Image.new('RGB', (1000, 250), (30, 30, 30))
draw = ImageDraw.Draw(img)

try:
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    print("Police DejaVu chargée avec succès.")
except Exception as e:
    print(f"Erreur police: {e} -> utilisation police par défaut")
    font_big = ImageFont.load_default()
    font_small = ImageFont.load_default()

draw.text((15, 10), "TYPE: Plan de Bornage", font=font_small, fill=(200, 200, 200))
draw.text((15, 50), "COMMUNE: ANDANCE", font=font_big, fill=(0, 255, 80))
draw.text((15, 110), "PROPRIETAIRE: MARTIN Roger", font=font_small, fill=(255, 255, 255))
draw.text((15, 145), "N DOSSIER: 2024-00123", font=font_small, fill=(255, 255, 255))

out = '/mnt/c/Users/Topo_4/Documents/AT_PFE/Anti/yolo/outputs/test_font.jpg'
img.save(out)
print(f"Image de test sauvegardée : {out}")
