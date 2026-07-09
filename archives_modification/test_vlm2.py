import cv2
import json
import base64
import subprocess
import os

def test_vlm():
    from plan_classifier import _extract_with_vlm, _get_vlm_prompt, _get_vlm_model
    import numpy as np
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    crops_data = {
        "commune": {"zone": [0.0, 0.0, 0.70, 0.35], "brut": ""}
    }
    
    # Let's write the exact inner code to see where it breaks
    import cv2
    z = crops_data["commune"]["zone"]
    h_img, w_img = img.shape[:2]
    x0, y0 = int(z[0]*w_img), int(z[1]*h_img)
    x1, y1 = int(z[2]*w_img), int(z[3]*h_img)
    crop_img = img[y0:y1, x0:x1]
    _, buffer = cv2.imencode('.jpg', crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    prompt = _get_vlm_prompt("DMPC", "commune")
    
    payload = {"model": "llama3.2-vision", "prompt": prompt, "images": [img_base64],
               "stream": False, "options": {"temperature": 0.0, "num_predict": 80, "seed": 42}}
    payload_path = "test_payload.json"
    with open(payload_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp)
    cmd = ["curl.exe", "-s", "-X", "POST", "http://127.0.0.1:11434/api/generate",
           "-H", "Content-Type: application/json", "-d", f"@{payload_path}"]
    
    print("Running curl...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    print("Return code:", res.returncode)
    print("Stdout (first 200 chars):", res.stdout[:200])
    if res.returncode == 0:
        ojson = json.loads(res.stdout)
        print("Response:", ojson.get("response"))

if __name__ == '__main__':
    test_vlm()
