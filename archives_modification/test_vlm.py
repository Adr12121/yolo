import cv2
import json
import base64
import subprocess
import os

def test_vlm():
    from plan_classifier import _extract_with_vlm, _get_vlm_prompt, _get_vlm_model
    img = cv2.imread('inputs/EXEMPLE_HARROIS.pdf') # can't imread pdf
    # Let's just create a dummy image
    import numpy as np
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    crops_data = {
        "commune": {"zone": [0.0, 0.0, 0.70, 0.35], "brut": ""}
    }
    def dummy_val(f, v): return v
    res = _extract_with_vlm(img, "DMPC", dummy_val, crops_data)
    print("VLM RES:", res)

if __name__ == '__main__':
    test_vlm()
