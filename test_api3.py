import json
import requests
from geofoncier_api import get_headers, BASE_URL

payload = {
    "enr_cab_createur": "1992C100001",
    "enr_ref_dossier": "TEST_DMPC",
    "enr_cab_detenteur": "1992C100001",
    "enr_ge_createur": "05141",
    "enr_code_insee": "07000",
    "enr_date_dossier": "2024-01-01",
    "op_code": ["Da"],
    "dmpc_ref": [{"dmpc_prefixe": "070", "dmpc_ref": "0385A"}]
}

url = f"{BASE_URL}/dossiersoge/dossiers/"
resp = requests.post(url, headers=get_headers(), json=payload)
print("TEST 0385A:", resp.status_code, resp.text)

payload["dmpc_ref"][0]["dmpc_ref"] = "385A"
resp = requests.post(url, headers=get_headers(), json=payload)
print("TEST 385A:", resp.status_code, resp.text)

payload["dmpc_ref"][0]["dmpc_ref"] = "3854A"
resp = requests.post(url, headers=get_headers(), json=payload)
print("TEST 3854A:", resp.status_code, resp.text)
