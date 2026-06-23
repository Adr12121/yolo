import urllib.request
import xml.etree.ElementTree as ET

url = 'https://api2.geofoncier.fr/api/referentielsoge/wxs?SERVICE=WMS&REQUEST=GetCapabilities'
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    r = urllib.request.urlopen(req)
    xml_data = r.read()
    root = ET.fromstring(xml_data)
    ns = {'wms': 'http://www.opengis.net/wms'}
    for layer in root.findall('.//wms:Layer/wms:Layer', ns):
        name = layer.find('wms:Name', ns)
        title = layer.find('wms:Title', ns)
        if name is not None:
            print(f"Layer: {name.text} - Title: {title.text if title is not None else ''}")
except Exception as e:
    print('ERR:', e)
