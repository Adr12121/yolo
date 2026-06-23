import sys
with open(r"c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py", "r", encoding="utf-8", errors="replace") as f:
    lines = f.readlines()
for i, line in enumerate(lines[1055:1070], start=1056):
    print(f"{i}: {line.rstrip()}")
