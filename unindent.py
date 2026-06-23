import sys

with open('c:\\Users\\Topo_4\\Documents\\AT_PFE\\Anti\\yolo\\plan_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
in_fallback = False
for line in lines:
    if 'Fallback Global Infaillible pour G' in line or 'Fallback Global Infaillible pour Geometres' in line:
        in_fallback = True
    
    if in_fallback:
        # Stop at the end of the fallbacks
        if line.strip() == '# ── FULL PAGE VLM FALLBACK (Phase 4) ──' or 'all_missing = ' in line:
            in_fallback = False
            new_lines.append(line)
            continue
            
        # If it's a fallback line, dedent by 4 spaces
        if line.startswith(' ' * 12):
            new_lines.append(line[4:])
        elif line.startswith(' ' * 8):
            new_lines.append(line)
        elif line.strip() == '':
            new_lines.append(line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('c:\\Users\\Topo_4\\Documents\\AT_PFE\\Anti\\yolo\\plan_classifier.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Done!')
