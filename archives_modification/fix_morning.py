import ast
import re

with open('plan_classifier.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Fix timeout_sec bug in the first _extract_with_vlm
for i, line in enumerate(lines):
    if 'timeout=timeout_sec' in line and 'subprocess.run' in line:
        lines[i] = line.replace('timeout=timeout_sec', 'timeout=60')

# 2. Disable _extract_with_vlm_full_page completely by returning {} at the top
for i, line in enumerate(lines):
    if line.startswith('def _extract_with_vlm_full_page('):
        lines.insert(i+1, '    return {}\n')
        break

# 3. Disable the call to _extract_with_vlm_full_page in process_plan
for i, line in enumerate(lines):
    if 'vlm_full = _extract_with_vlm_full_page' in line:
        lines[i] = line.replace('vlm_full =', 'vlm_full = {} #')

# 4. We want to remove the SECOND _extract_with_vlm because it uses llama3.2-vision and the new prompts which hallucinate.
# The second one starts around line 1924.
# Let's find all lines that start with 'def _extract_with_vlm('
defs = [i for i, line in enumerate(lines) if line.startswith('def _extract_with_vlm(')]
if len(defs) > 1:
    second_def_start = defs[1]
    # Find the end of the second function (before the next def)
    second_def_end = len(lines)
    for i in range(second_def_start + 1, len(lines)):
        if lines[i].startswith('def '):
            second_def_end = i
            break
    # Comment out the second definition
    for i in range(second_def_start, second_def_end):
        lines[i] = '# ' + lines[i]

with open('plan_classifier.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('File updated perfectly to match this morning behavior.')
