import re
import codecs

path = r'c:\Users\Topo_4\Documents\AT_PFE\Anti\yolo\plan_classifier.py'

with open(path, encoding='utf-8') as f:
    content = f.read()

# Let's look for the problematic block using a simpler regex.
# The block is: m_geo2 = re.search( r'(?:dress[e...
# We will just replace [A-ZÃ€-Ã ][a-zÃ -Ã¿] with [A-Za-z\xc0-\xff]
# and [A-ZÃ€-Ã ][A-Za-zÃ -Ã¿] with [A-Za-z\xc0-\xff]

original = content

# Common mojibake replacements:
replacements = [
    ('Ã€-Ã ', '\\xc0-\\xdd'),
    ('Ã -Ã¿', '\\xe0-\\xff'),
    ('Ã€-Ã¿', '\\xc0-\\xff'),
    ('€-Ã', '\\xc0-\\xdd') # specific one in the error
]

for old, new in replacements:
    content = content.replace(old, new)

# And specifically the line causing the error:
bad_regex_part = '[A-Z\xc0-\xdd][a-z\xe0-\xff]{2,}(?:\s+[A-Z\xc0-\xdd][A-Za-z\xe0-\xff]{2,}){0,3}'
# Actually, since it's already a raw string, we need to be careful. Let's just fix it at the string level.

if content != original:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Fixed some mojibake.")
else:
    print("No direct mojibake replacements made.")
    
# Test the file using our scanner
import ast

with open(path, 'r', encoding='utf-8') as f:
    source = f.read()

class RegexVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors = []
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 're' and node.func.attr in ('search', 'sub', 'match', 'findall', 'compile'):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    pat = node.args[0].value
                    try:
                        re.compile(pat)
                    except Exception as e:
                        self.errors.append((node.lineno, pat, e))
        self.generic_visit(node)

try:
    tree = ast.parse(source)
    visitor = RegexVisitor()
    visitor.visit(tree)
    if visitor.errors:
        print("Still found errors:")
        for line, pat, err in visitor.errors:
            print(f"Line {line}: {err} -> {repr(pat)}")
    else:
        print("All regexes are valid!")
except SyntaxError as e:
    print(f"Syntax error in file: {e}")

