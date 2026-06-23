import codecs
with codecs.open('tmp_rebuild/plan_classifier.py', 'r', 'utf-8', errors='ignore') as f:
    text = f.read()

text = text.replace("r'(?i)commune\\s+(?:de\\s+|d')([A-Za-z", "r'(?i)commune\\s+(?:de\\s+|d\\')([A-Za-z")

with codecs.open('tmp_rebuild/plan_classifier.py', 'w', 'utf-8') as f:
    f.write(text)
