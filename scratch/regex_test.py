import re

s = "Document d'arpentage dressé par M.GUIGUE"
p = r"(?:g[e\xe9]om[e\xe8]tre[\s\-]expert|cabinet\s+de\s+g[e\xe9]om[e\xe8]tre|le\s+soussign[e\xe9]\s+g[e\xe9]om[e\xe8]tre|g[e\xe9]om[e\xe8]tre\s*:|expert\s*:?|dress[e\xe9]\s+par\s*:?)[,\s:\-]+(?:M\.|Mme|Monsieur\s+)?([A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-\.]{2,60}?)(?:\s*,|\s*\n|\s*\.|$)"

m = re.search(p, s, re.IGNORECASE)
if m:
    print("Match:", m.groups())
else:
    print("No match")
