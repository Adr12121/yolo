import re

full_text = """
6462 T
Saut_
foxmune
'08
anc
Mod. 30 Caa
Sept. 19701
Section
A
N'
d' ordre
du document
6o.?
4 
Feuille
d arpentage
Tableau
modifier "
Echelle
AceC
c'assemblage
sans changt
107
tBv,
cotet;  ^
105
Vo i
f
LE Si
CH
aa9<  e 7
104
110
106
'03
102
101
66
L'HUBERT
1OO
43
Lot c
Lo 8
69
12 93
70
Lot A
1/.53
285
286
73
des
72
75
78
26
A|S P R E S
25
76
93
24
92
Voir la rubrique
INFORMATION DES PROPRITAIRES ,
au dos
de la chemise 6463
Extrait
du   plan
minute
tabli
Document d'arpentage dresse
par le Bureau Ou Cadastrel 
CERTIFICATION
par M .HAReai
Caog
(Art. 25 du dcret n' 55-471
au 30 avril 1955)
G2on 
Gadaotrel   .
Vo d'ordre au
Le prsent document d'arpentage,
certifie par les propnitaires soussigns n ,
tsbi
2
2
7(J
tatation des
aroas._[ f~s
d'aprs tes indications qu'ils Ont fournies Ju bureaul
Date
J4/ !9/ 93
Cachet
du Service
d'origine
en conformit dun piquetage qu'ils ont effectue sur le terrain  .
Signature :
(outs
 -d'aprs un Plan d'arpentage Ou de bornage; dont copie ci-tointe, dresoe+ 
cehvail
HARROIS
Ge8 8jinga
Dar
gomtre
"GEdMETRY
EXPERT DPLG
Ep 820
Kau
0/212.97
Chemir
de Constantine
O7cuc Pnvr5
CEDEX
TI
CostaR FEnt
07200   AUBEIaS
75 93 65 57.
"""

m = re.search(r"(?i)par\s+M\s*\.\s*([A-Z\xc0-\xdd][A-Za-z\xc0-\xff\s\-\.]{2,30})", full_text)
if m: print("Geometre matched:", m.group(1))
else: print("Geometre NO MATCH")

m_nord = re.search(r"(?i)(?:n[o\xb0°'’]?\s*d['’\s]*(?:ordre|arpentage)|num[e\xe9]ro\s*d['’\s]*(?:ordre|arpentage)|da\s*n[o\xb0°'’]?)[^\d]*([\dOo0]{1,6}[A-Za-z]?(?:\s*[_\-]\s*[\dOo0]{1,6}[A-Za-z]?)*(?:\s*\(\d+\))?)", full_text)
if m_nord: print("N ordre matched:", m_nord.group(1))
else: print("N ordre NO MATCH")

