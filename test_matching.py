
import unicodedata
import re

_VISUAL_GROUPS = [
    set("IL1J"), set("NMUVW"), set("O0DQG"), set("S58"), set("FT7"),
    set("B6H"), set("PRK"), set("AOU"), set("ECG"), set("VY"), set("I1EYL")
]

def visual_similarity(s1, s2):
    n, m = len(s1), len(s2)
    if n == 0 or m == 0: return 0.0 if n != m else 100.0
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = float(i)
    for j in range(m + 1): dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c1, c2 = s1[i-1], s2[j-1]
            if c1 == c2: cost = 0.0
            else:
                is_visual = any(c1 in group and c2 in group for group in _VISUAL_GROUPS)
                cost = 0.25 if is_visual else 1.0
            dp[i][j] = min(dp[i-1][j] + 1.0, dp[i][j-1] + 1.0, dp[i-1][j-1] + cost)
    return max(0.0, (1.0 - (dp[n][m] / max(n, m))) * 100)

def test(ocr, target):
    score = visual_similarity(ocr.upper(), target.upper())
    print(f"OCR: {ocr:10} | Target: {target:10} | Score: {score:.1f}%")

print("--- Test de Similarité Visuelle ---")
test("Luvinas", "Juvinas")
test("Bonzer", "Uzer")
test("Pomzer", "Uzer")
test("Meyras", "Heyras")
test("Barnas", "Barnas")
test("Barnas", "Barnas")
test("Barnas", "Varnas")
test("Pradges", "Prades")
test("St Martin", "Saint Martin") # Note: normalization should handle this, but let's see distance
test("ii", "ll")
