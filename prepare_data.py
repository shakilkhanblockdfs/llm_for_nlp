import os
import sys
import csv
import re
import random

# ---------------- CONFIG ----------------
SAMPLE_DENOMINATOR = 10000   # 1/10000 of total samples (100x smaller than original)
RANDOM_SEED = 42          # set to None for non-deterministic
# ----------------------------------------

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

csv.field_size_limit(sys.maxsize)
os.makedirs("data", exist_ok=True)

def clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", "")
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

kept = 0
skipped = 0
sampled = 0

with open("data/europarl-v10.de-en.tsv", encoding="utf-8", errors="ignore") as f, \
     open("data/train.en", "w", encoding="utf-8") as f_en, \
     open("data/train.de", "w", encoding="utf-8") as f_de:

    reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        if len(row) < 2:
            skipped += 1
            continue

        de = clean(row[0])
        en = clean(row[1])

        if not de or not en or len(de) < 3 or len(en) < 3:
            skipped += 1
            continue

        kept += 1

        # ---- sampling logic: keep 1/N ----
        if random.randint(1, SAMPLE_DENOMINATOR) != 1:
            continue
        # ----------------------------------

        f_en.write(en + "\n")
        f_de.write(de + "\n")
        sampled += 1

print(f"Total valid aligned pairs: {kept}")
print(f"Sampled (1/{SAMPLE_DENOMINATOR}): {sampled}")
print(f"Skipped rows: {skipped}")
