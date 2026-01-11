import pandas as pd
import sqlite3
from transformers import pipeline

# =============================
# 1. Load CSV Dataset (FIXED)
# =============================
DATA_PATH = r"C:\Users\HP\Downloads\archive (5)\ner.csv"

# Read CSV instead of Excel
df = pd.read_csv(DATA_PATH, header=None)

df.columns = [
    "sentence_id",
    "sentence",
    "pos",
    "gold_tag"
]

# Keep unique sentences only
sentences = (
    df[["sentence_id", "sentence"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Loaded {len(sentences)} unique sentences")

# =============================
# 2. Load Pretrained NER Model
# =============================
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

# =============================
# 3. Run NER
# =============================
all_entities = []

for _, row in sentences.iterrows():
    sentence_id = row["sentence_id"]
    text = row["sentence"]

    predictions = ner(text)

    for ent in predictions:
        all_entities.append({
            "sentence_id": sentence_id,
            "sentence": text,
            "entity_text": ent["word"],
            "entity_type": ent["entity_group"],
            "confidence": round(ent["score"], 4)
        })

print(f"Extracted {len(all_entities)} entities")

# =============================
# 4. Convert to DataFrame
# =============================
entities_df = pd.DataFrame(all_entities)

print("\nSample extracted entities:")
print(entities_df.head())

# =============================
# 5. Save to SQLite Database
# =============================
conn = sqlite3.connect("ner_entities.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id INTEGER,
    sentence TEXT,
    entity_text TEXT,
    entity_type TEXT,
    confidence REAL
)
""")

entities_df.to_sql(
    "entities",
    conn,
    if_exists="append",
    index=False
)

conn.commit()
conn.close()

print("\nEntities saved to ner_entities.db")

# =============================
# 6. Simple Analytics
# =============================
conn = sqlite3.connect("ner_entities.db")
cursor = conn.cursor()

print("\nEntity counts by type:")
for row in cursor.execute("""
    SELECT entity_type, COUNT(*) 
    FROM entities
    GROUP BY entity_type
    ORDER BY COUNT(*) DESC
"""):
    print(row)

print("\nMost frequent entities:")
for row in cursor.execute("""
    SELECT entity_text, COUNT(*) AS cnt
    FROM entities
    GROUP BY entity_text
    ORDER BY cnt DESC
    LIMIT 10
"""):
    print(row)

conn.close()