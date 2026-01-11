import pandas as pd
import sqlite3
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# =============================
# 1. Load CSV Dataset
# =============================
DATA_PATH = r"C:\Users\HP\Downloads\archive (5)\ner.csv"

df = pd.read_csv(DATA_PATH, header=None)
df.columns = ["sentence_id", "sentence", "pos", "gold_tag"]

sentences = (
    df[["sentence_id", "sentence"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"Loaded {len(sentences)} unique sentences")

# =============================
# 2. Load Local LLM (Ollama)
# =============================
llm = ChatOllama(model="mistral", temperature=0)

# =============================
# 3. NER Prompt Template
# =============================
NER_PROMPT = """
Extract named entities from the text below.

Return ONLY valid JSON in this format:
[
  {{
    "text": "entity text",
    "label": "PERSON | ORG | LOCATION | DATE | MISC"
  }}
]

Text:
"{text}"
"""

# =============================
# 4. Run NER using LLM
# =============================
all_entities = []

for _, row in sentences.iterrows():
    sentence_id = row["sentence_id"]
    text = row["sentence"]

    response = llm.invoke(
        [HumanMessage(content=NER_PROMPT.format(text=text))]
    )

    raw_output = response.content.strip()

    # Extract JSON safely
    try:
        json_text = re.search(r"\[.*\]", raw_output, re.S).group()
        entities = json.loads(json_text)
    except Exception:
        continue  # skip malformed output

    for ent in entities:
        all_entities.append({
            "sentence_id": sentence_id,
            "sentence": text,
            "entity_text": ent["text"],
            "entity_type": ent["label"]
        })

print(f"Extracted {len(all_entities)} entities")

# =============================
# 5. Convert to DataFrame
# =============================
entities_df = pd.DataFrame(all_entities)
print("\nSample entities:")
print(entities_df.head())

# =============================
# 6. Save to SQLite
# =============================
conn = sqlite3.connect("ner_entities_llm.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id INTEGER,
    sentence TEXT,
    entity_text TEXT,
    entity_type TEXT
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

print("\nEntities saved to ner_entities_llm.db")
