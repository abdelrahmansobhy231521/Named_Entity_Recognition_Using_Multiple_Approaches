# Named Entity Recognition (NER) – Three Practical Approaches in Python

This project demonstrates **three different approaches to Named Entity Recognition (NER)** applied to the same real-world dataset.  
It highlights **classical NLP, transformer-based models, and LLM-based extraction**, making it suitable for **data science, NLP, and AI engineering portfolios**.

---

## 📌 Project Overview

Named Entity Recognition (NER) is the task of identifying and classifying entities such as:

- **PERSON**
- **ORGANIZATION**
- **LOCATION**
- **DATE**
- **MISC**

In this project, I applied **three NER approaches** to the same dataset and stored the results in a **SQLite database** for analysis and comparison.

---

## 📂 Dataset Description

- Format: **CSV**
- Columns:
  - `sentence_id` – Sentence identifier
  - `sentence` – Input text
  - `pos` – Part-of-speech tag
  - `gold_tag` – Ground-truth NER label

Duplicate sentences are removed before processing.

---

## 🧠 NER Approaches Implemented

---

## 1️⃣ Transformer-Based NER (Hugging Face – BERT)

**Model used:**  
`dslim/bert-base-NER`

### How it works
- Uses a pretrained **BERT model fine-tuned for NER**
- Performs **token-level classification**
- Returns entity type and confidence score

### Strengths
- High accuracy
- Deterministic results
- Confidence scores available

### Limitations
- Heavier model
- Slower than spaCy

### Best Use Case
> High-accuracy NER for research or production pipelines where precision matters.

---

## 2️⃣ spaCy-Based NER

**Model used:**  
`en_core_web_sm`

### How it works
- Uses spaCy’s optimized NER pipeline
- Fast entity extraction using pretrained statistical + neural models

### Strengths
- Very fast
- Lightweight
- Easy deployment

### Limitations
- No confidence scores
- Slightly lower accuracy than transformers

### Best Use Case
> Large-scale or real-time NLP systems where speed is critical.

---

## 3️⃣ LLM-Based NER (Ollama + Mistral)

**Model used:**  
Local LLM via **Ollama (Mistral)**

### How it works
- Prompts an LLM to extract entities
- Returns structured JSON output
- No training required

### Strengths
- Extremely flexible
- Easy to adapt to custom entity types
- Works well for low-resource or domain-specific text

### Limitations
- Slower
- Non-deterministic
- Requires output validation

### Best Use Case
> Rapid prototyping, complex documents, or custom NER schemas.

---

## 🗄️ Data Storage & Analytics

All extracted entities are stored in **SQLite** with fields:

- `sentence_id`
- `sentence`
- `entity_text`
- `entity_type`
- `confidence` (when available)

Example analytics:
- Entity frequency by type
- Most common entities
- Model comparison across approaches

---

## 📊 Technologies Used

- Python
- pandas
- spaCy
- Hugging Face Transformers
- LangChain
- Ollama (Mistral)
- SQLite
- Regular Expressions

---
