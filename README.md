<div align="center">

# 🏃 India Runs — Candidate Ranking System

**AI-powered candidate discovery built for the India Runs Data & AI Challenge**

*Ranks 100,000+ resumes for a specialized AI Engineering role using a hybrid pipeline combining feature engineering, semantic search, and FAISS vector retrieval.*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embedding_Model-blue?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0064B5?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

</div>

---

## 📌 Overview

The challenge: identify the best-fit candidates for an **AI Engineering role** from a dataset of **100,000+ resumes** — accurately, explainably, and at scale.

This system solves that with a three-stage pipeline:

1. **Feature Scoring** — structured evaluation across role-critical signals (ML production experience, retrieval expertise, open-source contributions, etc.)
2. **Semantic Retrieval** — dense embeddings via `all-MiniLM-L6-v2` + FAISS nearest-neighbour search
3. **Hybrid Ranking** — a weighted combination of semantic similarity and feature scores, with recruiter-readable explanations for every top candidate

---

## ✨ Highlights

| | |
|---|---|
| 🔍 **Semantic Search** | `all-MiniLM-L6-v2` encodes candidate profiles into 384-dim dense embeddings |
| ⚡ **Scalable Retrieval** | FAISS delivers sub-second nearest-neighbour search over 100K+ vectors |
| 🧠 **Hybrid Ranking** | 60% semantic similarity + 40% domain-specific feature score |
| 📋 **Explainable Results** | Natural-language recruiter reasoning attached to every recommendation |
| 📦 **Modular Pipeline** | Feature extraction, embedding generation, indexing, ranking, and validation are separated into independent stages |
---
## 🏗️ Architecture

```

candidates.jsonl
                           │
                           ▼
              ┌────────────────────────┐
              │   batch_processor.py   │
              │  Feature Extraction &  │
              │        Scoring         │
              └────────────┬───────────┘
                           │
                           ▼
              final_ranked_leaderboard.jsonl
                           │
              ┌────────────┴──────────────────┐
              │                               │
              ▼                               ▼
       Feature Scores          ┌──────────────────────────────┐
                               │ process_candidate_embeddings │
                               │          .py                 │
                               └──────────────┬───────────────┘
                                              │
                                              ▼
                                candidate_embeddings.jsonl
                                              │
                                              ▼
                                   ┌──────────────────┐
                                   │  faiss_index.py  │
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   candidates.index
                                  faiss_metadata.pkl
                                            │
                                            ▼
                                   ┌────────────────┐
                                   │    rank.py     │
                                   │ Hybrid Scoring │
                                   │  & Re-ranking  │
                                   └───────┬────────┘
                                           │
                                           ▼
                                    submission.csv
```

---

## 🧠 Ranking Strategy

### 1 — Feature-Based Scoring

Each candidate is evaluated across eight domain-critical signals:

| Signal | Description |
|--------|-------------|
| 🏭 Production ML | Real-world ML systems in production |
| 🔎 Retrieval Expertise | RAG, dense/sparse retrieval, search systems |
| 🗄️ Vector DB Experience | Pinecone, Weaviate, Qdrant, Chroma, etc. |
| 📊 Ranking & RecSys | Recommendation and ranking system work |
| 🧪 Evaluation Frameworks |NDCG, MRR, MAP, A/B testing, benchmarking pipelines |
| 🌐 Open-Source Contributions | GitHub activity, published libraries |
| 📈 Experience Fit | Years and depth of relevant experience |
| 🤝 Recruiter Signals | Engagement, availability, response signals |

### 2 — Semantic Retrieval

Candidate profiles are embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed with FAISS for efficient top-K retrieval against the role description.

### 3 — Hybrid Ranking

```
Final Score = 0.6 × Semantic Similarity + 0.4 × Feature Score
```

This blend ensures candidates are ranked by both *relevance to the role* and *role-critical domain depth*, not either alone.

---

## 📁 Project Structure

```
india-runs-candidate-ranking/
│
├── src/
│   ├── batch_processor.py                 # Feature extraction + leaderboard generation
│   ├── feature_extractor.py               # Core feature scoring logic
│   ├── process_candidate_embeddings.py    # Embedding generation (all-MiniLM-L6-v2)
│   ├── faiss_index.py                     # FAISS index build + serialisation
│   └── rank.py                            # Hybrid ranking + submission output
│
├── candidate_schema.json                  # Candidate profile schema
├── sample_submission.csv                  # Example output format
├── validate_submission.py                 # Submission format validator
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` (384-dim) |
| Vector Search | `faiss-cpu` |
| NLP | Hugging Face Transformers |
| Data Processing | NumPy · Scikit-learn |

---

## ⚙️ Installation

```bash
git clone https://github.com/Noman-Ahmad25/india-runs-candidate-ranking.git
cd india-runs-candidate-ranking

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🤖 Model Setup

This project requires the **`all-MiniLM-L6-v2`** model loaded locally.

**Download:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Create a `local_model/` directory at the project root and place the model files inside:

```
local_model/
├── config.json
├── model.safetensors
├── modules.json
├── sentence_bert_config.json
├── tokenizer_config.json
├── vocab.txt
└── 1_Pooling/
    └── config.json
```

> `local_model/` is excluded from version control due to file size.

---

## 🚀 Usage

Run the pipeline in four stages:

### Step 1 — Feature Scoring

```bash
python3 src/batch_processor.py
```
Output: `final_ranked_leaderboard.jsonl`

### Step 2 — Generate Embeddings

```bash
python3 src/process_candidate_embeddings.py
```
Output: `candidate_embeddings.jsonl`

### Step 3 — Build FAISS Index

```bash
python3 src/faiss_index.py
```
Outputs: `candidates.index`, `faiss_metadata.pkl`

### Step 4 — Generate Final Rankings

```bash
python3 src/rank.py \
  --candidates candidates.jsonl \
  --out submission.csv
```
Output: `submission.csv`

### Step 5 — Validate Submission

```bash
python3 validate_submission.py submission.csv
```

---

## 📊 Results

- 🏆 Ranked **100,000+ candidates** end-to-end through the full hybrid pipeline
- 💬 Generated **explainable, recruiter-friendly recommendations** for every top result
- ⚡ FAISS enables efficient nearest-neighbour search over 100K+ vectors
- 🔬 Combined semantic relevance with structured domain signals for higher-precision ranking

---

## 🗂️ Excluded from Version Control

| File / Directory | Reason |
|------------------|--------|
| `candidates.jsonl` | Challenge dataset (licensing) |
| `candidate_embeddings.jsonl` | Generated artifact (size) |
| `candidates.index` | Generated artifact (size) |
| `faiss_metadata.pkl` | Generated artifact (size) |
| `final_ranked_leaderboard.jsonl` | Generated artifact (size) |
| `submission.csv` | Generated output |
| `local_model/` | Model weights (size) |

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
