<div align="center">

# 🏃 India Runs — Candidate Ranking System

**AI-powered candidate discovery built for the India Runs Data & AI Challenge**

*Processes and ranks candidates from a dataset of 100,000+ resumes using a hybrid pipeline combining feature engineering, semantic search, and FAISS vector retrieval.*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-Embeddings-blue?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0064B5?style=flat-square)
![Semantic Search](https://img.shields.io/badge/Semantic-Search-purple?style=flat-square)
![Hybrid Ranking](https://img.shields.io/badge/Hybrid-Ranking-orange?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-Demo-FF6F00?style=flat-square)
![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Spaces-FFD21E?style=flat-square)
![Challenge](https://img.shields.io/badge/India%20Runs-Data%20%26%20AI%20Challenge-success?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/Noman-Ahmad25/india-runs-candidate-ranking)

**GitHub:** [Repository](https://github.com/Noman-Ahmad25/india-runs-candidate-ranking)

</div>

---

## 📌 Overview

The challenge: identify the best-fit candidates for an **AI Engineering role** from a dataset of **100,000+ resumes** — accurately, explainably, and at scale.

This project implements a hybrid ranking system that combines:

- Feature extraction and recruiter-oriented scoring
- Dense semantic retrieval using Sentence Transformers
- FAISS vector similarity search
- Hybrid re-ranking using semantic and feature-based signals
- Explainable recruiter-facing recommendations

---

## ✨ Highlights

| | |
|---|---|
| 🔍 **Semantic Search** | `all-MiniLM-L6-v2` encodes candidate profiles into 384-dimensional dense embeddings |
| ⚡ **Scalable Retrieval** | FAISS enables efficient nearest-neighbour search across 100K+ candidate vectors |
| 🧠 **Hybrid Ranking** | Combines semantic similarity with feature-engineered candidate scoring |
| 📋 **Explainable Results** | Generates recruiter-friendly reasoning for every recommended candidate |
| 📦 **Modular Pipeline** | Feature extraction, embedding generation, indexing, and ranking are independent stages |
| 🌐 **Interactive Demo** | Deployed via Gradio on Hugging Face Spaces |

---

## 🏗️ Architecture

```
      candidates.jsonl                      candidates.jsonl
             │                                     │
             ▼                                     ▼
  ┌──────────────────────┐       ┌────────────────────────────────┐
  │  batch_processor.py  │       │ process_candidate_embeddings.py│
  │  Feature Extraction  │       │     Embedding Generation       │
  │      & Scoring       │       └─────────────────┬──────────────┘
  └──────────┬───────────┘                         │
             │                                     ▼
             ▼                         candidate_embeddings.jsonl
final_ranked_leaderboard.jsonl                     │
             │                                     ▼
             │                          ┌──────────────────────┐
             │                          │    faiss_index.py    │
             │                          │    Index Building    │
             │                          └──────────┬───────────┘
             │                                     │
             │                       ┌─────────────┴─────────────┐
             │                       │                           │
             │                candidates.index          faiss_metadata.pkl
             │                       │                           │
             │                       └─────────────┬─────────────┘
             │                                     │
             │                               local_model/
             │                                     │
             └───────────────────┬─────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │    rank.py      │
                        │ Hybrid Scoring  │
                        │  & Re-ranking   │
                        └────────┬────────┘
                                 │
                                 ▼
                          submission.csv
                                 │
                                 ▼
                        ┌─────────────────┐
                        │     app.py      │
                        │  Gradio Demo    │
                        └────────┬────────┘
                                 │
                                 ▼
                        Hugging Face Space

```
---

## 🧠 Ranking Strategy

### 1 — Feature-Based Scoring

Each candidate is evaluated across domain-critical signals:

| Signal | Description |
|--------|-------------|
| 🏭 Production ML | Real-world machine learning systems in production |
| 🔎 Retrieval Expertise | RAG, dense retrieval, sparse retrieval, search systems |
| 🗄️ Vector Databases | Pinecone, Weaviate, Qdrant, Chroma, Milvus, FAISS |
| 📊 Ranking Systems | Recommendation systems and ranking infrastructure |
| 🧪 Evaluation Frameworks | NDCG, MRR, MAP, A/B Testing, benchmarking |
| 🌐 Open Source | GitHub activity and contributions |
| 📈 Experience Fit | Relevant years and depth of experience |
| 🤝 Recruiter Signals | Availability and engagement indicators |

### 2 — Semantic Retrieval

Candidate profiles are embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed with FAISS for efficient similarity search against the target role description.

### 3 — Hybrid Ranking

```
Final Score = 0.6 × Semantic Similarity
            + 0.4 × Feature Score
```

This balances semantic relevance to the job description with domain-specific engineering expertise and recruiter-oriented signals.

---

## 📁 Project Structure

```
india-runs-candidate-ranking/
│
├── src/
│   ├── batch_processor.py                  # Feature extraction & leaderboard generation
│   ├── feature_extractor.py                # Core feature scoring logic
│   ├── process_candidate_embeddings.py     # Embedding generation
│   ├── faiss_index.py                      # FAISS index build & serialization
│   └── rank.py                             # Hybrid ranking & submission output
│
├── local_model/                            # all-MiniLM-L6-v2 model files
│   ├── config.json
│   ├── model.safetensors
│   ├── modules.json
│   ├── sentence_bert_config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── 1_Pooling/
│
├── app.py                                  # Gradio demo entrypoint
├── candidates.index                        # Precomputed FAISS index
├── faiss_metadata.pkl                      # Candidate metadata mapping
├── final_ranked_leaderboard.jsonl          # Precomputed feature scores
│
├── candidate_schema.json                   # Candidate data schema
├── sample_candidates.json                  # Sample input candidates
├── sample_submission.csv                   # Sample submission format
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| Embeddings | Sentence Transformers · `all-MiniLM-L6-v2` |
| Vector Search | FAISS |
| Data Processing | NumPy |
| Demo Interface | Gradio |

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

## 🤖 Model

This project uses the `all-MiniLM-L6-v2` Sentence Transformer model for embedding generation and semantic retrieval.

The model is included with the project for reproducible ranking and demo deployment.

---

## 🚀 Usage

Run the pipeline from raw candidate data to final ranked output:

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

## 🌐 Interactive Demo

A Gradio-based demo is deployed on Hugging Face Spaces. It loads precomputed artifacts and generates ranked candidate recommendations in real time.

**Artifacts used by the demo:**

| Artifact | Description |
|----------|-------------|
| `candidates.index` | FAISS vector index |
| `faiss_metadata.pkl` | Candidate ID → metadata mapping |
| `final_ranked_leaderboard.jsonl` | Precomputed feature scores |
| `local_model/` | `all-MiniLM-L6-v2` model weights |

👉 **Try it:** https://huggingface.co/spaces/Noman-Ahmad25/india-runs-candidate-ranking

---
## 📸 Demo Preview

The Gradio interface displays the top-ranked candidates along with ranking scores and recruiter-friendly reasoning generated by the hybrid retrieval pipeline.

![Demo Screenshot](asset/demo.png)

---

## 📊 Results

- 🏆 Processed and ranked candidates from a dataset of over **100,000 resumes**
- 💬 Generated recruiter-friendly reasoning for every recommended candidate
- ⚡ Enabled scalable semantic retrieval using FAISS vector indexing
- 🔬 Combined semantic relevance with structured feature engineering for higher-precision ranking
- 🌐 Deployed an interactive Gradio demo on Hugging Face Spaces

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
