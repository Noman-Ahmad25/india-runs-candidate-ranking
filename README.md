# 🏃 India Runs — Candidate Ranking System

> AI-powered candidate discovery and ranking built for the **India Runs Data & AI Challenge**.  
> Processes 100K+ resumes and surfaces the most relevant candidates for a target role using a hybrid pipeline combining semantic search, dense vector retrieval, and feature-engineered scoring.

---

## ✨ Highlights

| | |
|---|---|
| 🔍 **Semantic Search** | Sentence Transformers (`all-MiniLM-L6-v2`) encode candidate profiles into dense embeddings |
| ⚡ **Scalable Retrieval** | FAISS enables sub-second nearest-neighbour search over 100K+ vectors |
| 🧠 **Hybrid Ranking** | Combines semantic similarity scores with domain-specific feature engineering |
| 📋 **Explainable Results** | Recruiter-focused reasoning accompanies every top-ranked recommendation |
| 📦 **Pipeline Ready** | Modular, end-to-end pipeline from raw profiles to a ranked `submission.csv` |

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────┐
│          Candidate Profiles         │
│          (candidates.jsonl)         │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│          Feature Extraction         │
│       (feature_extractor.py)        │
│  Skills · Experience · Education    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│        Embedding Generation         │
│   (process_candidate_embeddings.py) │
│    all-MiniLM-L6-v2 · 384-dim       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│          FAISS Vector Index         │
│           (faiss_index.py)          │
│   IndexFlatIP · Metadata Store      │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         Semantic Retrieval          │
│      Top-K nearest neighbours       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│           Hybrid Ranking            │
│             (rank.py)               │
│  α · Semantic + β · Feature Score   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│       Ranked Submission File        │
│          (submission.csv)           │
└─────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
india-runs-candidate-ranking/
│
├── src/
│   ├── feature_extractor.py            # Extracts structured features from profiles
│   ├── process_candidate_embeddings.py # Generates sentence embeddings
│   ├── faiss_index.py                  # Builds and serialises the FAISS index
│   └── rank.py                         # Hybrid ranking + submission output
│
├── candidate_schema.json               # Schema definition for candidate profiles
├── sample_candidates.json              # Sample data for local testing
├── requirements.txt                    # Python dependencies
├── validate_submission.py              # Validates submission format
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Embeddings | `sentence-transformers` · `all-MiniLM-L6-v2` |
| Vector Search | `faiss-cpu` |
| ML Framework | PyTorch · Hugging Face Transformers |
| Numerics | NumPy · Scikit-learn |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/india-runs-candidate-ranking.git
cd india-runs-candidate-ranking

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🤖 Model Setup

This project uses **`all-MiniLM-L6-v2`** from Sentence Transformers — a compact, fast model that produces high-quality 384-dimensional semantic embeddings.

**Download from Hugging Face:**  
👉 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Place the downloaded model files in a `local_model/` directory at the project root:

```text
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

Run the pipeline in three steps:

### 1. Generate Embeddings

Encodes all candidate profiles into dense vectors.

```bash
python3 src/process_candidate_embeddings.py
```

**Output:** `candidate_embeddings.jsonl`

---

### 2. Build FAISS Index

Constructs and serialises the vector index for fast retrieval.

```bash
python3 src/faiss_index.py
```

**Outputs:**
```text
candidates.index      # FAISS binary index
faiss_metadata.pkl    # Candidate ID → metadata mapping
```

---

### 3. Generate Rankings

Runs hybrid ranking over retrieved candidates and writes the submission file.

```bash
python3 src/rank.py \
  --candidates candidates.jsonl \
  --out submission.csv
```

**Output:** `submission.csv`

---

### 4. Validate Submission

```bash
python3 validate_submission.py submission.csv
```

---

## 📊 Results

- ✅ Ranked candidates from a pool of **100,000+ resumes**
- ✅ Hybrid scoring fused semantic similarity with recruiter-oriented feature signals
- ✅ Produced **explainable, natural-language recommendations** for each top candidate
- ✅ Sub-second retrieval latency at scale via FAISS vector indexing

---

## 🗂️ Repository Notes

The following large or licensed files are excluded from version control:

| File / Directory | Reason Excluded |
|---|---|
| `candidates.jsonl` | Original challenge dataset (licensing) |
| `candidate_embeddings.jsonl` | Generated artefact (size) |
| `candidates.index` | Generated artefact (size) |
| `faiss_metadata.pkl` | Generated artefact (size) |
| `submission.csv` | Generated output |
| `local_model/` | Model weights (size) |

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
