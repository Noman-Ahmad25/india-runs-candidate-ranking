# India Runs Candidate Ranking

AI-powered candidate discovery and ranking system built for the India Runs Data & AI Challenge.

This project processes large-scale candidate datasets and identifies the most relevant candidates for a target role using a hybrid approach that combines semantic retrieval, vector search, and feature-engineered scoring.

## Features

- Semantic candidate search using Sentence Transformers
- Dense vector retrieval using FAISS
- Hybrid ranking (semantic similarity + feature-based scoring)
- Candidate feature extraction and evaluation
- Explainable candidate recommendations
- Scalable pipeline designed for 100K+ resumes

## Tech Stack

- Python
- FAISS
- Sentence Transformers
- Hugging Face Transformers
- PyTorch
- NumPy
- Scikit-learn

## Architecture

```text
Candidate Profiles
        │
        ▼
Feature Extraction
        │
        ▼
Embedding Generation
(all-MiniLM-L6-v2)
        │
        ▼
FAISS Vector Index
        │
        ▼
Semantic Retrieval
        │
        ▼
Hybrid Ranking
(Semantic + Feature Scores)
        │
        ▼
Top Candidate Selection
```

## Project Highlights

- Processed and ranked candidates from a dataset containing over 100,000 resumes.
- Built a hybrid ranking framework combining dense vector similarity and domain-specific feature engineering.
- Leveraged Sentence Transformers (all-MiniLM-L6-v2) to generate semantic embeddings for candidate profiles.
- Used FAISS for efficient large-scale vector retrieval.
- Generated explainable candidate recommendations with recruiter-focused reasoning.

## Project Structure

```text
.
├── src/
│   ├── feature_extractor.py
│   ├── process_candidate_embeddings.py
│   ├── faiss_index.py
│   └── rank.py
│
├── candidate_schema.json
├── sample_candidates.json
├── requirements.txt
├── validate_submission.py
└── README.md
```

## Installation

```bash
git clone https://github.com/your-username/india-runs-candidate-ranking.git
cd india-runs-candidate-ranking

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Model Setup

This project uses the **all-MiniLM-L6-v2** Sentence Transformer model.

Download the model from:

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Create a directory named `local_model` in the project root and place the downloaded model files inside:

```text
local_model/
├── config.json
├── model.safetensors
├── modules.json
├── sentence_bert_config.json
├── tokenizer_config.json
├── vocab.txt
└── 1_Pooling/
```

The `local_model/` directory is excluded from version control because of its size.

## Usage

### Generate Embeddings

```bash
python3 src/process_candidate_embeddings.py
```

Output:

```text
candidate_embeddings.jsonl
```

### Build FAISS Index

```bash
python3 src/faiss_index.py
```

Outputs:

```text
candidates.index
faiss_metadata.pkl
```

### Generate Final Rankings

```bash
python3 src/rank.py \
  --candidates candidates.jsonl \
  --out submission.csv
```

Output:

```text
submission.csv
```

### Validate Submission

```bash
python3 validate_submission.py submission.csv
```

## Results

- Ranked candidates from a pool of 100K+ resumes.
- Combined semantic retrieval with recruiter-oriented feature scoring.
- Produced explainable recommendations for top-ranked candidates.
- Enabled scalable candidate discovery using vector search and hybrid ranking.

## Repository Notes

The following files are excluded from the repository:

- Original challenge dataset (`candidates.jsonl`)
- Generated embeddings (`candidate_embeddings.jsonl`)
- FAISS index files (`candidates.index`, `faiss_metadata.pkl`)
- Generated ranking outputs
- Local model weights (`local_model/`)

These artifacts are omitted due to storage and licensing constraints.

## Resume Highlights

- Built an AI-powered candidate ranking system to process and rank 100K+ resumes using semantic retrieval and feature-engineered scoring.
- Developed a hybrid ranking pipeline combining Sentence Transformers, FAISS vector search, and recruiter-focused candidate evaluation.
- Engineered scalable embedding and retrieval workflows for large-scale talent discovery.
- Generated explainable candidate recommendations through semantic matching and domain-specific ranking signals.

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.