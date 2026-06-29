import gradio as gr
import subprocess
import pandas as pd
import sys

def run_ranking():
    result = subprocess.run([
        sys.executable,
        "src/rank.py",
        "--candidates",
        "demo_candidates.jsonl",
        "--out",
        "submission.csv"
    ])

    if result.returncode != 0:
        return pd.DataFrame({
            "Error": ["Ranking failed. Check logs."]
        })

    try:
        df = pd.read_csv("submission.csv")
    except FileNotFoundError:
        return pd.DataFrame({
            "Error": ["submission.csv not found."]
        })

    return df[
        ["rank", "candidate_id", "score", "reasoning"]
    ].head(10)

demo = gr.Interface(
    fn=run_ranking,
    inputs=None,
    outputs="dataframe",
    title="India Runs Candidate Ranking System",
    description=(
        "Ranks candidates for a specialized AI Engineering role using "
        "feature engineering, semantic retrieval, and FAISS vector search."
    )
)

demo.launch()