import subprocess
import sys

STEPS = [
    ("Feature Scoring", [
        "python3",
        "src/batch_processor.py"
    ]),
    ("Generate Embeddings", [
        "python3",
        "src/process_candidate_embeddings.py"
    ]),
    ("Build FAISS Index", [
        "python3",
        "src/faiss_index.py"
    ]),
    ("Generate Final Rankings", [
        "python3",
        "src/rank.py",
        "--candidates",
        "candidates.jsonl",
        "--out",
        "submission.csv",
    ]),
]


def run_step(name, command):
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"{'=' * 60}")

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n❌ {name} failed.")
        sys.exit(result.returncode)

    print(f"✅ {name} completed.")


def main():
    print("Starting end-to-end candidate ranking pipeline...")

    for name, command in STEPS:
        run_step(name, command)

    print("\n🎉 Pipeline completed successfully!")
    print("Output generated: submission.csv")


if __name__ == "__main__":
    main()