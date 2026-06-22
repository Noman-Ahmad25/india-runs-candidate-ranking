import json
import time

# Put this BEFORE the heavy imports so you know it's running!
print("⏳ Loading AI libraries (PyTorch, SciPy)... this might take 30 seconds...")

import torch
from sentence_transformers import SentenceTransformer

def process_candidate_embeddings(input_file: str, output_file: str):
    # Route processing to Apple Silicon GPU (MPS)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"🚀 Initializing SentenceTransformer on: {device.upper()}")
    
    model = SentenceTransformer("local_model", device=device)
    
    processed_count = 0
    start_time = time.time()

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip(): continue
            
            candidate = json.loads(line)
            candidate_id = candidate.get("candidate_id")
            chunks = []
            
            # --- 1. Identity Chunk ---
            profile = candidate.get("profile", {})
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            if headline or summary:
                chunks.append(f"Profile: {headline}. {summary}")
                
            # --- 2. Experience Chunks ---
            for job in candidate.get("career_history", []):
                title = job.get("title", "")
                company = job.get("company", "")
                desc = job.get("description", "")
                chunks.append(f"Experience: {title} at {company}. {desc}")
                
            # --- 3. Skills Chunk ---
            skills_list = [skill.get("name", "") for skill in candidate.get("skills", [])]
            if skills_list:
                chunks.append(f"Skills: {', '.join(skills_list)}")
                
            # --- Generate Embeddings ---
            if chunks:
                # model.encode returns a numpy array, we convert to list for JSON serialization
                embeddings = model.encode(chunks, convert_to_numpy=True).tolist()
                
                # Create the payload linking the chunks to their vectors
                payload = {
                    "candidate_id": candidate_id,
                    "vectors": []
                }
                
                for text_chunk, vector in zip(chunks, embeddings):
                    payload["vectors"].append({
                        "text": text_chunk,
                        "embedding": vector
                    })
                    
                outfile.write(json.dumps(payload) + '\n')
            
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"📦 Embedded {processed_count:,} candidates...")

    elapsed = time.time() - start_time
    print(f"✅ Complete! Embedded {processed_count:,} candidates in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    process_candidate_embeddings("candidates.jsonl", "candidate_embeddings.jsonl")