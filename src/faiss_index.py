import json
import numpy as np
import faiss
import time
import pickle

def build_faiss_index(embeddings_file: str, index_output: str, meta_output: str):
    print("📥 Loading 4.45GB of embeddings into memory... this will take 1-2 minutes.")
    start_time = time.time()
    
    dimension = 384 
    # SWITCH TO INNER PRODUCT FOR COSINE SIMILARITY
    index = faiss.IndexFlatIP(dimension)
    
    metadata = []
    vector_list = []
    
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            cand_id = data.get("candidate_id")
            
            for vector_data in data.get("vectors", []):
                vector_list.append(vector_data["embedding"])
                
                metadata.append({
                    "candidate_id": cand_id,
                    "text": vector_data["text"]
                })
                
    print("⚙️ Normalizing and compressing vectors into FAISS binary index...")
    vector_matrix = np.array(vector_list).astype('float32')
    
    # NORMALIZE VECTORS TO FORCE COSINE SIMILARITY
    faiss.normalize_L2(vector_matrix)
    index.add(vector_matrix)
    
    print(f"💾 Saving FAISS index to {index_output}...")
    faiss.write_index(index, index_output)
    
    print(f"💾 Saving Metadata to {meta_output}...")
    with open(meta_output, 'wb') as f:
        pickle.dump(metadata, f)
        
    elapsed = time.time() - start_time
    print(f"✅ Success! Saved {index.ntotal:,} normalized vectors to disk in {elapsed:.2f}s!")

if __name__ == "__main__":
    build_faiss_index("candidate_embeddings.jsonl", "candidates.index", "faiss_metadata.pkl")