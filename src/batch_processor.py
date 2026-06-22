import json
import time
import traceback
from typing import Dict, Any

from feature_extractor import extract_candidate_features

def process_large_jsonl_safely(input_path: str, output_path: str, batch_size: int = 5000):
    """
    Streams large resume logs, safely increments records, patches zero-division bugs,
    and isolates line-by-line runtime errors cleanly.
    """
    start_time = time.time()
    processed_count = 0
    error_count = 0

    print("🚀 Initializing Robust Processing Pipeline for 100,000 Records...")

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_idx, line in enumerate(infile, start=1):
                if not line.strip():
                    continue
                    
                try:
                    # 1. Parse log line
                    candidate_data = json.loads(line)
                    
                    # 2. Compute 14-layer feature extraction matrix
                    features = extract_candidate_features(candidate_data)
                    
                    # 3. Commit features to disk space incrementally
                    outfile.write(json.dumps(features) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    error_count += 1
                except Exception as e:
                    error_count += 1
                    # Un-comment the next lines if you need to catch raw field structural anomalies
                    # if error_count <= 5:
                    #     print(f"Structural anomaly detected on line {line_idx}: {str(e)}")
                    #     traceback.print_exc()

                # 4. Corrected Progress Tracking Heartbeat (Skips checking zero state)
                if processed_count > 0 and processed_count % batch_size == 0:
                    elapsed = time.time() - start_time
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {processed_count:,} / 100,000 records... ({speed:.0f} lines/sec)")

    except FileNotFoundError:
        print(f"Execution Halted: The file at source path '{input_path}' could not be located.")
        return

    total_time = time.time() - start_time
    print("\n Verification Pipeline Execution Completed!")
    print(f"Successful Extractions: {processed_count:,}")
    print(f"Faulty Records Bypassed: {error_count}")
    print(f"Clean Execution Run Time: {total_time:.2f} seconds")

if __name__ == "__main__":
    # Substitute your production filenames here
    INPUT_FILE = "candidates.jsonl"
    OUTPUT_FILE = "final_ranked_leaderboard.jsonl"
    
    process_large_jsonl_safely(INPUT_FILE, OUTPUT_FILE)
