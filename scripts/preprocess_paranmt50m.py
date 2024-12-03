import json
import argparse
import uuid
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def parse_chunk_file(filepath):
    """Parse a single chunk file"""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            try:
                sentence1 = parts[0]
                sentence2 = parts[1]
                similarity_score = float(parts[2])
                entries.append((sentence1, sentence2, similarity_score))
            except ValueError as e:
                print(f"Skipping malformed line: {line}\nError: {e}")
                continue
    return entries

def process_chunk_to_jsonl(chunk_path):
    """Convert a chunk file to a list of JSONL-formatted records"""
    entries = parse_chunk_file(chunk_path)
    records = []
    for sentence1, sentence2, similarity_score in entries:
        record = {
            'captionID': str(uuid.uuid4()),
            'pairID': str(uuid.uuid4()),
            'sentence1': sentence1,
            'sentence2': sentence2,
            'similarity_score': similarity_score,
            'gold_label': similarity_score
        }
        records.append(json.dumps(record))
    return records

def process_all_chunks_to_jsonl(chunk_dir, output_file):
    """Process all chunk files in parallel and save the results to JSONL"""
    chunk_files = list(Path(chunk_dir).glob("chunk_*.txt"))
    if not chunk_files:
        raise ValueError("No chunk files found in the specified directory.")
    print(f"Found {len(chunk_files)} chunk files in {chunk_dir}")

    # Use multiprocessing pool to process chunk files
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_chunk_to_jsonl, chunk_files), total=len(chunk_files), desc="Processing chunks"))

    # Write results to the output JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk_records in results:
            for record in chunk_records:
                f.write(record + '\n')

def main():
    parser = argparse.ArgumentParser(description='Process chunked ParaNMT files to JSONL format with multiprocessing')
    parser.add_argument('chunk_dir', type=str, help='Directory containing chunked text files')
    parser.add_argument('output_file', type=str, help='Output JSONL file')
    args = parser.parse_args()

    chunk_dir = Path(args.chunk_dir)
    output_file = Path(args.output_file)

    print(f"Processing chunked files in directory: {chunk_dir}")
    process_all_chunks_to_jsonl(chunk_dir, output_file)
    print(f"JSONL file written to: {output_file}")

if __name__ == '__main__':
    main()

