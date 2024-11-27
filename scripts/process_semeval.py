# scripts/process_semeval.py
import json
import pandas as pd
from pathlib import Path

def load_semeval_file(filepath):
    """Load SemEval STS TSV file"""
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            line = line.strip()
            if not line:
                continue
                
            # Split on tabs
            parts = line.split('\t')
            # Remove empty strings that come from consecutive tabs
            parts = [p for p in parts if p]
            
            if len(parts) < 3:
                print(f"Skipping malformed line (not enough parts): {line}")
                continue
            
            try:
                # First non-empty part should be score
                score = float(parts[0])
                # Second part is sentence1, third part is sentence2
                sentence1 = parts[1]
                sentence2 = parts[2]
                
                pairs.append({
                    'score': score,
                    'sentence1': sentence1.strip(),
                    'sentence2': sentence2.strip()
                })
            except (ValueError, IndexError) as e:
                print(f"Skipping invalid line: {line}\nError: {e}")
                continue
                
    return pd.DataFrame(pairs)

def convert_to_jsonl(input_file, output_file):
    """Convert SemEval TSV to JSONL format"""
    df = load_semeval_file(input_file)
    
    # Normalize scores to [-1, 1] range
    # SemEval scores are 0-5, so we map:
    # 0 -> -1
    # 2.5 -> 0 
    # 5 -> 1
    records = []
    for _, row in df.iterrows():
        record = {
            'sentence1': row['sentence1'],
            'sentence2': row['sentence2'],
            'similarity_score': row['score'],  # Center and normalize
            'gold_label': 'similarity'  # Mark as similarity task
        }
        records.append(record)
        
    # Write as JSONL
    with open(output_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

def main():
    # Process main splits
    base_dir = Path('/home/jlunder/research/dataset-sts/data/sts/semeval-sts/all')
    
    for split in ['train', 'val', 'test']:
        input_file = base_dir / f'2015.{split}.tsv'
        output_file = base_dir / f'2015.{split}.jsonl'
        convert_to_jsonl(input_file, output_file)

if __name__ == '__main__':
    main()
