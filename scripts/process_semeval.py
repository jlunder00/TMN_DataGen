# scripts/process_semeval.py
import json
import pandas as pd
from pathlib import Path
import uuid

def load_semeval_file(filepath):
    """Load SemEval STS TSV file"""
    scores = []
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
                
            try:
                score = float(parts[0])
                sentence1 = parts[1]
                sentence2 = parts[2]
                scores.append(score)
                sentences.append((sentence1, sentence2))
            except (ValueError, IndexError) as e:
                print(f"Skipping malformed line: {line}\nError: {e}")
                continue
                
    return scores, sentences

def convert_to_jsonl(input_file, output_file):
    """Convert SemEval TSV to JSONL format compatible with TMN_DataGen"""
    
    scores, sentence_pairs = load_semeval_file(input_file)
    
    records = []
    for score, (sent1, sent2) in zip(scores, sentence_pairs):
        record = {
            'captionID': str(uuid.uuid4()),  # Generate unique ID
            'pairID': str(uuid.uuid4()),     # Generate unique ID 
            'sentence1': sent1,
            'sentence2': sent2,
            'similarity_score': float(score),
            'gold_label': float(score)  # Mark as similarity task
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
