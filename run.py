# run.py
import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from TMN_DataGen import DatasetGenerator

class BatchProcessor:
    def __init__(self, 
                 input_file: str,
                 output_dir: str,
                 batch_size: int = 1000,
                 checkpoint_every: int = 5000):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset generator
        self.generator = DatasetGenerator()
        
        # Track progress
        self.progress_file = self.output_dir / 'progress.json'
        self.processed_pairs = self._load_progress()
        
    def _load_progress(self) -> set:
        """Load set of already processed sentence pair IDs"""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return set(json.load(f))
        return set()
    
    def _save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.processed_pairs), f)
            
    def _load_snli_data(self) -> Dict:
        """Load SNLI data"""
        with open(self.input_file) as f:
            data = [json.loads(line) for line in f]
        return data
    
    def process_batch(self, batch_data: List[Dict], batch_idx: int):
        """Process one batch of sentence pairs"""
        sentence_pairs = []
        labels = []
        pair_ids = []
        
        for item in batch_data:
            if item['pairID'] not in self.processed_pairs:
                sentence_pairs.append(
                    (item['sentence1'], item['sentence2'])
                )
                labels.append(item['gold_label'])
                pair_ids.append(item['pairID'])
        
        if not sentence_pairs:
            return
            
        # Generate trees and save batch
        output_path = self.output_dir / f'batch_{batch_idx}.json'
        self.generator.generate_dataset(
            sentence_pairs=sentence_pairs,
            labels=labels,
            output_path=str(output_path)
        )
        
        # Update progress
        self.processed_pairs.update(pair_ids)
        if len(self.processed_pairs) % self.checkpoint_every == 0:
            self._save_progress()
            
    def merge_batches(self):
        """Merge all batch files into final dataset"""
        all_graph_pairs = []
        all_labels = []
        
        batch_files = sorted(self.output_dir.glob('batch_*.json'))
        for batch_file in tqdm(batch_files, desc="Merging batches"):
            with open(batch_file) as f:
                data = json.load(f)
                all_graph_pairs.extend(data['graph_pairs'])
                all_labels.extend(data['labels'])
                
        # Save merged dataset
        final_dataset = {
            'graph_pairs': all_graph_pairs,
            'labels': all_labels
        }
        
        with open(self.output_dir / 'final_dataset.json', 'w') as f:
            json.dump(final_dataset, f)
            
        # Cleanup batch files
        for batch_file in batch_files:
            batch_file.unlink()
            
    def process_all(self):
        """Process entire dataset with batching"""
        data = self._load_snli_data()
        
        for i in tqdm(range(0, len(data), self.batch_size)):
            batch = data[i:i + self.batch_size]
            self.process_batch(batch, i // self.batch_size)
            
        self._save_progress()
        self.merge_batches()

if __name__ == '__main__':
    processor = BatchProcessor(
        input_file='../data/snli_1.0_train.jsonl',
        output_dir='../data/processed_data',
        batch_size=100,
        checkpoint_every=1000
    )
    processor.process_all()
