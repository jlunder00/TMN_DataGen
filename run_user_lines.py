import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm
from TMN_DataGen import DatasetGenerator
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
import uuid
from functools import partial
from dotenv import load_dotenv

load_dotenv()


class BatchProcessor:
    def __init__(self, 
                 input_file: str,
                 output_dir: str,
                 num_lines: int,
                 batch_size: int = 1000,
                 checkpoint_every: int = 5000,
                 verbosity: str = 'quiet',
                 parser_config: Optional[Dict] = None,
                 preprocessing_config: Optional[Dict] = None,
                 feature_config: Optional[Dict] = None,
                 num_partitions: Optional[int] = None,
                 num_workers: Optional[int] = None):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.num_lines = num_lines
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.verbosity = verbosity
        self.parser_config = parser_config
        self.preprocessing_config = preprocessing_config
        self.feature_config = feature_config
        self.num_partitions = num_partitions
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers or max(1, cpu_count()-4)
        
        # Set up logging
        log_file = self.output_dir / 'processing.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using {self.num_workers} worker processes for parallel operations")
        
        # Load dataset generator
        self.generator = DatasetGenerator()
        
        # Track progress
        self.progress_file = self.output_dir / 'progress.json'
        self.last_batch_file = self.output_dir / 'last_batch.txt'
        self.partition_info_file = self.output_dir / 'partition_info.json'
        
        self.processed_pairs = self._load_progress()
        self.last_batch_idx = self._load_last_batch()
        self.partition_info = self._load_partition_info()
        
        self.logger.info(f"\nInitialized BatchProcessor:")
        self.logger.info(f"Input file: {self.input_file}")
        self.logger.info(f"Number of lines: {self.num_lines}")
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.num_partitions:
            self.logger.info(f"Splitting into {self.num_partitions} partitions")
            
        # Log resume information if applicable
        if self.partition_info['completed_partitions']:
            self.logger.info("\nResuming from previous state:")
            self.logger.info(f"Completed partitions: {self.partition_info['completed_partitions']}")
            self.logger.info(f"Current partition: {self.partition_info['current_partition']}")

    def _load_progress(self) -> set:
        """Load set of already processed sentence pair IDs."""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return set(json.load(f))
        return set()

    def _save_progress(self):
        """Save current progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.processed_pairs), f)

    def _load_last_batch(self) -> int:
        """Load the index of the last successfully processed batch."""
        if self.last_batch_file.exists():
            with open(self.last_batch_file) as f:
                return int(f.read().strip())
        return -1

    def _save_last_batch(self, batch_idx: int):
        """Save the index of the last successfully processed batch."""
        with open(self.last_batch_file, 'w') as f:
            f.write(str(batch_idx))

    def _load_partition_info(self) -> Dict:
        """Load information about completed partitions and batch tracking."""
        if self.partition_info_file.exists():
            with open(self.partition_info_file) as f:
                info = json.load(f)
                if 'batch_offsets' not in info:
                    info['batch_offsets'] = {}
                return info
        return {
            'completed_partitions': [], 
            'current_partition': 0,
            'batch_offsets': {}
        }

    def _save_partition_info(self):
        """Save partition tracking information."""
        with open(self.partition_info_file, 'w') as f:
            json.dump(self.partition_info, f)

    def _calculate_partition_sizes(self, total_batches: int) -> List[int]:
        """Calculate number of batches per partition."""
        if not self.num_partitions:
            return [total_batches]
            
        base_size = total_batches // self.num_partitions
        extra = total_batches % self.num_partitions
        sizes = [base_size for _ in range(self.num_partitions)]
        
        # Distribute extra batches
        for i in range(extra):
            sizes[i] += 1
            
        self.logger.info("\nPartition sizes (in batches):")
        for i, size in enumerate(sizes):
            self.logger.info(f"Partition {i}: {size} batches")
            
        return sizes

    def _stream_data(self, start_idx: int, end_idx: int) -> List[str]:
        """Stream lines from the file for the specified range."""
        lines = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for current_idx, line in enumerate(f):
                if current_idx < start_idx:
                    continue
                if current_idx >= end_idx:
                    break
                lines.append(line.strip())
        return lines

    def _process_line(self, line: str) -> Optional[Dict]:
        """Process a single line."""
        parts = line.split('\t')
        if len(parts) != 3:
            return None
        try:
            return {
                'captionID': str(uuid.uuid4()),
                'pairID': str(uuid.uuid4()),
                'sentence1': parts[0].strip(),
                'sentence2': parts[1].strip(),
                'gold_label': float(parts[2].strip())
            }
        except ValueError:
            return None

    def process_batch(self, batch_data: List[str], batch_idx: int):
        """Process one batch of sentence pairs."""
        if batch_idx <= self.last_batch_idx:
            return

        with Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_line, batch_data)

        # Filter out None results (invalid lines)
        results = [result for result in results if result]

        if not results:
            return

        # Generate dataset file for this batch
        output_path = self.output_dir / f'batch_{batch_idx}.json'
        self.generator.generate_dataset(
            sentence_pairs=[(r['sentence1'], r['sentence2']) for r in results],
            labels=[r['gold_label'] for r in results],
            output_path=str(output_path),
            verbosity=self.verbosity,
            parser_config=self.parser_config,
            preprocessing_config=self.preprocessing_config,
            feature_config=self.feature_config
        )
        
        # Update progress tracking
        self.processed_pairs.update(r['pairID'] for r in results)
        self._save_progress()
        self._save_last_batch(batch_idx)

    def merge_partition(self, start_batch: int, end_batch: int, partition_num: int):
        """Merge a range of batches into a partition file."""
        self.logger.info(f"\nMerging partition {partition_num}")
        self.logger.info(f"Processing batches {start_batch} to {end_batch-1}")

        batch_files = [
            self.output_dir / f'batch_{idx}.json'
            for idx in range(start_batch, end_batch)
        ]

        # Load batch data in parallel
        with Pool(processes=self.num_workers) as pool:
            batch_data = pool.map(self._read_batch_file, batch_files)

        all_graph_pairs = []
        all_labels = []

        for data in tqdm(batch_data, desc=f"Merging partition {partition_num}"):
            all_graph_pairs.extend(data['graph_pairs'])
            all_labels.extend(data['labels'])

        # Save partition
        partition_data = {
            'graph_pairs': all_graph_pairs,
            'labels': all_labels
        }
        partition_file = self.output_dir / f'part_{partition_num}.json'
        with open(partition_file, 'w') as f:
            json.dump(partition_data, f)

    def process_all(self):
        """Process entire dataset with batching."""
        self.logger.info("\nStarting dataset processing")
        start_time = datetime.now()

        total_batches = (self.num_lines + self.batch_size - 1) // self.batch_size
        partition_sizes = self._calculate_partition_sizes(total_batches)
        
        # Determine starting partition and batch
        current_partition = self.partition_info['current_partition']
        completed_parts = set(self.partition_info['completed_partitions'])
        batch_start = sum(partition_sizes[:current_partition])

        self.partition_info['batch_offsets'] = {
            str(p): sum(partition_sizes[:p]) for p in range(len(partition_sizes))
        }
        self._save_partition_info()

        try:
            for partition_idx in range(current_partition, len(partition_sizes)):
                if partition_idx in completed_parts:
                    self.logger.info(f"\nSkipping completed partition {partition_idx}")
                    continue
                
                partition_size = partition_sizes[partition_idx]
                batch_end = batch_start + partition_size
                
                self.logger.info(f"\nProcessing partition {partition_idx + 1}/{len(partition_sizes)}")
                self.logger.info(f"Batch range: {batch_start} to {batch_end - 1}")

                for batch_idx in tqdm(range(batch_start, batch_end), desc=f"Partition {partition_idx}"):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, self.num_lines)
                    batch_data = self._stream_data(start_idx, end_idx)
                    
                    try:
                        self.process_batch(batch_data, batch_idx)
                    except Exception as e:
                        self.logger.error(f"Error processing batch {batch_idx}: {e}")
                        raise
                
                self.merge_partition(batch_start, batch_end, partition_idx)
                self.partition_info['completed_partitions'].append(partition_idx)
                self.partition_info['current_partition'] = partition_idx + 1
                self._save_partition_info()
                
                batch_start = batch_end

        except Exception as e:
            self.logger.error(f"\nProcessing stopped: {e}")
            self._save_progress()
            self._save_partition_info()
            raise

        self.logger.info("\nProcessing complete")
        end_time = datetime.now()
        self.logger.info(f"Processing completed in {end_time - start_time}")

    def _read_batch_file(self, batch_file: Path) -> Dict:
        """Read a single batch file."""
        with open(batch_file) as f:
            return json.load(f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Process text pairs into graph matching network format")

    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')

    # Processing mode
    process_parser = subparsers.add_parser('process', help='Process data into graph format')
    process_parser.add_argument("-if", "--input_file", type=str, required=True, help="Input data path")
    process_parser.add_argument("-od", "--out_dir", type=str, required=True, help="Output directory path")
    process_parser.add_argument("-nl", "--num_lines", type=int, required=True, help="Number of lines in the file")
    process_parser.add_argument("-n", "--num_partitions", type=int, help="Split output into this many partitions")
    process_parser.add_argument("-bs", "--batch_size", type=int, default=1000, help="Number of pairs per batch")
    process_parser.add_argument("-w", "--workers", type=int, default=None, help="Number of worker processes")

    args = parser.parse_args()

    if args.mode == 'process':
        processor = BatchProcessor(
            input_file=args.input_file,
            output_dir=args.out_dir,
            num_lines=args.num_lines,
            batch_size=args.batch_size,
            num_partitions=args.num_partitions,
            num_workers=args.workers
        )
        processor.process_all()

