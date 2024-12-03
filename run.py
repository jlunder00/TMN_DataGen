# run.py
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from tqdm import tqdm
from TMN_DataGen import DatasetGenerator
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
from functools import partial
from dotenv import load_dotenv

load_dotenv()

class BatchProcessor:
    def __init__(self, 
                 input_file: str,
                 output_dir: str,
                 max_lines: Optional[int] = None,
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
        self.max_lines = max_lines
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
        self.logger.info(f"Output directory: {self.output_dir}")
        if self.num_partitions:
            self.logger.info(f"Splitting into {self.num_partitions} partitions")
            
        # Log resume information if applicable
        if self.partition_info['completed_partitions']:
            self.logger.info("\nResuming from previous state:")
            self.logger.info(f"Completed partitions: {self.partition_info['completed_partitions']}")
            self.logger.info(f"Current partition: {self.partition_info['current_partition']}")
            
    def _load_progress(self) -> set:
        """Load set of already processed sentence pair IDs"""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return set(json.load(f))
        return set()

    def _prepare_batch_data(self, item: Dict) -> Optional[Tuple[Tuple[str, str], str, str]]:
        """Process single item for batch preparation"""
        if item['pairID'] not in self.processed_pairs:
            return (
                (item['sentence1'], item['sentence2']),
                item['gold_label'],
                item['pairID']
            )
        return None
    
    def _save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(list(self.processed_pairs), f)
            
    def _load_last_batch(self) -> int:
        """Load the index of the last successfully processed batch"""
        if self.last_batch_file.exists():
            with open(self.last_batch_file) as f:
                return int(f.read().strip())
        return -1

    def _save_last_batch(self, batch_idx: int):
        """Save the index of the last successfully processed batch"""
        with open(self.last_batch_file, 'w') as f:
            f.write(str(batch_idx))

    def _load_partition_info(self) -> Dict:
        """Load information about completed partitions and batch tracking"""
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
        """Save partition tracking information"""
        with open(self.partition_info_file, 'w') as f:
            json.dump(self.partition_info, f)
            
    def _load_snli_data(self) -> List[Dict]:
        """Load SNLI data"""
        self.logger.info(f"Loading data from {self.input_file}")
        data = []
        with open(self.input_file) as f:
            next(f)
            for i, line in enumerate(f):
                if self.max_lines and i >= self.max_lines:
                    break
                data.append(json.loads(line))
            # data = [json.loads(line) for line in f]
        self.logger.info(f"Loaded {len(data)} sentence pairs")
        return data

    def _calculate_partition_sizes(self, total_batches: int) -> List[int]:
        """Calculate number of batches per partition"""
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

    def process_batch(self, batch_data: List[Dict], batch_idx: int):
        """Process one batch of sentence pairs"""
        if batch_idx <= self.last_batch_idx:
            return

        # sentence_pairs = []
        # labels = []
        # pair_ids = []
        # 
        # for item in batch_data:
        #     if item['pairID'] not in self.processed_pairs:
        #         sentence_pairs.append(
        #             (item['sentence1'], item['sentence2'])
        #         )
        #         labels.append(item['gold_label'])
        #         pair_ids.append(item['pairID'])
        # 
        # if not sentence_pairs:
        #     return

        with Pool(processes=self.num_workers) as pool:
            results = list(filter(None, pool.map(self._prepare_batch_data, batch_data)))

        if not results:
            return

        sentence_pairs, labels, pair_ids = zip(*results) 

        output_path = self.output_dir / f'batch_{batch_idx}.json'
        self.generator.generate_dataset(
            sentence_pairs=sentence_pairs,
            labels=labels,
            output_path=str(output_path),
            verbosity=self.verbosity,
            parser_config=self.parser_config,
            preprocessing_config=self.preprocessing_config,
            feature_config=self.feature_config
        )
        
        self.processed_pairs.update(pair_ids)
        self._save_progress()
        self._save_last_batch(batch_idx)

    def _read_batch_file(self, batch_file: Path) -> Dict:
        """Read single batch file"""
        with open(batch_file) as f:
            return json.load(f)

    def merge_partition(self, start_batch: int, end_batch: int, partition_num: int):
        """Merge a range of batches into a partition file"""
        self.logger.info(f"\nMerging partition {partition_num}")
        self.logger.info(f"Processing batches {start_batch} to {end_batch-1}")

        batch_files = [
            self.output_dir / f'batch_{idx}.json'
            for idx in range(start_batch, end_batch)
        ]
        
        # Parallel batch file reading
        with Pool(processes=self.num_workers) as pool:
            batch_data = pool.map(self._read_batch_file, batch_files)

        all_graph_pairs = []
        all_labels = []

        for data in tqdm(batch_data, desc=f"Merging batches for partition {partition_num}"):
            all_graph_pairs.extend(data['graph_pairs'])
            all_labels.extend(data['labels'])
        
        # for batch_idx in tqdm(range(start_batch, end_batch), desc=f"Merging batches for partition {partition_num}"):
        #     batch_file = self.output_dir / f'batch_{batch_idx}.json'
        #     if not batch_file.exists():
        #         raise ValueError(f"Missing batch file: {batch_file}")
        #         
        #     with open(batch_file) as f:
        #         data = json.load(f)
        #         all_graph_pairs.extend(data['graph_pairs'])
        #         all_labels.extend(data['labels'])
                
        # Save partition
        partition_data = {
            'graph_pairs': all_graph_pairs,
            'labels': all_labels
        }
        
        partition_file = self.output_dir / f'part_{partition_num}.json'
        self.logger.info(f"Saving partition file: {partition_file}")
        with open(partition_file, 'w') as f:
            json.dump(partition_data, f)
            
        # Clean up batch files in parallel
        self.logger.info("Cleaning up batch files...")
        with Pool(processes=self.num_workers) as pool:
            pool.map(Path.unlink, batch_files)
        # for batch_idx in range(start_batch, end_batch):
        #     batch_file = self.output_dir / f'batch_{batch_idx}.json'
        #     batch_file.unlink()
            
        self.logger.info(f"Partition {partition_num} complete with {len(all_graph_pairs)} pairs")

    def process_all(self):
        """Process entire dataset with batching"""
        self.logger.info("\nStarting dataset processing")
        start_time = datetime.now()
        
        data = self._load_snli_data()
        total_batches = (len(data) + self.batch_size - 1) // self.batch_size
        partition_sizes = self._calculate_partition_sizes(total_batches)
        
        # Calculate correct starting point
        current_partition = self.partition_info['current_partition']
        completed_parts = set(self.partition_info['completed_partitions'])
        
        # Calculate actual batch start based on completed partitions
        batch_start = sum(partition_sizes[:current_partition])
        
        # Update batch offset tracking
        self.partition_info['batch_offsets'] = {
            str(p): sum(partition_sizes[:p]) 
            for p in range(len(partition_sizes))
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
                self.logger.info(f"Expected pairs: {partition_size * self.batch_size}")
                
                # Process batches for this partition
                batch_progress = tqdm(range(batch_start, batch_end), 
                                    desc=f"Processing partition {partition_idx}")
                for batch_idx in batch_progress:
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, len(data))
                    batch = data[start_idx:end_idx]
                    
                    try:
                        self.process_batch(batch, batch_idx)
                    except Exception as e:
                        self.logger.error(f"\nError processing batch {batch_idx}: {e}")
                        raise
                
                # Merge partition
                self.merge_partition(batch_start, batch_end, partition_idx)
                
                # Update partition tracking
                self.partition_info['completed_partitions'].append(partition_idx)
                self.partition_info['current_partition'] = partition_idx + 1
                self._save_partition_info()
                
                batch_start = batch_end

        except Exception as e:
            self.logger.error(f"\nProcessing stopped: {e}")
            self._save_progress()
            self._save_partition_info()
            raise

        # Clean up tracking files when completely done
        if current_partition >= len(partition_sizes) - 1:
            self.logger.info("\nCleaning up tracking files...")
            for file in [self.progress_file, self.last_batch_file, self.partition_info_file]:
                if file.exists():
                    file.unlink()
                    
        end_time = datetime.now()
        duration = end_time - start_time
        self.logger.info(f"\nProcessing complete in {duration}")

def merge_partition_files(file_paths: List[str], output_path: str):
    """Merge multiple partition files into one"""
    logging.info(f"\nMerging {len(file_paths)} partition files")
    logging.info(f"Output file: {output_path}")
    
    all_graph_pairs = []
    all_labels = []
    
    for file_path in tqdm(file_paths, desc="Merging partitions"):
        with open(file_path) as f:
            data = json.load(f)
            all_graph_pairs.extend(data['graph_pairs'])
            all_labels.extend(data['labels'])
            
    merged_data = {
        'graph_pairs': all_graph_pairs,
        'labels': all_labels
    }
    
    logging.info(f"Saving merged file with {len(all_graph_pairs)} pairs")
    with open(output_path, 'w') as f:
        json.dump(merged_data, f)
    logging.info("Merge complete")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process text pairs into graph matching network format")
    
    # Add subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operating mode')
    
    # Parser for processing mode
    process_parser = subparsers.add_parser('process', help='Process SNLI data into graph format')
    process_parser.add_argument("-if", "--input_file", 
                              type=str, 
                              required=True,
                              help="Input data path")
    process_parser.add_argument("-od", "--out_dir", 
                              type=str, 
                              required=True,
                              help="Output directory path")
    process_parser.add_argument("-ml", "--max_lines", type=int, help="Maximum number of lines to process")
    process_parser.add_argument("-sm", "--spacy_model",
                              type=str,
                              default="en_core_web_sm",
                              help="Spacy model to use (e.g., en_core_web_trf)")
    process_parser.add_argument("-v", "--verbosity",
                              type=str,
                              choices=['quiet', 'normal', 'debug'],
                              default='quiet',
                              help="Verbosity level")
    process_parser.add_argument("-n", "--num_partitions",
                              type=int,
                              help="Split output into this many partitions")
    process_parser.add_argument("-bs", "--batch_size",
                              type=int,
                              default=100,
                              help="Number of pairs per batch")
    process_parser.add_argument("-w", "--workers",
                          type=int,
                          default=None,
                          help="Number of worker processes for parallel operations")

    
    # Parser for merge mode
    merge_parser = subparsers.add_parser('merge', help='Merge partition files')
    merge_parser.add_argument("-if", "--input_files",
                            type=str,
                            nargs='+',
                            required=True,
                            help="Input partition files to merge")
    merge_parser.add_argument("-o", "--output",
                            type=str,
                            required=True,
                            help="Output merged file path")
    
    args = parser.parse_args()

    if args.mode == 'process':
        parser_config = {
            'parser': {
                'type': 'multi',
                'parsers': {
                    'diaparser': {
                        'enabled': True,
                        'model_name': 'en_ewt.electra-base'
                    },
                    'spacy': {
                        'enabled': True,
                        'model_name': args.spacy_model
                    }
                }
            }
        }
        
        processor = BatchProcessor(
            input_file=args.input_file,
            output_dir=args.out_dir,
            max_lines=args.max_lines,
            batch_size=args.batch_size,
            checkpoint_every=1000,
            verbosity=args.verbosity,
            parser_config=parser_config,
            num_partitions=args.num_partitions,
            num_workers=args.workers
        )
        processor.process_all()
        
    elif args.mode == 'merge':
        merge_partition_files(args.input_files, args.output)


