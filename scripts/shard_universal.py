# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

import json
import os
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any, Union
from dataclasses import dataclass, field
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass 
class UniversalShardConfig:
    """Configuration for universal file sharding and splitting"""
    input_directory: str
    output_directory: str
    target_dataset_name: str
    mode: str  # 'text' or 'json'
    lines_per_shard: Optional[int] = None  # For text mode
    groups_per_shard: Optional[int] = None  # For JSON mode
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = None
    random_seed: int = 42
    file_pattern: str = "part-*"
    
    # JSON-specific options
    spacy_variant: str = "trf"
    remove_fields: List[str] = field(default_factory=list)
    rename_map: Dict[str, str] = field(default_factory=dict)
    min_trees_per_group: int = 2
    nested: bool = False
    dataset_type: str = ""
    
    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = max(1, mp.cpu_count() - 1)
            
        # Validate mode and required parameters
        if self.mode not in ['text', 'json']:
            raise ValueError(f"Mode must be 'text' or 'json', got '{self.mode}'")
            
        if self.mode == 'text' and self.lines_per_shard is None:
            raise ValueError("lines_per_shard is required for text mode")
            
        if self.mode == 'json' and self.groups_per_shard is None:
            raise ValueError("groups_per_shard is required for json mode")
            
        # Validate ratios
        total_ratio = self.train_ratio + self.dev_ratio + self.test_ratio
        assert abs(total_ratio - 1.0) < 0.001, f"Split ratios must sum to 1.0, got {total_ratio}"
        
        # Ensure paths exist
        self.input_path = Path(self.input_directory)
        self.output_path = Path(self.output_directory)
        
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_directory}")

class ProcessorRegistry:
    """Registry for different processing modes using decorator pattern"""
    
    def __init__(self):
        self._processors = {}
        self._split_strategies = {}
    
    def processor(self, mode: str):
        """Decorator to register processing modes"""
        def decorator(cls):
            self._processors[mode] = cls
            return cls
        return decorator
    
    def split_strategy(self, strategy_name: str):
        """Decorator to register split strategies"""
        def decorator(func):
            self._split_strategies[strategy_name] = func
            return func
        return decorator
    
    def get_processor(self, mode: str, config: UniversalShardConfig):
        """Get processor instance for the specified mode"""
        if mode not in self._processors:
            raise ValueError(f"Unknown processing mode: {mode}. Available: {list(self._processors.keys())}")
        return self._processors[mode](config)
    
    def get_split_strategy(self, strategy_name: str):
        """Get split strategy function"""
        if strategy_name not in self._split_strategies:
            raise ValueError(f"Unknown split strategy: {strategy_name}. Available: {list(self._split_strategies.keys())}")
        return self._split_strategies[strategy_name]

# Global registry instance
registry = ProcessorRegistry()

def _process_file_worker(file_info: Tuple[Path, str], config: UniversalShardConfig) -> Dict[str, int]:
    """Worker function for multiprocessing - needs to be at module level for pickling"""
    file_path, category_name = file_info
    processor = registry.get_processor(config.mode, config)
    return processor.process_file(file_path, category_name)

@registry.split_strategy('random')
def random_split_strategy(items: List[Any], config: UniversalShardConfig) -> Dict[str, List[Any]]:
    """Random split strategy for any list of items"""
    shuffled_items = items.copy()
    random.shuffle(shuffled_items)
    
    total_items = len(shuffled_items)
    train_size = int(total_items * config.train_ratio)
    dev_size = int(total_items * config.dev_ratio)
    
    splits = {}
    if train_size > 0:
        splits['train'] = shuffled_items[:train_size]
    if dev_size > 0:
        splits['dev'] = shuffled_items[train_size:train_size + dev_size]
    if total_items - train_size - dev_size > 0:
        splits['test'] = shuffled_items[train_size + dev_size:]
        
    return splits

class BaseProcessor:
    """Base class for file processors"""
    
    def __init__(self, config: UniversalShardConfig):
        self.config = config
    
    def split_data(self, data: List[Any], strategy: str = 'random') -> Dict[str, List[Any]]:
        """Split data according to specified strategy"""
        split_func = registry.get_split_strategy(strategy)
        return split_func(data, self.config)
    
    def get_output_directory_name(self, split_name: str, category_name: str = None) -> str:
        """Generate output directory name based on config"""
        if self.config.mode == 'json':
            if self.config.nested and category_name:
                base_name = f"{self.config.target_dataset_name}_{self.config.dataset_type}_converted_{self.config.spacy_variant}_sharded"
            else:
                base_name = f"{self.config.target_dataset_name}_converted_{self.config.spacy_variant}_sharded"
        else:  # text mode
            base_name = f"{self.config.target_dataset_name}_{split_name}_converted_{self.config.spacy_variant}_sharded"
            
        return base_name
    
    def create_output_path(self, split_name: str, category_name: str = None) -> Path:
        """Create full output path for a split"""
        dir_name = self.get_output_directory_name(split_name, category_name)
        
        if self.config.nested and category_name:
            output_path = self.config.output_path / split_name / dir_name / category_name
        else:
            output_path = self.config.output_path / split_name / dir_name
            
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def process_file(self, file_path: Path, category_name: str = None) -> Dict[str, int]:
        """Process a single file - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_file method")

@registry.processor('text')
class TextProcessor(BaseProcessor):
    """Processor for text files"""
    
    def create_shards(self, lines: List[str], split_name: str, part_number: int) -> List[Tuple[str, List[str]]]:
        """Create shards from lines with proper naming"""
        shards = []
        for shard_idx, start_idx in enumerate(range(0, len(lines), self.config.lines_per_shard)):
            end_idx = min(start_idx + self.config.lines_per_shard, len(lines))
            shard_lines = lines[start_idx:end_idx]
            
            shard_filename = f"part_{part_number}_shard_{shard_idx:06d}.txt"
            shards.append((shard_filename, shard_lines))
        
        return shards
    
    def write_shard_file(self, filepath: Path, lines: List[str]) -> int:
        """Write lines to shard file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(line if line.endswith('\n') else line + '\n' for line in lines)
            return len(lines)
        except Exception as e:
            logger.error(f"Error writing shard file {filepath}: {e}")
            return 0
    
    def process_file(self, file_path: Path, category_name: str = None) -> Dict[str, int]:
        """Process a single text file"""
        try:
            # Extract part number from filename
            part_number = int(file_path.name.split('-')[-1])
            
            # Read all lines from file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n\r') for line in f.readlines() if line.strip()]
            
            if not lines:
                logger.warning(f"No valid lines found in {file_path}")
                return {'train': 0, 'dev': 0, 'test': 0}
            
            # Split lines into train/dev/test
            split_lines = self.split_data(lines)
            
            results = {}
            for split_name, split_lines_data in split_lines.items():
                if not split_lines_data:
                    results[split_name] = 0
                    continue
                
                # Create output directory for this split
                split_output_dir = self.create_output_path(split_name, category_name)
                
                # Create and write shards
                shards = self.create_shards(split_lines_data, split_name, part_number)
                total_lines_written = 0
                
                for shard_filename, shard_lines in shards:
                    shard_path = split_output_dir / shard_filename
                    lines_written = self.write_shard_file(shard_path, shard_lines)
                    total_lines_written += lines_written
                
                results[split_name] = total_lines_written
                logger.debug(f"Wrote {len(shards)} shards for {split_name} split of {file_path.name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return {'train': 0, 'dev': 0, 'test': 0}

@registry.processor('json')
class JsonProcessor(BaseProcessor):
    """Processor for JSON files with groups"""
    
    def process_group(self, group: dict) -> Optional[dict]:
        """Process a single group according to configuration"""
        # Skip groups with too few trees
        if len(group.get("trees", [])) + len(group.get("trees_b", [])) < self.config.min_trees_per_group:
            return None
            
        processed = {k: v for k, v in group.items() 
                    if k not in self.config.remove_fields}
        
        for old_name, new_name in self.config.rename_map.items():
            if old_name in processed:
                processed[new_name] = processed.pop(old_name)
                
        return processed
    
    def create_shards(self, groups: List[dict], metadata: dict, split_name: str, 
                     file_stem: str, source_name: str) -> List[Tuple[str, dict, dict]]:
        """Create shards from groups with proper naming"""
        shards = []
        for shard_idx, start_idx in enumerate(range(0, len(groups), self.config.groups_per_shard)):
            end_idx = min(start_idx + self.config.groups_per_shard, len(groups))
            shard_groups = groups[start_idx:end_idx]
            
            shard_data = {
                **metadata,
                'groups': shard_groups,
                'shard_info': {
                    'shard_index': shard_idx,
                    'groups_in_shard': len(shard_groups),
                    'source_name': source_name,
                    'source_file': f"{file_stem}.json"
                }
            }
            
            # Create counts data
            counts = {
                'n_groups': len(shard_groups),
                'trees_per_group': [len(group.get("trees", [])) for group in shard_groups]
            }
            if shard_groups and 'trees_b' in shard_groups[0]:
                counts['trees_b_per_group'] = [len(group.get('trees_b', [])) for group in shard_groups]
            
            shard_filename = f"{file_stem}_shard_{shard_idx:06d}.json"
            shards.append((shard_filename, shard_data, counts))
        
        return shards
    
    def write_shard_files(self, output_dir: Path, shard_filename: str, 
                         shard_data: dict, counts_data: dict) -> int:
        """Write shard and counts files"""
        try:
            # Write shard file
            shard_path = output_dir / shard_filename
            with open(shard_path, 'w') as f:
                json.dump(shard_data, f)
            
            # Write counts file
            counts_filename = shard_filename.replace('.json', '_counts.json')
            counts_path = output_dir / counts_filename
            with open(counts_path, 'w') as f:
                json.dump(counts_data, f)
            
            return len(shard_data.get('groups', []))
            
        except Exception as e:
            logger.error(f"Error writing shard files {shard_filename}: {e}")
            return 0
    
    def process_file(self, file_path: Path, category_name: str = None) -> Dict[str, int]:
        """Process a single JSON file"""
        try:
            # Load and process file
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract metadata
            metadata = {k: v for k, v in data.items() if k != 'groups'}
            
            # Process and filter groups
            valid_groups = []
            for group in data.get('groups', []):
                processed = self.process_group(group)
                if processed:
                    valid_groups.append(processed)
            
            if not valid_groups:
                logger.info(f"No valid groups found in {file_path.name}")
                return {'train': 0, 'dev': 0, 'test': 0}
                
            # Split groups
            split_groups = self.split_data(valid_groups)
            
            # Determine source name
            if self.config.nested:
                source_name = f"{self.config.target_dataset_name}_{self.config.dataset_type}_{category_name if category_name else ''}"
            else:
                source_name = f"{self.config.target_dataset_name}"
                
            file_stem = file_path.stem
            
            results = {}
            for split_name, groups in split_groups.items():
                if not groups:
                    results[split_name] = 0
                    continue
                    
                # Create output directory
                split_output_dir = self.create_output_path(split_name, category_name)
                
                # Create and write shards
                shards = self.create_shards(groups, metadata, split_name, file_stem, source_name)
                total_groups_written = 0
                
                for shard_filename, shard_data, counts_data in shards:
                    groups_written = self.write_shard_files(split_output_dir, shard_filename, 
                                                          shard_data, counts_data)
                    total_groups_written += groups_written
                
                results[split_name] = total_groups_written
                logger.debug(f"Wrote {len(shards)} shards for {split_name} split of {file_path.name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")
            return {'train': 0, 'dev': 0, 'test': 0}

class UniversalFileShardingManager:
    """Main manager for universal file sharding operations"""
    
    def __init__(self, config: UniversalShardConfig):
        self.config = config
        self.processor = registry.get_processor(config.mode, config)
    
    def setup_output_directories(self):
        """Create the output directory structure"""
        for split in ['train', 'dev', 'test']:
            # Create base split directory
            (self.config.output_path / split).mkdir(parents=True, exist_ok=True)
            
            # Create dataset-specific directory structure
            if self.config.nested:
                # For nested structure, directories will be created per category
                pass
            else:
                # For flat structure, create the target directory
                target_dir = self.config.output_path / split / self.processor.get_output_directory_name(split)
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {target_dir}")
    
    def find_input_files(self, base_path: Path = None) -> List[Tuple[Path, str]]:
        """Find all input files matching the pattern"""
        if base_path is None:
            base_path = self.config.input_path
            
        pattern = self.config.file_pattern
        input_files = list(base_path.glob(pattern))
        
        # Filter out files that don't match the expected naming pattern
        valid_files = []
        for file_path in input_files:
            if file_path.is_file():
                if self.config.mode == 'text' and file_path.name.startswith('part-'):
                    try:
                        # Verify we can extract a part number for text files
                        int(file_path.name.split('-')[-1])
                        valid_files.append((file_path, None))
                    except (ValueError, IndexError):
                        logger.warning(f"Skipping file with invalid naming: {file_path.name}")
                elif self.config.mode == 'json' and file_path.name.startswith('part_') and file_path.suffix == '.json':
                    # For JSON files, we don't need counts files
                    if not file_path.name.endswith('_counts.json'):
                        valid_files.append((file_path, None))
        
        return sorted(valid_files)
    
    def find_nested_files(self) -> List[Tuple[Path, str]]:
        """Find files in nested directory structure"""
        all_files = []
        
        # Look for category directories
        for category_dir in self.config.input_path.iterdir():
            if category_dir.is_dir():
                category_files = self.find_input_files(category_dir)
                # Add category name to each file tuple
                for file_path, _ in category_files:
                    all_files.append((file_path, category_dir.name))
        
        return all_files
    
    def process_files_parallel(self, input_files: List[Tuple[Path, str]]) -> Dict[str, int]:
        """Process files in parallel using multiprocessing"""
        if self.config.num_workers <= 1:
            # Sequential processing
            results = {'train': 0, 'dev': 0, 'test': 0}
            for file_path, category_name in tqdm(input_files, desc="Processing files"):
                file_results = self.processor.process_file(file_path, category_name)
                for split_name, count in file_results.items():
                    results[split_name] += count
            return results
        
        # Parallel processing - use partial to create picklable function
        process_func = partial(_process_file_worker, config=self.config)
        
        with mp.Pool(processes=self.config.num_workers) as pool:
            file_results = list(tqdm(
                pool.imap(process_func, input_files),
                total=len(input_files),
                desc=f"Processing files ({self.config.num_workers} workers)"
            ))
        
        # Aggregate results
        total_results = {'train': 0, 'dev': 0, 'test': 0}
        for file_result in file_results:
            for split_name, count in file_result.items():
                total_results[split_name] += count
        
        return total_results
    
    def run(self):
        """Main execution method"""
        logger.info(f"Starting {self.config.mode} sharding with config:")
        logger.info(f"  Input: {self.config.input_directory}")
        logger.info(f"  Output: {self.config.output_directory}")
        logger.info(f"  Dataset name: {self.config.target_dataset_name}")
        logger.info(f"  Mode: {self.config.mode}")
        if self.config.mode == 'text':
            logger.info(f"  Lines per shard: {self.config.lines_per_shard}")
        else:
            logger.info(f"  Groups per shard: {self.config.groups_per_shard}")
        logger.info(f"  Split ratios: train={self.config.train_ratio}, dev={self.config.dev_ratio}, test={self.config.test_ratio}")
        logger.info(f"  Workers: {self.config.num_workers}")
        logger.info(f"  Nested: {self.config.nested}")
        
        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        
        # Setup output directories
        self.setup_output_directories()
        
        # Find input files
        if self.config.nested:
            input_files = self.find_nested_files()
        else:
            input_files = self.find_input_files()
            
        if not input_files:
            logger.error(f"No valid input files found in {self.config.input_directory} matching pattern '{self.config.file_pattern}'")
            return
        
        logger.info(f"Found {len(input_files)} input files to process")
        
        # Process all files
        results = self.process_files_parallel(input_files)
        
        # Log results
        logger.info("Processing complete!")
        unit = "lines" if self.config.mode == 'text' else "groups"
        logger.info(f"Total {unit} written:")
        for split_name, count in results.items():
            logger.info(f"  {split_name}: {count:,} {unit}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Universal file sharding script for text and JSON files"
    )
    
    # Required arguments
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                      help="Input directory containing files to shard")
    parser.add_argument("-o", "--output_dir", type=str, required=True, 
                      help="Output directory for train/dev/test splits")
    parser.add_argument("-n", "--dataset_name", type=str, required=True,
                      help="Name for the dataset")
    parser.add_argument("-m", "--mode", type=str, choices=['text', 'json'], required=True,
                      help="Processing mode: 'text' for text files, 'json' for JSON files")
    
    # Mode-specific arguments
    parser.add_argument("-l", "--lines_per_shard", type=int,
                      help="Number of lines per shard file (required for text mode)")
    parser.add_argument("-g", "--groups_per_shard", type=int,
                      help="Number of groups per shard file (required for JSON mode)")
    
    # Split ratios
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.8,
                      help="Ratio of data for training set (default: 0.8)")
    parser.add_argument("-dr", "--dev_ratio", type=float, default=0.1,
                      help="Ratio of data for dev set (default: 0.1)")
    parser.add_argument("-te", "--test_ratio", type=float, default=0.1,
                      help="Ratio of data for test set (default: 0.1)")
    
    # Processing options
    parser.add_argument("-w", "--workers", type=int, default=None,
                      help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--pattern", type=str, default="part-*",
                      help="File pattern to match input files (default: 'part-*')")
    parser.add_argument("--nested", action="store_true",
                      help="Use nested directory structure (categories as subdirectories)")
    
    # JSON-specific options
    parser.add_argument("--spacy_variant", type=str, default="trf",
                      help="SpaCy model variant for JSON mode (default: trf)")
    parser.add_argument("--min_trees", type=int, default=2,
                      help="Minimum number of trees per group for JSON mode (default: 2)")
    parser.add_argument("--dataset_type", type=str, default="",
                      help="Dataset type for JSON mode (e.g., 'single', 'multiple')")
    parser.add_argument("--remove_fields", type=str, nargs='*', default=[],
                      help="Fields to remove from JSON groups")
    
    args = parser.parse_args()
    
    # Validate mode-specific requirements
    if args.mode == 'text' and args.lines_per_shard is None:
        parser.error("--lines_per_shard is required when mode is 'text'")
    if args.mode == 'json' and args.groups_per_shard is None:
        parser.error("--groups_per_shard is required when mode is 'json'")
    
    # Create configuration
    config = UniversalShardConfig(
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        target_dataset_name=args.dataset_name,
        mode=args.mode,
        lines_per_shard=args.lines_per_shard,
        groups_per_shard=args.groups_per_shard,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.workers,
        random_seed=args.seed,
        file_pattern=args.pattern,
        spacy_variant=args.spacy_variant,
        remove_fields=args.remove_fields,
        min_trees_per_group=args.min_trees,
        nested=args.nested,
        dataset_type=args.dataset_type
    )
    
    # Run the sharding process
    manager = UniversalFileShardingManager(config)
    manager.run()

if __name__ == "__main__":
    main()
