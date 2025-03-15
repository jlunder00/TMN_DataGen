import json
import random
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Set
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ShardConfig:
    """Configuration for sharding and field modifications"""
    target_shard_size: int
    train_ratio: float = 0.8
    dev_ratio: float = 0.1  
    test_ratio: float = 0.1
    spacy_variant: str = "trf"
    remove_fields: List[str] = None
    rename_map: Dict[str, str] = None
    num_processes: int = None
    min_trees_per_group: int = 2  # Minimum number of trees per group to keep
    num_workers: int = 1
    nested: bool = True  # Whether the input directory has nested structure
    dataset_name: str = "amazonqa"  # Name of the dataset for output directory naming
    
    def __post_init__(self):
        self.remove_fields = self.remove_fields or []
        self.rename_map = self.rename_map or {}
        if self.num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)
        # Validate ratios
        assert abs(self.train_ratio + self.dev_ratio + self.test_ratio - 1.0) < 0.001, "Split ratios must sum to 1.0"

def process_group(group: dict, config: ShardConfig) -> Optional[dict]:
    """
    Process a single group according to configuration
    Returns None if the group should be skipped (e.g., not enough trees)
    """
    # Skip groups with too few trees
    if len(group.get("trees", [])) + len(group.get("trees_b", [])) < config.min_trees_per_group:
        return None
        
    processed = {k: v for k, v in group.items() 
                if k not in config.remove_fields}
    
    for old_name, new_name in config.rename_map.items():
        if old_name in processed:
            processed[new_name] = processed.pop(old_name)
            
    return processed

def process_partition_file(file_path: Path, output_base: Path, config: ShardConfig, dataset_type: str, category_name: str = None):
    """
    Process a single partition file:
    1. Load and filter groups
    2. Split into train/dev/test
    3. Shard each split and write files
    """
    try:
        # Load and process file
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract metadata
        metadata = {k: v for k, v in data.items() if k != 'groups'}
        
        # Process and filter groups
        valid_groups = []
        for group in data.get('groups', []):
            processed = process_group(group, config)
            if processed:
                valid_groups.append(processed)
        
        if not valid_groups:
            logger.info(f"No valid groups found in {file_path.name}")
            return 0
            
        # Split groups
        random.shuffle(valid_groups)
        total = len(valid_groups)
        train_size = int(total * config.train_ratio)
        dev_size = int(total * config.dev_ratio)
        
        split_data = {}
        if train_size > 0:
            split_data['train'] =  valid_groups[:train_size]
        if dev_size > 0:
            split_data['dev'] = valid_groups[train_size:train_size + dev_size]
        test_size = len(valid_groups) - (train_size+dev_size)
        if test_size > 0:
            split_data['test'] = valid_groups[train_size + dev_size:]
        
        # Determine source name and directory structure
        if config.nested:
            source_name = f"{config.dataset_name}_{dataset_type}_{category_name if category_name else ''}"
            base_dir_name = f"{config.dataset_name}_{dataset_type}_converted_{config.spacy_variant}_sharded"
        else:
            source_name = f"{config.dataset_name}"
            base_dir_name = f"{config.dataset_name}_converted_{config.spacy_variant}_sharded"
            
        file_stem = file_path.stem
        
        for split_name, groups in split_data.items():
            if not groups:
                continue
                
            # Create output directory
            if config.nested and category_name:
                # Nested structure with category subdirectory
                output_dir = output_base / split_name / base_dir_name / category_name
            else:
                # Flat structure without category
                output_dir = output_base / split_name / base_dir_name
                
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write shard(s)
            for shard_idx, start_idx in enumerate(range(0, len(groups), config.target_shard_size)):
                end_idx = min(start_idx + config.target_shard_size, len(groups))
                shard_groups = groups[start_idx:end_idx]
                
                shard_data = {
                    **metadata,
                    'groups': shard_groups,
                    'shard_info': {
                        'shard_index': shard_idx,
                        'groups_in_shard': len(shard_groups),
                        'source_name': source_name,
                        'source_file': file_path.name
                    }
                }
                
                # Write shard file - use original file stem to maintain traceability
                shard_filename = f"{file_stem}_shard_{shard_idx:06d}.json"
                shard_file = output_dir / shard_filename
                with open(shard_file, 'w') as f:
                    json.dump(shard_data, f)
                
                # Write counts file
                counts = {
                    'n_groups': len(shard_groups),
                    'trees_per_group': [len(group.get("trees", [])) for group in shard_groups]
                }
                if shard_groups and 'trees_b' in shard_groups[0]:
                    counts['trees_b_per_group'] = [len(group.get('trees_b', [])) for group in shard_groups]
                counts_filename = f"{file_stem}_shard_{shard_idx:06d}_counts.json"
                counts_file = output_dir / counts_filename
                with open(counts_file, 'w') as f:
                    json.dump(counts, f)
        
        return len(valid_groups)
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0

def process_category_directory(category_dir: Path, output_base: Path, config: ShardConfig, dataset_type: str):
    """Process all partition files in a category directory"""
    category_name = category_dir.name
    
    # Find all JSON part files (not counts files)
    part_files = list(category_dir.glob("part_*.json"))
    
    if not part_files:
        logger.warning(f"No part_*.json files found in {category_dir}")
        return 0
    
    logger.info(f"Processing {len(part_files)} files in category {category_name}")
    
    # Process each file individually
    total_groups = 0
    for file_path in tqdm(part_files, desc=f"Processing {category_name}", leave=False):
        groups_processed = process_partition_file(
            file_path=file_path,
            output_base=output_base,
            config=config,
            dataset_type=dataset_type,
            category_name=category_name
        )
        total_groups += groups_processed
    
    logger.info(f"Processed {total_groups} valid groups from {category_name}")
    return total_groups

def process_flat_directory(input_dir: Path, output_base: Path, config: ShardConfig):
    """Process a directory with partition files directly in it (non-nested) using multiprocessing"""
    # Find all JSON part files (not counts files)
    part_files = list(input_dir.glob("part_*.json"))
    
    if not part_files:
        logger.warning(f"No part_*.json files found in {input_dir}")
        return 0
    
    logger.info(f"Processing {len(part_files)} files in flat directory {input_dir}")
    
    # Process files in parallel if workers > 1
    total_groups = 0
    if config.num_workers > 1:
        process_func = partial(
            process_partition_file,
            output_base=output_base,
            config=config,
            dataset_type="",  # No dataset type for flat directory
            category_name=None  # No category for flat directory
        )
        
        with mp.Pool(processes=config.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, part_files),
                total=len(part_files),
                desc=f"Processing partition files"
            ))
            total_groups = sum(results)
    else:
        # Process sequentially
        for file_path in tqdm(part_files, desc=f"Processing files", leave=False):
            groups_processed = process_partition_file(
                file_path=file_path,
                output_base=output_base,
                config=config,
                dataset_type="",  # No dataset type for flat directory
                category_name=None  # No category for flat directory
            )
            total_groups += groups_processed
    
    logger.info(f"Processed a total of {total_groups} groups from flat directory")
    return total_groups

def process_nested_directory(input_dir: Path, output_base: Path, config: ShardConfig, dataset_type: str):
    """Process a directory with nested category directories"""
    # Find all subdirectories (category directories)
    category_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not category_dirs:
        logger.warning(f"No subdirectories found in {input_dir}")
        return 0
    
    logger.info(f"Found {len(category_dirs)} category directories in {input_dir}")
    
    # Process categories in parallel if workers > 1
    total_groups = 0
    if config.num_workers > 1:
        process_func = partial(
            process_category_directory,
            output_base=output_base,
            config=config,
            dataset_type=dataset_type
        )
        
        with mp.Pool(processes=config.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, category_dirs),
                total=len(category_dirs),
                desc=f"Processing {dataset_type} categories"
            ))
            total_groups = sum(results)
    else:
        # Process sequentially
        for category_dir in tqdm(category_dirs, desc=f"Processing {dataset_type} categories"):
            groups_processed = process_category_directory(
                category_dir=category_dir,
                output_base=output_base,
                config=config,
                dataset_type=dataset_type
            )
            total_groups += groups_processed
    
    logger.info(f"Processed a total of {total_groups} groups from {dataset_type}")
    return total_groups

def main(input_base: str, output_base: str, config: ShardConfig):
    """Main processing function"""
    input_path = Path(input_base)
    output_path = Path(output_base)
    
    # Make sure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories for splits
    for split in ['train', 'dev', 'test']:
        (output_path / split).mkdir(exist_ok=True)
    
    if config.nested:
        # Process nested structure (Amazon QA style)
        # Process dest_multiple directory if exists
        multiple_dir = input_path / "dest_multiples"
        if multiple_dir.exists():
            logger.info(f"Processing multiple-answer QA data from {multiple_dir}")
            process_nested_directory(multiple_dir, output_path, config, "multiple")
        
        # Process dest_single directory if exists
        single_dir = input_path / "dest_singles"
        if single_dir.exists():
            logger.info(f"Processing single-answer QA data from {single_dir}")
            process_nested_directory(single_dir, output_path, config, "single")
            
        if not multiple_dir.exists() and not single_dir.exists():
            logger.warning(f"No expected subdirectories found in {input_path}")
    else:
        # Process flat structure (partition files directly in input directory)
        logger.info(f"Processing partition files directly from {input_path}")
        process_flat_directory(input_path, output_path, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and shard dataset into train/dev/test splits")
    parser.add_argument("-i", "--input_dir", type=str, required=True, 
                      help="Base input directory")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                      help="Base output directory for train/dev/test splits")
    parser.add_argument("-s", "--shard_size", type=int, default=500,
                      help="Target number of groups per shard")
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.8,
                      help="Ratio of data for training set")
    parser.add_argument("-dr", "--dev_ratio", type=float, default=0.1,
                      help="Ratio of data for dev/validation set")
    parser.add_argument("-te", "--test_ratio", type=float, default=0.1,
                      help="Ratio of data for test set")
    parser.add_argument("--spacy_variant", type=str, default="trf",
                      help="SpaCy model variant (trf, lg, sm)")
    parser.add_argument("-m", "--min_trees", type=int, default=2,
                      help="Minimum number of trees per group to keep")
    parser.add_argument("-p", "--processes", type=int, default=None,
                      help="Number of processes to use (default: CPU count - 1)")
    parser.add_argument("--num_workers", type=int, default=1,
                      help="Number of workers for multiprocessing. Single threading used when <2")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--nested", action="store_true",
                      help="Use if input has nested structure (Amazon QA style)")
    parser.add_argument("--dataset_name", type=str, default="amazonqa",
                      help="Name of dataset (used in output directory naming)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create config
    config = ShardConfig(
        target_shard_size=args.shard_size,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        spacy_variant=args.spacy_variant,
        min_trees_per_group=args.min_trees,
        num_processes=args.processes,
        num_workers=args.num_workers,
        nested=args.nested,
        dataset_name=args.dataset_name
    )
    
    logger.info(f"Starting processing with config: {config}")
    main(args.input_dir, args.output_dir, config)
    logger.info("Processing complete!")
