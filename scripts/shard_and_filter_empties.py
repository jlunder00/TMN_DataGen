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
    if len(group.get("trees", [])) < config.min_trees_per_group:
        return None
        
    processed = {k: v for k, v in group.items() 
                if k not in config.remove_fields}
    
    for old_name, new_name in config.rename_map.items():
        if old_name in processed:
            processed[new_name] = processed.pop(old_name)
            
    return processed

def process_partition_file(file_path: Path, output_base: Path, config: ShardConfig, dir_suffix: str, category_name: str):
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
            os.remove(str(file_path))
            logger.info(f'Removed {file_path}')
            return 0
            
        # Split groups
        random.shuffle(valid_groups)
        total = len(valid_groups)
        train_size = int(total * config.train_ratio)
        dev_size = int(total * config.dev_ratio)
        
        split_data = {
            'train': valid_groups[:train_size],
            'dev': valid_groups[train_size:train_size + dev_size],
            'test': valid_groups[train_size + dev_size:]
        }
        
        # Write shards for each split
        source_name = f"amazonqa_{dir_suffix}_{category_name}"
        file_stem = file_path.stem
        
        for split_name, groups in split_data.items():
            if not groups:
                continue
                
            # Create output directory with category subdirectory
            output_dir = output_base / split_name / f"amazonqa_{dir_suffix}_converted_{config.spacy_variant}_sharded" / category_name
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
                counts_filename = f"{file_stem}_shard_{shard_idx:06d}_counts.json"
                counts_file = output_dir / counts_filename
                with open(counts_file, 'w') as f:
                    json.dump(counts, f)
        
        return len(valid_groups)
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0

def process_category_directory(category_dir: Path, output_base: Path, config: ShardConfig, dir_suffix: str):
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
            dir_suffix=dir_suffix,
            category_name=category_name
        )
        total_groups += groups_processed
    
    logger.info(f"Processed {total_groups} valid groups from {category_name}")
    return total_groups

def process_input_directory(input_dir: Path, output_base: Path, config: ShardConfig, dir_suffix: str):
    """Process a single input directory with its subdirectories using multiprocessing"""
    
    # Find all subdirectories (category directories)
    category_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not category_dirs:
        logger.warning(f"No subdirectories found in {input_dir}")
        return
    
    logger.info(f"Found {len(category_dirs)} category directories in {input_dir}")
    
    # Process categories in parallel if workers > 1
    total_groups = 0
    if config.num_workers > 1:
        process_func = partial(
            process_category_directory,
            output_base=output_base,
            config=config,
            dir_suffix=dir_suffix
        )
        
        with mp.Pool(processes=config.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, category_dirs),
                total=len(category_dirs),
                desc=f"Processing {dir_suffix} categories"
            ))
            total_groups = sum(results)
    else:
        # Process sequentially
        for category_dir in tqdm(category_dirs, desc=f"Processing {dir_suffix} categories"):
            groups_processed = process_category_directory(
                category_dir=category_dir,
                output_base=output_base,
                config=config,
                dir_suffix=dir_suffix
            )
            total_groups += groups_processed
    
    logger.info(f"Processed a total of {total_groups} groups from {dir_suffix}")

def main(input_base: str, output_base: str, config: ShardConfig):
    """Main processing function"""
    input_path = Path(input_base)
    output_path = Path(output_base)
    
    # Make sure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directories for splits
    for split in ['train', 'dev', 'test']:
        (output_path / split).mkdir(exist_ok=True)
    
    # Process dest_multiple directory if exists
    multiple_dir = input_path / "dest_multiples"
    if multiple_dir.exists():
        logger.info(f"Processing multiple-answer QA data from {multiple_dir}")
        process_input_directory(multiple_dir, output_path, config, "multiple")
    
    # Process dest_single directory if exists
    single_dir = input_path / "dest_singles"
    if single_dir.exists():
        logger.info(f"Processing single-answer QA data from {single_dir}")
        process_input_directory(single_dir, output_path, config, "single")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Amazon QA data into train/dev/test splits")
    parser.add_argument("-i", "--input_dir", type=str, required=True, 
                      help="Base input directory containing dest_multiple and dest_single")
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
        num_workers=args.num_workers
    )
    
    logger.info(f"Starting processing with config: {config}")
    main(args.input_dir, args.output_dir, config)
    logger.info("Processing complete!")

# import json
# import random
# import shutil
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Iterator, Set
# from dataclasses import dataclass
# import multiprocessing as mp
# from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm
# import logging
# import argparse
# from multiprocessing import Pool

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# @dataclass
# class ShardConfig:
#     """Configuration for sharding and field modifications"""
#     target_shard_size: int
#     train_ratio: float = 0.8
#     dev_ratio: float = 0.1  
#     test_ratio: float = 0.1
#     spacy_variant: str = "trf"
#     remove_fields: List[str] = None
#     rename_map: Dict[str, str] = None
#     num_processes: int = None
#     min_trees_per_group: int = 2  # Minimum number of trees per group to keep
#     num_workers: int = 1
#     
#     def __post_init__(self):
#         self.remove_fields = self.remove_fields or []
#         self.rename_map = self.rename_map or {}
#         if self.num_processes is None:
#             self.num_processes = max(1, mp.cpu_count() - 1)
#         # Validate ratios
#         assert abs(self.train_ratio + self.dev_ratio + self.test_ratio - 1.0) < 0.001, "Split ratios must sum to 1.0"

# def process_group(group: dict, config: ShardConfig) -> Optional[dict]:
#     """
#     Process a single group according to configuration
#     Returns None if the group should be skipped (e.g., not enough trees)
#     """
#     # Skip groups with too few trees
#     if len(group.get("trees", [])) < config.min_trees_per_group:
#         return None
#         
#     processed = {k: v for k, v in group.items() 
#                 if k not in config.remove_fields}
#     
#     for old_name, new_name in config.rename_map.items():
#         if old_name in processed:
#             processed[new_name] = processed.pop(old_name)
#             
#     return processed

# def _load_and_filter_single(args):
#     """
#     Helper function to load and process groups from a single JSON file.
#     Returns a tuple (metadata, list_of_valid_groups).
#     """
#     file_path, config = args

#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)

#         # metadata will be everything except 'groups'
#         # but we only use the first file's non-empty metadata in main function
#         if 'groups' in data:
#             metadata = {k: v for k, v in data.items() if k != 'groups'}
#             valid_groups = []
#             for group in data['groups']:
#                 processed = process_group(group, config)
#                 if processed:
#                     valid_groups.append(processed)
#             return metadata, valid_groups
#         else:
#             return {}, []
#     except Exception as e:
#         logger.error(f"Error processing file {file_path}: {e}")
#         return {}, []

# def load_and_filter_groups_mp(input_dir: Path, config) -> Tuple[List[dict], dict]:
#     """
#     Load all groups from JSON files in the input directory in parallel
#     Returns (filtered_groups, metadata).
#     
#     :param input_dir: Directory containing part_*.json files
#     :param config: Your configuration object
#     :param num_workers: Number of parallel worker processes
#     """
#     num_workers = config.num_workers
#     # Find all JSON part files
#     part_files = list(input_dir.glob("part_*.json"))

#     if not part_files:
#         logger.warning(f"No part_*.json files found in {input_dir}")
#         return [], {}

#     all_groups = []
#     metadata = {}

#     # Multiprocessing pool
#     with Pool(processes=num_workers) as pool:
#         # Use imap (or starmap) to process files in parallel
#         # Wrapping with tqdm to see progress
#         for meta_part, groups_part in tqdm(
#             pool.imap(_load_and_filter_single, [(f, config) for f in part_files]),
#             total=len(part_files),
#             desc=f"Loading files from {input_dir.name}",
#         ):
#             # Set metadata from the first file that has it (if it's empty so far)
#             if not metadata and meta_part:
#                 metadata = meta_part
#             # Accumulate all processed groups
#             all_groups.extend(groups_part)

#     logger.info(f"Loaded {len(all_groups)} valid groups from {input_dir.name}")
#     return all_groups, metadata

# def load_and_filter_groups(input_dir: Path, config: ShardConfig) -> Tuple[List[dict], dict]:
#     """
#     Load all groups from JSON files in the input directory
#     Returns (filtered_groups, metadata)
#     """
#     all_groups = []
#     metadata = {}
#     
#     # Find all JSON part files (not counts files)
#     part_files = list(input_dir.glob("part_*.json"))
#     
#     if not part_files:
#         logger.warning(f"No part_*.json files found in {input_dir}")
#         return [], {}
#     
#     # Process each file
#     for file_path in tqdm(part_files, desc=f"Loading files from {input_dir.name}"):
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#                 
#             # Get metadata from first file
#             if not metadata and 'groups' in data:
#                 metadata = {k: v for k, v in data.items() if k != 'groups'}
#                 
#             # Process and filter groups
#             if 'groups' in data:
#                 for group in data.get('groups', []):
#                     processed = process_group(group, config)
#                     if processed:
#                         all_groups.append(processed)
#                         
#         except Exception as e:
#             logger.error(f"Error processing file {file_path}: {e}")
#             continue
#     
#     logger.info(f"Loaded {len(all_groups)} valid groups from {input_dir.name}")
#     return all_groups, metadata

# def split_groups(groups: List[dict], config: ShardConfig) -> Dict[str, List[dict]]:
#     """
#     Split groups into train/dev/test sets
#     Returns dict with keys 'train', 'dev', 'test' and corresponding group lists
#     """
#     # Shuffle the groups to ensure randomness
#     random.shuffle(groups)
#     
#     total = len(groups)
#     train_size = int(total * config.train_ratio)
#     dev_size = int(total * config.dev_ratio)
#     
#     splits = {
#         'train': groups[:train_size],
#         'dev': groups[train_size:train_size + dev_size],
#         'test': groups[train_size + dev_size:]
#     }
#     
#     logger.info(f"Split {total} groups into: "
#                f"train={len(splits['train'])}, "
#                f"dev={len(splits['dev'])}, "
#                f"test={len(splits['test'])}")
#     
#     return splits

# def process_and_write_shards(groups: List[dict], 
#                             output_dir: Path,
#                             config: ShardConfig,
#                             metadata: dict,
#                             source_name: str):
#     """
#     Process groups and write shard files with counts
#     """
#     output_dir.mkdir(parents=True, exist_ok=True)
#     
#     # Split groups into shards
#     for shard_idx, start_idx in enumerate(range(0, len(groups), config.target_shard_size)):
#         end_idx = min(start_idx + config.target_shard_size, len(groups))
#         shard_groups = groups[start_idx:end_idx]
#         
#         if not shard_groups:
#             continue
#             
#         # Create base filename stem (without extension)
#         file_stem = f"part_{shard_idx}"
#         
#         shard_data = {
#             **metadata,
#             'groups': shard_groups,
#             'shard_info': {
#                 'shard_index': shard_idx,
#                 'groups_in_shard': len(shard_groups),
#                 'source_name': source_name
#             }
#         }
#         
#         # Write shard file
#         shard_filename = f"{file_stem}_shard_{shard_idx:06d}.json"
#         shard_file = output_dir / shard_filename
#         with open(shard_file, 'w') as f:
#             json.dump(shard_data, f)
#         
#         # Write counts file - using same naming convention as original script
#         counts = {
#             'n_groups': len(shard_groups),
#             'trees_per_group': [len(group.get("trees", [])) for group in shard_groups]
#         }
#         counts_filename = f"{file_stem}_shard_{shard_idx:06d}_counts.json"
#         counts_file = output_dir / counts_filename
#         with open(counts_file, 'w') as f:
#             json.dump(counts, f)
#     
#     logger.info(f"Wrote {(len(groups) + config.target_shard_size - 1) // config.target_shard_size} "
#                f"shards to {output_dir}")

# def process_input_directory(input_dir: Path, 
#                            output_base: Path, 
#                            config: ShardConfig,
#                            dir_suffix: str):
#     """Process a single input directory with its subdirectories"""
#     
#     # Find all subdirectories (category directories)
#     category_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
#     
#     if not category_dirs:
#         logger.warning(f"No subdirectories found in {input_dir}")
#         return
#     
#     logger.info(f"Found {len(category_dirs)} category directories in {input_dir}")
#     
#     # Dictionary to collect groups from all categories
#     all_groups_by_category = {}
#     metadata = {}
#     
#     # Process each category directory
#     for category_dir in tqdm(category_dirs, desc=f"Processing {dir_suffix} categories"):
#         # Load and filter groups from this category
#         category_groups, category_metadata = load_and_filter_groups(category_dir, config) if config.num_workers < 2 else load_and_filter_groups_mp(category_dir, config)
#         
#         if not category_groups:
#             logger.warning(f"No valid groups found in {category_dir}")
#             continue
#         
#         category_name, groups = category_dir.name, category_groups
#     #     # Store valid groups with their category
#     #     all_groups_by_category[category_dir.name] = category_groups
#     #     
#     #     # Store metadata from the first category that has valid groups
#     #     if not metadata and category_metadata:
#     #         metadata = category_metadata
#     # 
#     # # Split all groups by category
#     # for category_name, groups in all_groups_by_category.items():
#         # Split just this category's groups
#         split_data = split_groups(groups, config)
#         
#         # Create output directories and write shards for each split
#         for split_name, split in split_data.items():
#             if not split:
#                 continue
#                 
#             # Create output directory preserving the category structure
#             output_dir = output_base / split_name / f"amazonqa_{dir_suffix}_converted_{config.spacy_variant}_sharded" / category_name
#             output_dir.mkdir(parents=True, exist_ok=True)
#             
#             # Write shards for this category and split
#             process_and_write_shards(
#                 groups=split,
#                 output_dir=output_dir,
#                 config=config,
#                 metadata=metadata,
#                 source_name=f"amazonqa_{dir_suffix}/{category_name}"
#             )

# def main(input_base: str, output_base: str, config: ShardConfig):
#     """Main processing function"""
#     input_path = Path(input_base)
#     output_path = Path(output_base)
#     
#     # Make sure output directory exists
#     output_path.mkdir(parents=True, exist_ok=True)
#     
#     # Create directories for splits
#     for split in ['train', 'dev', 'test']:
#         (output_path / split).mkdir(exist_ok=True)
#     
#     # Process dest_multiple directory if exists
#     multiple_dir = input_path / "dest_multiples"
#     if multiple_dir.exists():
#         logger.info(f"Processing multiple-answer QA data from {multiple_dir}")
#         process_input_directory(multiple_dir, output_path, config, "multiple")
#     
#     # Process dest_single directory if exists
#     single_dir = input_path / "dest_singles"
#     if single_dir.exists():
#         logger.info(f"Processing single-answer QA data from {single_dir}")
#         process_input_directory(single_dir, output_path, config, "single")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process Amazon QA data into train/dev/test splits")
#     parser.add_argument("-i", "--input_dir", type=str, required=True, 
#                       help="Base input directory containing dest_multiple and dest_single")
#     parser.add_argument("-o", "--output_dir", type=str, required=True,
#                       help="Base output directory for train/dev/test splits")
#     parser.add_argument("-s", "--shard_size", type=int, default=500,
#                       help="Target number of groups per shard")
#     parser.add_argument("-tr", "--train_ratio", type=float, default=0.8,
#                       help="Ratio of data for training set")
#     parser.add_argument("-dr", "--dev_ratio", type=float, default=0.1,
#                       help="Ratio of data for dev/validation set")
#     parser.add_argument("-te", "--test_ratio", type=float, default=0.1,
#                       help="Ratio of data for test set")
#     parser.add_argument("--spacy_variant", type=str, default="trf",
#                       help="SpaCy model variant (trf, lg, sm)")
#     parser.add_argument("-m", "--min_trees", type=int, default=2,
#                       help="Minimum number of trees per group to keep")
#     parser.add_argument("-p", "--processes", type=int, default=None,
#                       help="Number of processes to use (default: CPU count - 1)")
#     parser.add_argument("--seed", type=int, default=42,
#                       help="Random seed for reproducibility")
#     parser.add_argument("--num_workers", type=int, default=1,
#                       help="number of workers for multiprocessing. single threading used when <2")
#     
#     args = parser.parse_args()
#     
#     # Set random seed
#     random.seed(args.seed)
#     
#     # Create config
#     config = ShardConfig(
#         target_shard_size=args.shard_size,
#         train_ratio=args.train_ratio,
#         dev_ratio=args.dev_ratio,
#         test_ratio=args.test_ratio,
#         spacy_variant=args.spacy_variant,
#         min_trees_per_group=args.min_trees,
#         num_processes=args.processes,
#         num_workers = args.num_workers
#     )
#     
#     logger.info(f"Starting processing with config: {config}")
#     main(args.input_dir, args.output_dir, config)
#     logger.info("Processing complete!")
