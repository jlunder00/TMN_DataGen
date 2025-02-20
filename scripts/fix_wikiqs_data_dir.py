import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from functools import partial
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ShardConfig:
    """Configuration for sharding and field modifications"""
    target_shard_size: int
    remove_fields: List[str] = None
    rename_map: Dict[str, str] = None
    num_processes: int = None
    chunk_size: int = 5  # Number of files to process at once
    
    def __post_init__(self):
        self.remove_fields = self.remove_fields or []
        self.rename_map = self.rename_map or {}
        if self.num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 1)

def process_group(group: dict, config: ShardConfig) -> dict:
    """Process a single group according to configuration"""
    processed = {k: v for k, v in group.items() 
                if k not in config.remove_fields}
    
    for old_name, new_name in config.rename_map.items():
        if old_name in processed:
            processed[new_name] = processed.pop(old_name)
            
    return processed

def process_and_write_chunk(input_files: List[Path], 
                              output_path: Path,
                              start_shard_idx: int,
                              config: ShardConfig,
                              metadata: dict) -> int:
    """
    Process a chunk of partition files and write results directly.
    
    Each input file is a partition file. For each partition file, its groups are split into 
    shards (chunks of groups with size config.target_shard_size). Each shard file is named 
    using the original input file's stem with an appended suffix _shard_<global_shard_idx:06d>.json.
    
    In addition, for each shard a corresponding counts file is written with the same stem plus 
    _shard_<global_shard_idx:06d>_counts.json, containing:
      - 'n_groups': number of groups in the shard.
      - 'trees_per_group': a list with the number of trees per group.
    
    The start_shard_idx is used to maintain shard numbering across chunks. The function returns 
    the total number of groups processed across all input files.
    """
    import json
    groups_processed = 0
    current_shard_idx = start_shard_idx

    # Process each partition file individually.
    for file_path in input_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Process groups from this partition file.
            groups = [process_group(group, config) for group in data['groups']]
            groups_processed += len(groups)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            continue

        # Split groups from this partition into shards.
        for i in range(0, len(groups), config.target_shard_size):
            shard_groups = groups[i:i + config.target_shard_size]
            
            shard_data = {
                **metadata,
                'groups': shard_groups,
                'shard_info': {
                    'shard_index': current_shard_idx,
                    'groups_in_shard': len(shard_groups),
                    'source_file': file_path.name
                }
            }
            
            # File naming: keep original partition file's stem and append _shard_<global_shard_idx>.
            shard_filename = f"{file_path.stem}_shard_{current_shard_idx:06d}.json"
            output_file = output_path / shard_filename
            with open(output_file, 'w') as f:
                json.dump(shard_data, f)
            
            # Create the counts file alongside the shard file.
            counts = {
                'n_groups': len(shard_groups),
                'trees_per_group': [len(group.get("trees", [])) for group in shard_groups]
            }
            counts_filename = f"{file_path.stem}_shard_{current_shard_idx:06d}_counts.json"
            counts_file = output_path / counts_filename
            with open(counts_file, 'w') as f:
                json.dump(counts, f)
            
            current_shard_idx += 1

    return groups_processed


# def process_and_write_chunk(input_files: List[Path], 
#                           output_path: Path,
#                           start_shard_idx: int,
#                           config: ShardConfig,
#                           metadata: dict) -> int:
#     """Process a chunk of input files and write results directly"""
#     all_groups = []
#     groups_processed = 0
#     
#     # Process each file in the chunk
#     for file_path in input_files:
#         try:
#             with open(file_path, 'r') as f:
#                 data = json.load(f)
#                 groups = [process_group(group, config) for group in data['groups']]
#                 all_groups.extend(groups)
#                 groups_processed += len(groups)
#         except Exception as e:
#             logger.error(f"Error processing file {file_path}: {str(e)}")
#             continue

#     # Write shards for this chunk
#     current_shard_idx = start_shard_idx
#     for i in range(0, len(all_groups), config.target_shard_size):
#         shard_groups = all_groups[i:i + config.target_shard_size]
#         
#         shard_data = {
#             **metadata,
#             'groups': shard_groups,
#             'shard_info': {
#                 'shard_index': current_shard_idx,
#                 'groups_in_shard': len(shard_groups),
#                 'source_files': [str(f.name) for f in input_files]
#             }
#         }
#         
#         output_file = output_path / f'shard_{current_shard_idx:06d}.json'
#         with open(output_file, 'w') as f:
#             json.dump(shard_data, f)
#         
#         current_shard_idx += 1
#     
#     return groups_processed

def chunk_iterator(items: List, chunk_size: int) -> Iterator[List]:
    """Yield chunks of the input list"""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

def reshard_data(input_dir: str, output_dir: str, config: Optional[ShardConfig] = None):
    """
    Process and reshard multiple input files into new shards
    """
    if config is None:
        config = ShardConfig(target_shard_size=1000)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all input files
    input_files = list(input_path.glob('*.json'))
    if not input_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    
    # Get metadata from first file
    with open(input_files[0], 'r') as f:
        first_file = json.load(f)
        metadata = {k: v for k, v in first_file.items() if k != 'groups'}
    
    # Process files in chunks
    total_groups_processed = 0
    current_shard_idx = 0
    
    with mp.Pool(processes=config.num_processes) as pool:
        chunks = list(chunk_iterator(input_files, config.chunk_size))
        
        with tqdm(total=len(input_files), desc="Processing files") as pbar:
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    process_func = partial(
                        process_and_write_chunk,
                        output_path=output_path,
                        start_shard_idx=current_shard_idx,
                        config=config,
                        metadata=metadata
                    )
                    
                    # Process chunk
                    groups_processed = process_func(chunk)
                    total_groups_processed += groups_processed
                    
                    # Update progress
                    pbar.update(len(chunk))
                    
                    # Calculate next shard index
                    shards_in_chunk = (groups_processed + config.target_shard_size - 1) // config.target_shard_size
                    current_shard_idx += shards_in_chunk
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                    continue
    
    logger.info(f"Processed {total_groups_processed} groups into {current_shard_idx} shards")

if __name__ == "__main__":
    config = ShardConfig(
        target_shard_size=500,
        remove_fields=['trees2', 'text2'],
        rename_map={
            'trees1': 'trees',
            'text1': 'text'
        },
        num_processes=12,
        chunk_size=1  # Process 5 files at a time
    )
    
    reshard_data(
        input_dir='/home/jlunder/research/data/wikiqs/dest3/part-00000/',
        output_dir='/home/jlunder/research/data/wikiqs/dest3/test_output_shard/',
        config=config
    )
