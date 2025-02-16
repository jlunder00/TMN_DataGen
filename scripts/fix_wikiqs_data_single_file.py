import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class ShardConfig:
    """Configuration for sharding and field modifications"""
    shard_size: int
    remove_fields: List[str] = None
    rename_map: Dict[str, str] = None
    
    def __post_init__(self):
        self.remove_fields = self.remove_fields or []
        self.rename_map = self.rename_map or {}

def get_group_shard_indices(groups: List[dict], shard_size: int) -> List[tuple]:
    """
    Calculate shard indices for groups
    Returns list of (start_idx, end_idx) tuples for each shard
    """
    total_size = len(groups)
    num_shards = (total_size + shard_size - 1) // shard_size  # Ceiling division
    
    return [(i * shard_size, min((i + 1) * shard_size, total_size)) 
            for i in range(num_shards)]

def process_group(group: dict, config: ShardConfig) -> dict:
    """Process a single group according to configuration"""
    # Remove unwanted fields
    processed = {k: v for k, v in group.items() 
                if k not in config.remove_fields}
    
    # Rename fields according to map
    for old_name, new_name in config.rename_map.items():
        if old_name in processed:
            processed[new_name] = processed.pop(old_name)
            
    return processed

def shard_data(input_file: str, output_dir: str, config: Optional[ShardConfig] = None):
    """
    Shard JSON data according to configuration
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory for output shards
        config: ShardConfig object with processing settings
    """
    if config is None:
        config = ShardConfig(shard_size=1000)  # Default configuration
        
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get metadata and groups
    metadata = {k: v for k, v in data.items() if k != 'groups'}
    groups = data['groups']
    
    # Calculate shard indices
    shard_indices = get_group_shard_indices(groups, config.shard_size)
    
    # Process and write each shard
    for shard_idx, (start_idx, end_idx) in enumerate(shard_indices):
        shard_groups = [process_group(group, config) 
                       for group in groups[start_idx:end_idx]]
        
        shard_data = {
            **metadata,  # Include all metadata
            'groups': shard_groups,
            'shard_info': {
                'shard_index': shard_idx,
                'total_shards': len(shard_indices),
                'start_idx': start_idx,
                'end_idx': end_idx
            }
        }
        
        # Write shard to file
        output_file = output_path / f'shard_{shard_idx:04d}.json'
        with open(output_file, 'w') as f:
            json.dump(shard_data, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Configuration for your specific case
    config = ShardConfig(
        shard_size=1000,
        remove_fields=['trees2', 'text2'],
        rename_map={
            'trees1': 'trees',
            'text1': 'text'
        }
    )
    
    shard_data(
        input_file='/home/jlunder/research/data/wikiqs/dest3/part-00000/part_0.json',
        output_dir='/home/jlunder/research/data/wikiqs/dest3/test_output_shard/',
        config=config
    )
