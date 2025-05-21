# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/


#TMN_DataGen/utils/embedding_cache.py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from ..utils.logging_config import setup_logger
from typing import Dict, Optional
import gc

class ParallelEmbeddingCache:
    """A drop-in replacement for FeatureExtractor's embedding cache that supports parallel processing"""
    
    def __init__(self, 
                 cache_dir: Path,
                 shard_size: int = 10000,
                 num_workers: Optional[int] = None,
                 config: Optional[Dict] = None,
                 device = torch.device('cuda')):
        """
        Initialize the parallel embedding cache system.
        
        Args:
            cache_dir: Directory to store cache files
            shard_size: Number of embeddings per shard
            num_workers: Number of parallel workers (defaults to CPU count)
            config: Configuration dictionary
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.num_workers = num_workers or min(mp.cpu_count(), 4)  # Limit default workers
        self.embedding_cache: Dict[str, torch.Tensor] = {}
        self._current_shard = 0
        self._items_in_current_shard = 0
        self.config = config or {}
        self.device = device
        self.logger = setup_logger(
            self.__class__.__name__,
            self.config.get('verbose', 'normal')
        )
        
        # Track which shard each item belongs to
        self.item_to_shard: Dict[str, int] = {}
        # Track which items have been modified
        self.modified_items: set = set()

    def _get_shard_path(self, shard_idx: int) -> Path:
        """Get the path for a specific shard file."""
        return self.cache_dir / f"embedding_cache_shard_{shard_idx}.npz"

    def __getitem__(self, word: str) -> Optional[torch.Tensor]:
        """Enable dictionary-like access to cache."""
        if word == 'file':
            word = '\\file'
        # return self.embedding_cache.get(word).cpu() if self.device == 'cpu' else self.embedding_cache.get(word)
        return self.embedding_cache.get(word)

    def __setitem__(self, word: str, embedding: torch.Tensor):
        """Enable dictionary-like setting of cache values."""
        if word == 'file':
            word = '\\file'
            
        # Check if this is a new item or a modification
        is_new = word not in self.embedding_cache
        is_modified = not is_new and not torch.equal(embedding, self.embedding_cache[word])
        
        if is_new:
            # New item - add to cache and assign to current shard
            self.embedding_cache[word] = embedding.to(self.device)
            self.item_to_shard[word] = self._current_shard
            self._items_in_current_shard += 1
        elif is_modified:
            # Modified item - update in cache and track it
            self.embedding_cache[word] = embedding.to(self.device)
            self.modified_items.add(word)
        else:
            # No change - do nothing
            return
            
        # Auto-save shards when they reach the size limit
        if self._items_in_current_shard >= self.shard_size:
            if self._save_current_shard():
                self._current_shard += 1
            self._items_in_current_shard = 0

    def __contains__(self, word: str) -> bool:
        """Enable 'in' operator for cache."""
        if word == 'file':
            word = '\\file'
        return word in self.embedding_cache

    def items(self):
        """Enable items() access for cache."""
        return self.embedding_cache.items()

    def _save_current_shard(self):
        """Save the current shard of embeddings."""
        if not self.embedding_cache:
            return False

        # Get only items that belong to current shard
        shard_path = self._get_shard_path(self._current_shard)
        shard_items = {word: emb for word, emb in self.embedding_cache.items() 
                      if self.item_to_shard.get(word) == self._current_shard}
        
        if shard_items:
            # Convert to numpy and save
            np_data = {word: emb.cpu().numpy() for word, emb in shard_items.items()}
            self.logger.info(f"ok really maybe saving fr: {shard_path}")
            np.savez(shard_path, **np_data)
            return True
        return False

    @staticmethod
    def _load_shard(shard_path: Path) -> Dict[str, np.ndarray]:
        """Load a single shard of embeddings."""
        if not shard_path.exists():
            return {}
        
        try:
            with np.load(shard_path, allow_pickle=True) as npz:
                # Return as numpy arrays initially to avoid PyTorch shared memory issues
                return {word: emb for word, emb in npz.items()}
        except Exception as e:
            print(f"Error loading shard {shard_path}: {e}")
            return {}

    def _convert_to_torch(self, numpy_dict: Dict[str, np.ndarray]) -> None:
        """Convert numpy arrays to torch tensors and add to cache."""
        for word, emb in numpy_dict.items():
            self.embedding_cache[word] = torch.from_numpy(emb).to(self.device)

    def load(self):
        """Load all cached embeddings from shards."""
        shard_paths = sorted(self.cache_dir.glob("embedding_cache_shard_*.npz"))
        
        if not shard_paths:
            old_cache = self.cache_dir / "embedding_cache.npz"
            if old_cache.exists():
                self.logger.info("Found old-style cache file, loading it")
                with np.load(old_cache, allow_pickle=True) as cache_data:
                    # Load as numpy first
                    numpy_cache = {
                        word: emb for word, emb in 
                        tqdm(cache_data.items(), desc="Loading old cache")
                    }
                # Convert to torch tensors
                for word, emb in tqdm(numpy_cache.items(), desc="Converting to torch tensors"):
                    self.embedding_cache[word] = torch.from_numpy(emb).to(device)
                    # Assign to current shard
                    self.item_to_shard[word] = self._current_shard
                
                self.logger.info(f"Loaded {len(self.embedding_cache)} items from old cache")
                # Save in new format and optionally remove old cache
                self.save()
                if self.config.get('remove_old_cache', False):
                    self.logger.info("Removing old cache file")
                    old_cache.unlink()
                return
            else:
                self.logger.info("No cache files found")
                return

        self.logger.info(f"Loading embeddings from {len(shard_paths)} shards")

        # Process shards sequentially if only one worker
        if self.num_workers == 1:
            for shard_idx, shard_path in enumerate(tqdm(shard_paths, desc="Loading embedding shards")):
                numpy_data = self._load_shard(shard_path)
                self._convert_to_torch(numpy_data)
                
                # Track which shard each word came from
                for word in numpy_data.keys():
                    self.item_to_shard[word] = shard_idx
                    
                gc.collect()  # Help manage memory
        else:
            # Process shards in parallel
            shard_data_map = {}
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Use map to preserve order
                results = list(tqdm(
                    executor.map(self._load_shard, shard_paths),
                    total=len(shard_paths),
                    desc="Loading embedding shards"
                ))
                
                # Store with indices to maintain shard mapping
                for i, data in enumerate(results):
                    shard_data_map[i] = data
            
            # Process in order of shards
            for shard_idx in range(len(shard_paths)):
                numpy_data = shard_data_map[shard_idx]
                self._convert_to_torch(numpy_data)
                
                # Track which shard each word came from
                for word in numpy_data.keys():
                    self.item_to_shard[word] = shard_idx
                    
                gc.collect()  # Help manage memory
            
        self._current_shard = len(shard_paths)
        self.logger.info(f"Successfully loaded {len(self.embedding_cache)} embeddings")

    def save(self):
        """Save all cached embeddings to shards."""
        if not self.embedding_cache:
            self.logger.info("No embeddings to save")
            return

        # Save any remaining items in the current shard
        if self._items_in_current_shard > 0:
            self.logger.info("maybe saving fr")
            self._save_current_shard()
            
        # Handle modified items that need to be updated in existing shards
        if self.modified_items:
            self.logger.info(f"Updating {len(self.modified_items)} modified items in shards")
            shard_updates = {}
            
            # Group modified items by shard
            for word in self.modified_items:
                shard_idx = self.item_to_shard.get(word)
                if shard_idx is not None and shard_idx != self._current_shard:
                    if shard_idx not in shard_updates:
                        shard_updates[shard_idx] = {}
                    shard_updates[shard_idx][word] = self.embedding_cache[word].to(self.device)
            
            # Update each affected shard
            for shard_idx, items in shard_updates.items():
                shard_path = self._get_shard_path(shard_idx)
                if shard_path.exists():
                    # Load existing shard
                    existing_data = self._load_shard(shard_path)
                    # Update with modified items
                    for word, emb in items.items():
                        existing_data[word] = emb.cpu().numpy()
                    # Save updated shard
                    self.logger.info(f"Updating shard {shard_idx} with {len(items)} modified items")
                    np.savez(shard_path, **existing_data)
            
            # Clear modified items after saving
            self.modified_items.clear()

        self.logger.info(f"Saved {len(self.embedding_cache)} embeddings across {self._current_shard + 1} shards")

    def __len__(self):
        """Enable len() operator for cache."""
        return len(self.embedding_cache)
        
    def __del__(self):
        """Ensure modified items are saved when object is destroyed."""
        if hasattr(self, 'modified_items') and self.modified_items:
            try:
                self.save()
            except Exception as e:
                print(f"Error saving cache during cleanup: {e}")


# class ParallelEmbeddingCache:
#     """A drop-in replacement for FeatureExtractor's embedding cache that supports parallel processing"""
#     
#     def __init__(self, 
#                  cache_dir: Path,
#                  shard_size: int = 10000,
#                  num_workers: Optional[int] = None,
#                  config: Optional[Dict] = None,
#                  device: Optional[str] = 'cuda'):
#         """
#         Initialize the parallel embedding cache system.
#         
#         Args:
#             cache_dir: Directory to store cache files
#             shard_size: Number of embeddings per shard
#             num_workers: Number of parallel workers (defaults to CPU count)
#             config: Configuration dictionary
#         """
#         self.cache_dir = cache_dir
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.shard_size = shard_size
#         self.num_workers = num_workers or min(mp.cpu_count(), 4)  # Limit default workers
#         self.embedding_cache: Dict[str, torch.Tensor] = {}
#         self._current_shard = 0
#         self._items_in_current_shard = 0
#         self.config = config or {}
#         self.device = device
#         self.logger = setup_logger(
#             self.__class__.__name__,
#             self.config.get('verbose', 'normal')
#         )

#     def _get_shard_path(self, shard_idx: int) -> Path:
#         """Get the path for a specific shard file."""
#         return self.cache_dir / f"embedding_cache_shard_{shard_idx}.npz"

#     def __getitem__(self, word: str) -> Optional[torch.Tensor]:
#         """Enable dictionary-like access to cache."""
#         if word == 'file':
#             word = '\\file'
#         return self.embedding_cache.get(word).cpu() if self.device == 'cpu' else self.embedding_cache.get(word)

#     def __setitem__(self, word: str, embedding: torch.Tensor):
#         """Enable dictionary-like setting of cache values."""
#         if word == 'file':
#             word = '\\file'
#         if word not in self.embedding_cache:
#             self.embedding_cache[word] = embedding
#         
#         elif embedding != self.embedding_cache[word]:
#             #replace it such that when you take the list of .items(), its in the new shard
#             del self.embedding_cache[word]
#             self.embedding_cache[word] = embedding
#         else:
#             return
#         self._items_in_current_shard += 1
#         # Auto-save shards when they reach the size limit
#         if self._items_in_current_shard >= self.shard_size:
#             if self._save_current_shard():
#                 self._current_shard += 1
#             self._items_in_current_shard = 0

#     def __contains__(self, word: str) -> bool:
#         """Enable 'in' operator for cache."""
#         if word == 'file':
#             word = '\\file'
#         return word in self.embedding_cache

#     def items(self):
#         """Enable items() access for cache."""
#         return self.embedding_cache.items()

#     def _save_current_shard(self):
#         """Save the current shard of embeddings."""
#         if not self.embedding_cache:
#             return False

#         shard_path = self._get_shard_path(self._current_shard)
#         items_to_save = list(self.embedding_cache.items())
#         start_idx = self._current_shard * self.shard_size
#         end_idx = start_idx + self.shard_size
#         shard_items = items_to_save[start_idx:end_idx]
#         
#         if shard_items:
#             # Convert to numpy and save
#             np_data = {word: emb.cpu().numpy() for word, emb in shard_items}
#             self.logger.info(f"ok really maybe saving fr: {shard_path}")
#             np.savez(shard_path, **np_data)
#             return True
#         return False

#     @staticmethod
#     def _load_shard(shard_path: Path) -> Dict[str, np.ndarray]:
#         """Load a single shard of embeddings."""
#         if not shard_path.exists():
#             return {}
#         
#         try:
#             with np.load(shard_path, allow_pickle=True) as npz:
#                 # Return as numpy arrays initially to avoid PyTorch shared memory issues
#                 return {word: emb for word, emb in npz.items()}
#         except Exception as e:
#             print(f"Error loading shard {shard_path}: {e}")
#             return {}

#     def _convert_to_torch(self, numpy_dict: Dict[str, np.ndarray]) -> None:
#         """Convert numpy arrays to torch tensors and add to cache."""
#         for word, emb in numpy_dict.items():
#             self.embedding_cache[word] = torch.from_numpy(emb)

#     def load(self):
#         """Load all cached embeddings from shards."""
#         shard_paths = sorted(self.cache_dir.glob("embedding_cache_shard_*.npz"))
#         
#         if not shard_paths:
#             old_cache = self.cache_dir / "embedding_cache.npz"
#             if old_cache.exists():
#                 self.logger.info("Found old-style cache file, loading it")
#                 with np.load(old_cache, allow_pickle=True) as cache_data:
#                     # Load as numpy first
#                     numpy_cache = {
#                         word: emb for word, emb in 
#                         tqdm(cache_data.items(), desc="Loading old cache")
#                     }
#                 # Convert to torch tensors
#                 for word, emb in tqdm(numpy_cache.items(), desc="Converting to torch tensors"):
#                     self.embedding_cache[word] = torch.from_numpy(emb)
#                 
#                 self.logger.info(f"Loaded {len(self.embedding_cache)} items from old cache")
#                 # Save in new format and optionally remove old cache
#                 self.save()
#                 if self.config.get('remove_old_cache', False):
#                     self.logger.info("Removing old cache file")
#                     old_cache.unlink()
#                 return
#             else:
#                 self.logger.info("No cache files found")
#                 return

#         self.logger.info(f"Loading embeddings from {len(shard_paths)} shards")

#         # Process shards sequentially if only one worker
#         if self.num_workers == 1:
#             for shard_path in tqdm(shard_paths, desc="Loading embedding shards"):
#                 numpy_data = self._load_shard(shard_path)
#                 self._convert_to_torch(numpy_data)
#                 gc.collect()  # Help manage memory
#         else:
#             # Process shards in parallel but convert to torch tensors sequentially
#             with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
#                 for numpy_data in tqdm(
#                     executor.map(self._load_shard, shard_paths),
#                     total=len(shard_paths),
#                     desc="Loading embedding shards"
#                 ):
#                     self._convert_to_torch(numpy_data)
#                     gc.collect()  # Help manage memory
#             
#         self._current_shard = len(shard_paths)
#         self.logger.info(f"Successfully loaded {len(self.embedding_cache)} embeddings")

#     def save(self):
#         """Save all cached embeddings to shards."""
#         if not self.embedding_cache:
#             self.logger.info("No embeddings to save")
#             return

#         # Save any remaining items in the current shard
#         if self._items_in_current_shard > 0:
#             self.logger.info("maybe saving fr")
#             self._save_current_shard()

#         self.logger.info(f"Saved {len(self.embedding_cache)} embeddings across {self._current_shard + 1} shards")

#     def __len__(self):
#         """Enable len() operator for cache."""
#         return len(self.embedding_cache)
