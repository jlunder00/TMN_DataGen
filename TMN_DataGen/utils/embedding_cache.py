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
                 config: Optional[Dict] = None):
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
        self.logger = setup_logger(
            self.__class__.__name__,
            self.config.get('verbose', 'normal')
        )

    def _get_shard_path(self, shard_idx: int) -> Path:
        """Get the path for a specific shard file."""
        return self.cache_dir / f"embedding_cache_shard_{shard_idx}.npz"

    def __getitem__(self, word: str) -> Optional[torch.Tensor]:
        """Enable dictionary-like access to cache."""
        if word == 'file':
            word = '\\file'
        return self.embedding_cache.get(word)

    def __setitem__(self, word: str, embedding: torch.Tensor):
        """Enable dictionary-like setting of cache values."""
        if word == 'file':
            word = '\\file'
        if word not in self.embedding_cache:
            self.embedding_cache[word] = embedding
        
        elif embedding != self.embedding_cache[word]:
            #replace it such that when you take the list of .items(), its in the new shard
            del self.embedding_cache[word]
            self.embedding_cache[word] = embedding
        else:
            return
        self._items_in_current_shard += 1
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

        shard_path = self._get_shard_path(self._current_shard)
        items_to_save = list(self.embedding_cache.items())
        start_idx = self._current_shard * self.shard_size
        end_idx = start_idx + self.shard_size
        shard_items = items_to_save[start_idx:end_idx]
        
        if shard_items:
            # Convert to numpy and save
            np_data = {word: emb.cpu().numpy() for word, emb in shard_items}
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
            self.embedding_cache[word] = torch.from_numpy(emb)

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
                    self.embedding_cache[word] = torch.from_numpy(emb)
                
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
            for shard_path in tqdm(shard_paths, desc="Loading embedding shards"):
                numpy_data = self._load_shard(shard_path)
                self._convert_to_torch(numpy_data)
                gc.collect()  # Help manage memory
        else:
            # Process shards in parallel but convert to torch tensors sequentially
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                for numpy_data in tqdm(
                    executor.map(self._load_shard, shard_paths),
                    total=len(shard_paths),
                    desc="Loading embedding shards"
                ):
                    self._convert_to_torch(numpy_data)
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

        self.logger.info(f"Saved {len(self.embedding_cache)} embeddings across {self._current_shard + 1} shards")

    def __len__(self):
        """Enable len() operator for cache."""
        return len(self.embedding_cache)
