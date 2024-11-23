#base_parser.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..tree.dependency_tree import DependencyTree
from omegaconf import DictConfig
import torch
from tqdm import tqdm

class BaseTreeParser(ABC):
    _instances: Dict[str, 'BaseTreeParser'] = {}
    
    def __new__(cls, config: Optional[DictConfig] = None):
        """Singleton pattern implementation"""
        if cls not in cls._instances:
            cls._instances[cls.__name__] = super(BaseTreeParser, cls).__new__(cls)
        return cls._instances[cls.__name__]
    
    @abstractmethod
    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize parser with configuration"""
        if not hasattr(self, 'initialized'):
            self.config = config or {}
            self.batch_size = self.config.get('batch_size', 32)
            self.initialized = True
    
    @abstractmethod
    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        """Parse a batch of sentences into dependency trees"""
        pass
    
    @abstractmethod
    def parse_single(self, sentence: str) -> DependencyTree:
        """Parse a single sentence into a dependency tree"""
        pass
    
    # def parse_all(self, sentences: List[str], show_progress: bool = True) -> List[DependencyTree]:
    #     """Parse all sentences with batching and progress bar"""
    #     if not isinstance(sentences, list):
    #         raise TypeError("sentences must be a list of strings")
    #     
    #     trees = []
    #     iterator = range(0, len(sentences), self.batch_size)
    #     if show_progress:
    #         iterator = tqdm(iterator, desc="Parsing sentences")
    #         
    #     for i in iterator:
    #         batch = sentences[i:i + self.batch_size]
    #         trees.extend(self.parse_batch(batch))
    #     return trees

    def parse_all(self, sentences: List[str], show_progress: bool = True) -> List[DependencyTree]:
        """Parse all sentences with batching and progress bar"""
        if not isinstance(sentences, list):
            sentences = list(sentences)  # Try to convert to list if possible
        
        if not sentences:  # Handle empty input
            return []
            
        if not all(isinstance(s, str) for s in sentences):
            raise TypeError("All elements must be strings")
        
        trees = []
        total_sentences = len(sentences)
        
        # Create batches
        for i in range(0, total_sentences, self.batch_size):
            batch = sentences[i:min(i + self.batch_size, total_sentences)]
            if show_progress:
                print(f"\rProcessing {i+1}/{total_sentences} sentences...", end="")
            trees.extend(self.parse_batch(batch))
        
        if show_progress:
            print("\nDone!")
    
        return trees

    # def parse_all(self, sentences: List[str], show_progress: bool = True) -> List[DependencyTree]:
    #     """Parse all sentences with batching and progress bar"""
    #     trees = []
    #     iterator = range(0, len(sentences), self.batch_size)
    #     if show_progress:
    #         iterator = tqdm(iterator, desc="Parsing sentences")
    #         
    #     for i in iterator:
    #         batch = sentences[i:i + self.batch_size]
    #         trees.extend(self.parse_batch(batch))
    #     return trees
