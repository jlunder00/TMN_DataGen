# TMN_DataGen/TMN_DataGen/parsers/base_parser.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..tree.dependency_tree import DependencyTree
from ..utils.viz_utils import print_tree_text
from ..utils.logging_config import get_logger
from omegaconf import DictConfig
import torch
from tqdm import tqdm

class BaseTreeParser(ABC):
    _instances: Dict[str, 'BaseTreeParser'] = {}
    
    def __new__(cls, config: Optional[DictConfig] = None):
        if cls not in cls._instances:
            cls._instances[cls.__name__] = super(BaseTreeParser, cls).__new__(cls)
        return cls._instances[cls.__name__]
    
    @abstractmethod
    def __init__(self, config: Optional[DictConfig] = None):
        if not hasattr(self, 'initialized'):
            self.config = config or {}
            self.batch_size = self.config.get('batch_size', 32)
            
            # Get global verbosity setting
            self.verbose = self.config.get('verbose')
            
            # Set up logger
            self.logger = get_logger(
                name=self.__class__.__name__,
                verbose=self.verbose
            )
            
            self.initialized = True
    
    @abstractmethod
    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        """Parse a batch of sentences into dependency trees"""
        pass
    
    @abstractmethod
    def parse_single(self, sentence: str) -> DependencyTree:
        """Parse a single sentence into a dependency tree"""
        pass
    
    def parse_all(self, sentences: List[str], show_progress: bool = True) -> List[DependencyTree]:
        """Parse all sentences with batching and progress bar."""
        if not isinstance(sentences, list):
            if hasattr(sentences, '__iter__'):
                sentences = list(sentences)
            else:
                raise TypeError("sentences must be a list of strings")
        
        trees = []
        total_sentences = len(sentences)
        
        if self.verbose == 'normal':
            self.logger.info(f"Processing {total_sentences} sentences total...")
        
        # Create batches
        for i in range(0, total_sentences, self.batch_size):
            batch = sentences[i:min(i + self.batch_size, total_sentences)]
            if show_progress and self.verbose == 'normal':
                self.logger.info(f"Processing batch {i//self.batch_size + 1}...")
            
            batch_trees = self.parse_batch(batch)
            
            if self.verbose == 'debug':
                for sent, tree in zip(batch, batch_trees):
                    self.logger.debug("\n" + "="*80)
                    self.logger.debug(f"Processed sentence: {sent}")
                    self.logger.debug("\nTree structure:")
                    self.logger.debug(print_tree_text(tree, self.config))
                    self.logger.debug("="*80)
            
            trees.extend(batch_trees)
        
        if show_progress and self.verbose == 'normal':
            self.logger.info("Done!")
        
        return trees

