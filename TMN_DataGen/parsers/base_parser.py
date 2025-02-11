# TMN_DataGen/TMN_DataGen/parsers/base_parser.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..tree.dependency_tree import DependencyTree
from ..utils.viz_utils import print_tree_text
from ..utils.logging_config import setup_logger
from ..utils.text_preprocessing import BasePreprocessor
from ..utils.tokenizers import RegexTokenizer, StanzaTokenizer
from omegaconf import DictConfig
import torch
from tqdm import tqdm

class BaseTreeParser(ABC):
    _instances: Dict[str, 'BaseTreeParser'] = {}
    
    def __new__(cls, config=None, pkg_config=None, logger=None):
        if cls not in cls._instances:
            cls._instances[cls.__name__] = super(BaseTreeParser, cls).__new__(cls)
        return cls._instances[cls.__name__]
    
    @abstractmethod
    def __init__(self, config: Optional[DictConfig] = None, pkg_config=None, logger=None):
        if not hasattr(self, 'initialized'):
            self.config = config or {}
            self.verbose = self.config.get('verbose', 'normal') 
            self.batch_size = self.config.get('batch_size', 32)
            
            # Set up logger
            self.logger = logger or setup_logger(
                    self.__class__.__name__,
                    self.config.get('verbose', 'normal')
                    )

            self.preprocessor = BasePreprocessor(self.config)

            # Initialize Tokenizer
            if self.config.preprocessing.tokenizer == "stanza":
                self.tokenizer = StanzaTokenizer(self.config)
            else:
                self.tokenizer = RegexTokenizer(self.config)
            
            self.initialized = True
    
    @abstractmethod
    def parse_batch(self, sentence_groups: List[List[str]]) -> List[List[DependencyTree]]:
        """Parse a batch of sentence groups into dependency trees groups"""
        pass
    
    @abstractmethod
    def parse_single(self, sentence_group: List[str]) -> List[DependencyTree]:
        """Parse a single sentence group into a dependency tree group"""
        pass

    def preprocess_and_tokenize(self, text: str) -> List[str]:
        """Preprocess text and tokenize into words"""
        clean_text = self.preprocessor.preprocess(text)
        tokens = self.tokenizer.tokenize(clean_text)
        return tokens
    
    def parse_all(self, sentence_groups: List[List[str]], show_progress: bool = True) -> List[List[DependencyTree]]:
        """Parse all sentences with batching and progress bar."""
        if not isinstance(sentence_groups, list):
            raise TypeError("sentence_groups must be a list of lists of strings")

        tree_groups = []
        total_sentence_groups = len(sentence_groups)
        
        if self.verbose == 'normal' or self.verbose == 'debug':
            self.logger.info(f"Processing {total_sentence_groups} sentence_groups total...")
        
        # Create batches
        for i in range(0, total_sentence_groups, self.batch_size):
            batch = sentence_groups[i:min(i + self.batch_size, total_sentence_groups)]
            if show_progress and self.verbose == 'normal' or self.verbose == 'debug':
                self.logger.info(f"Processing batch {i//self.batch_size + 1}...")
            
            batch_trees = self.parse_batch(batch)
            
            if self.verbose == 'debug':
                for group, group_trees in zip(batch, batch_trees):
                    self.logger.debug("\n" + "="*80)
                    for sent, tree in zip(group, group_trees):
                        self.logger.debug(f"Processed sentence: {sent}")
                        self.logger.debug("\nTree structure:")
                        self.logger.debug(print_tree_text(tree, self.config))
                    self.logger.debug("="*80)
            
            tree_groups.extend(batch_trees)
        
        if show_progress and self.verbose == 'normal' or self.verbose == 'debug':
            self.logger.info("Done!")
        
        return tree_groups

