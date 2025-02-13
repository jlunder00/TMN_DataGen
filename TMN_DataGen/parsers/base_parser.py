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

# At the top of your module (or in a dedicated helper module)
from concurrent.futures import ProcessPoolExecutor

# Global variables for the worker processes.
_worker_preprocessor = None
_worker_tokenizer = None

def _init_worker(config):
    """Initializer for each worker process.
    This sets up the preprocessor and tokenizer using the given config.
    """
    global _worker_preprocessor, _worker_tokenizer
    from TMN_DataGen.utils.text_preprocessing import BasePreprocessor
    from TMN_DataGen.utils.tokenizers import RegexTokenizer, StanzaTokenizer
    _worker_preprocessor = BasePreprocessor(config)
    if config.preprocessing.tokenizer == "stanza":
        _worker_tokenizer = StanzaTokenizer(config)
    else:
        _worker_tokenizer = RegexTokenizer(config)

def _parallel_preprocess_tokenize(text: str) -> List[str]:
    """Worker function: preprocess and tokenize a single text."""
    global _worker_preprocessor, _worker_tokenizer
    clean_text = _worker_preprocessor.preprocess(text)
    tokens = _worker_tokenizer.tokenize(clean_text)
    return tokens

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
    def parse_batch(self, sentence_groups: List[List[str]], num_workers: int = 1) -> List[List[DependencyTree]]:
        """Parse a batch of sentence groups into dependency trees groups"""
        pass
    
    @abstractmethod
    def parse_single(self, sentence_group: List[str], num_workers: int = 1) -> List[DependencyTree]:
        """Parse a single sentence group into a dependency tree group"""
        pass

    def preprocess_and_tokenize(self, text: str) -> List[str]:
        """Preprocess text and tokenize into words"""
        clean_text = self.preprocessor.preprocess(text)
        tokens = self.tokenizer.tokenize(clean_text)
        return tokens

    def parallel_preprocess_tokenize(self, texts: List[str], num_workers: int = None) -> List[List[str]]:
        """
        Process many texts in parallel by preprocessing and tokenizing each.
        
        Args:
            texts (List[str]): A flat list of texts to be processed.
            num_workers (int, optional): The maximum number of worker processes.
                Defaults to the number of processors on the machine.
                
        Returns:
            List[List[str]]: A list where each element is the list of tokens for the corresponding text.
        """
        self.logger.info("Parallel preprocessing/tokenization of {} texts".format(len(texts)))
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(self.config,)) as executor:
            token_lists = list(executor.map(_parallel_preprocess_tokenize, texts))
        return token_lists
    
    def parse_all(self, sentence_groups: List[List[str]], show_progress: bool = True, num_workers: int = 1) -> List[List[DependencyTree]]:
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
            
            batch_trees = self.parse_batch(batch, num_workers)
            
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

