# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# TMN_DataGen/TMN_DataGen/parsers/base_parser.py
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..tree.dependency_tree import DependencyTree
from ..utils.viz_utils import print_tree_text
from ..utils.logging_config import setup_logger
from ..utils.text_preprocessing import BasePreprocessor
from ..utils.tokenizers import RegexTokenizer, StanzaTokenizer, VocabTokenizer
from omegaconf import DictConfig
# from gensim.models import KeyedVectors
import torch
from tqdm import tqdm
from itertools import repeat
# from english_words import get_english_words_set


# At the top of your module (or in a dedicated helper module)
from concurrent.futures import ProcessPoolExecutor

# Global variables for the worker processes.
_worker_preprocessor = None
_worker_tokenizer = None

def _init_worker(config, vocabs, logger, preprocess_only=False):
    """Initializer for each worker process.
    This sets up the preprocessor and tokenizer using the given config.
    """
    global _worker_preprocessor, _worker_tokenizer
    from TMN_DataGen.utils.text_preprocessing import BasePreprocessor
    from TMN_DataGen.utils.tokenizers import RegexTokenizer, StanzaTokenizer
    _worker_preprocessor = BasePreprocessor(config)
    if config.preprocessing.tokenizer == "stanza" and not preprocess_only:
        _worker_tokenizer = StanzaTokenizer(config, vocabs, logger)
    else:
        _worker_tokenizer = RegexTokenizer(config, vocabs, logger)

def _parallel_preprocess_tokenize(text: str) -> List[str]:
    """Worker function: preprocess and tokenize a single text."""
    global _worker_preprocessor, _worker_tokenizer
    clean_text = _worker_preprocessor.preprocess(text)
    tokens = _worker_tokenizer.tokenize(clean_text)
    return tokens

def _parallel_preprocess_only(text: str) -> str:
    global _worker_preprocessor
    return _worker_preprocessor.preprocess(text)

class BaseTreeParser(ABC):
    _instances: Dict[str, 'BaseTreeParser'] = {}
    
    def __new__(cls, config=None, pkg_config=None, vocabs= [set({})], logger=None):
        if cls not in cls._instances:
            cls._instances[cls.__name__] = super(BaseTreeParser, cls).__new__(cls)
        return cls._instances[cls.__name__]
    
    @abstractmethod
    def __init__(self, config: Optional[DictConfig] = None, pkg_config=None, vocabs = [set({})], logger=None):
        if not hasattr(self, 'initialized'):
            self.config = config or {}
            self.verbose = self.config.get('verbose', 'normal') 
            self.batch_size = self.config.get('parser', {}).get('batch_size', 32)
            self.spacy_batch_size = self.config.get('parser', {}).get('spacy_batch_size', 10000)
            self.diaparser_batch_size = self.config.get('parser', {}).get('diaparser_batch_size', 5000)
            
            # Set up logger
            self.logger = logger or setup_logger(
                    self.__class__.__name__,
                    self.config.get('verbose', 'normal')
                    )
            # self.vocabs = []

            # vocab_model =  KeyedVectors.load_word2vec_format(self.config.preprocessing.get('vocab_model_path', '/home/jlunder/research/data/word2vec_model/GoogleNews-vectors-negative300.bin'), binary=True, limit=self.config.preprocessing.get('vocab_limit', 500000)) #take only top n common words 
            # self.vocabs.append(vocab_model.index_to_key)

            # all_words = set()
            # all_words_lower = get_english_words_set(['web2', 'gcide'], lower=True)
            # all_words = all_words.union(all_words_lower)
            # all_words_standard = get_english_words_set(['web2', 'gcide'])
            # all_words = all_words.union(all_words_standard)
            # all_words_alpha_standard = get_english_words_set(['web2', 'gcide'], alpha=True)
            # all_words = all_words.union(all_words_alpha_standard)
            # all_words_alpha_lower = get_english_words_set(['web2', 'gcide'], alpha=True, lower=True)
            # all_words = all_words.union(all_words_alpha_lower)
            # self.vocabs.append(all_words)
            self.vocabs = vocabs

            self.preprocessor = BasePreprocessor(self.config)

            # Initialize Tokenizer
            if self.config.preprocessing.tokenizer == "stanza":
                self.tokenizer = StanzaTokenizer(self.config, self.vocabs, self.logger)
            elif self.config.preprocessing.tokenizer == 'vocab':
                self.tokenizer = VocabTokenizer(self.config, self.vocabs, self.logger)
            else:
                self.tokenizer = RegexTokenizer(self.config, self.vocabs, self.logger)

            
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
        if num_workers < 2:
            token_lists = [self.preprocess_and_tokenize(text) for text in texts]
        elif self.config.preprocessing.tokenizer == "stanza":
            with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(self.config, self.vocabs, self.logger, True)) as executor:
                clean_texts = list(executor.map(_parallel_preprocess_only, texts))
            # batch_size = 25
            # batches = [clean_texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            batches = [clean_texts]
            token_lists = []
            for batch in batches:
                token_lists.extend(self.tokenizer.tokenize_parallel_stanza(batch))

        else:
            with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(self.config, self.vocabs, self.logger)) as executor:
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
            batch_time_start = time.time()
            batch = sentence_groups[i:min(i + self.batch_size, total_sentence_groups)]
            if show_progress and self.verbose == 'normal' or self.verbose == 'debug':
                self.logger.info(f"Processing batch {i//self.batch_size + 1} with {self.batch_size} group pairs")
            
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
            batch_time_end = time.time()
            self.logger.info(f"Batch took: {batch_time_end-batch_time_start} seconds")
        
        if show_progress and self.verbose == 'normal' or self.verbose == 'debug':
            self.logger.info("Done!")
        
        return tree_groups

