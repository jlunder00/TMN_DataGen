#diaparser_impl.py
from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
from ..utils.logging_config import logger
from diaparser.parsers import Parser
from typing import List, Any, Optional, Tuple
from omegaconf import DictConfig
import numpy as np

class DiaParserTreeParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None):
        super().__init__(config)
        if not hasattr(self, 'model'):
            model_name = self.config.get('model_name', 'en_ewt.electra-base')
            self.model = Parser.load(model_name)
    
    def _process_prediction(self, dataset) -> Tuple[List[str], List[int], List[str]]:
        """Process diaparser prediction output into words, heads, and relations."""
        sentence = dataset.sentences[0]  # CoNLLSentence object
        
        # Access values directly - these are already lists
        words = list(sentence.values[1])  # Get all words
        heads = list(sentence.values[6])  # Get all head indices
        rels = list(sentence.values[7])   # Get all dependency relations

        if self.verbose == 'debug':
            self.logger.debug(f"Raw parser output:")
            self.logger.debug(f"Words: {words}")
            self.logger.debug(f"Heads: {heads}")
            self.logger.debug(f"Relations: {rels}")

        return words, heads, rels

    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        trees = []
        
        if self.verbose == 'debug':
            self.logger.debug(f"Parsing batch of {len(sentences)} sentences")
        elif self.verbose == 'normal':
            self.logger.info(f"Processing {len(sentences)} sentences...")
            
        for sentence in sentences:
            # Get prediction dataset
            dataset = self.model.predict([sentence])
            words, heads, rels = self._process_prediction(dataset)
            
            if self.verbose == 'debug':
                self.logger.debug(f"Creating nodes for sentence: {sentence}")
            
            # Create nodes
            nodes = [
                Node(
                    word=word,
                    lemma=word.lower(),  # Simple lemmatization for now
                    pos_tag="",  # We could add POS tags if needed
                    idx=idx,
                    features={'original_text': word}
                )
                for idx, word in enumerate(words)
            ]
            
            # Connect nodes
            root = None
            for idx, (head_idx, dep_label) in enumerate(zip(heads, rels)):
                if self.verbose == 'debug':
                    self.logger.debug(f"Processing node {idx}: word={words[idx]}, "
                                    f"head={head_idx}, relation={dep_label}")
                
                if head_idx == 0:  # Root node
                    root = nodes[idx]
                    if self.verbose == 'debug':
                        self.logger.debug(f"Found root node: {root.word}")
                else:
                    parent = nodes[head_idx - 1]  # diaparser uses 1-based indices
                    parent.add_child(nodes[idx], dep_label)
                    if self.verbose == 'debug':
                        self.logger.debug(f"Added {nodes[idx].word} as child of "
                                        f"{parent.word} with label {dep_label}")
            
            tree = DependencyTree(root, config=self.config)
            
            if self.verbose == 'normal':
                from ..utils.viz_utils import print_tree_text
                self.logger.info("\nParsed tree structure:")
                self.logger.info(print_tree_text(tree, self.config))
            
            trees.append(tree)
        
        return trees
    
    def parse_single(self, sentence: str) -> DependencyTree:
        return self.parse_batch([sentence])[0]
