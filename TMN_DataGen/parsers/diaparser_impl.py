#diaparser_impl.py
from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
from diaparser.parsers import Parser
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import DictConfig
import numpy as np

class DiaParserTreeParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None, pkg_config=None, logger=None):
        super().__init__(config, pkg_config, logger)
        if not hasattr(self, 'model'):
            self.verbose = self.config.get('verbose', 'normal')
            model_name = self.config.get('model_name', 'en_ewt.electra-base')
            self.model = Parser.load(model_name)
    
    def _process_prediction(self, dataset) -> Dict[str, List]:
        """
        Process diaparser output into aligned lists of token information.
        DiaParser provides output in CoNLL-X format with these indices:
        1: FORM - Word form/token
        2: LEMMA - Lemma
        3: UPOS - Universal POS tag
        6: HEAD - Head token id
        7: DEPREL - Dependency relation

        Returns:
            Dict with keys: words, lemmas, pos_tags, heads, rels
            All lists are aligned by token position
        """
        sentence = dataset.sentences[0]
        
        self.logger.debug(f"Processing CoNLL format sentence:")
        self.logger.debug(f"Raw values: {sentence.values}")
        
        def ensure_list(val) -> List[str]:
            """Convert various input formats to list"""
            if isinstance(val, str):
                return val.split()
            # elif isinstance(val, tuple):
            #     # DiaParser sometimes returns tuples
            #     return ensure_list(val[0])
            return list(val)

        try:
            token_data = {
                'words': ensure_list(sentence.values[1]),
                'lemmas': ensure_list(sentence.values[2]),
                'pos_tags': ensure_list(sentence.values[3]),
                'heads': [int(h) for h in ensure_list(sentence.values[6])],
                'rels': ensure_list(sentence.values[7])
            }

            # Verify all lists have same length
            list_lens = [len(lst) for lst in token_data.values()]
            if len(set(list_lens)) != 1:
                raise ValueError(
                    f"Inconsistent token list lengths: {list_lens}"
                )
            
            self.logger.debug("Processed token data:")
            for key, value in token_data.items():
                self.logger.debug(f"{key}: {value}")

            return token_data
            
        except Exception as e:
            self.logger.error(f"Error processing parser output: {e}")
            self.logger.debug(f"Values: {sentence.values}")
            raise
    
    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        trees = []
        
        self.logger.debug(f"Parsing batch of {len(sentences)} sentences")
        
        for sentence in sentences:
            
            # Preprocess and tokenize first
            tokens = self.preprocess_and_tokenize(sentence)

            self.logger.debug(f"\nParsing sentence: {sentence}")
            
            # Get DiaParser output
            dataset = self.model.predict([tokens])
            token_data = self._process_prediction(dataset)
            
            # Step 1: Create all nodes
            nodes = []
            for i in range(len(token_data['words'])):
                node = Node(
                    word=token_data['words'][i],
                    lemma=token_data['lemmas'][i],
                    pos_tag=token_data['pos_tags'][i],
                    idx=i,
                    features={
                        'original_text': token_data['words'][i]
                    }
                )
                nodes.append(node)
                
                self.logger.debug(f"Created node {i}: {node}")

            # Step 2: Connect nodes using head indices
            root = None
            for i, (node, head_idx, rel) in enumerate(zip(nodes, 
                                                         token_data['heads'],
                                                         token_data['rels'])):
                if head_idx == 0:  # Root node
                    root = node
                    self.logger.debug(f"Found root node: {node}")
                else:
                    # Head indices are 1-based in CoNLL format
                    parent = nodes[head_idx - 1]
                    parent.add_child(node, rel)
                    self.logger.debug(f"Connected node {node} to parent {parent}")

            # Step 3: Verify we found a root and built valid tree
            if root is None:
                raise ValueError(f"No root node found in parse: {sentence}")

            tree = DependencyTree(root, config=self.config)
            
            # Verify all nodes are reachable and structure is valid
            tree_nodes = tree.root.get_subtree_nodes()
            if len(tree_nodes) != len(nodes):
                raise ValueError(
                    f"Tree structure incomplete: only {len(tree_nodes)} of {len(nodes)} "
                    f"nodes reachable from root"
                )
                
            # Verify tree structure is valid
            if not root.verify_tree_structure():
                raise ValueError(
                    f"Invalid tree structure detected for sentence: {sentence}"
                )

            if self.verbose in ('normal', 'debug'):
                from ..utils.viz_utils import print_tree_text
                self.logger.info("\nDiapaarser parsed tree structure:")
                self.logger.info(print_tree_text(tree, self.config))

            trees.append(tree)
            
        return trees
    
    def parse_single(self, sentence: str) -> DependencyTree:
        """Parse a single sentence into a dependency tree"""
        return self.parse_batch([sentence])[0]

