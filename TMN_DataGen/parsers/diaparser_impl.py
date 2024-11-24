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
        """Process diaparser prediction output"""
        sentence = dataset.sentences[0]  # CoNLLSentence object
        
        # Access the values directly from the CoNLLSentence object
        # The sentence.values contains tuples for each field in order:
        # [id, form, lemma, upos, xpos, feats, head, deprel, deps, misc]
        
        # Debug log raw parser output
        if self.verbose:
            logger.info("\nParser raw output:")
            for i, field in enumerate(sentence.values):
                logger.info(f"Field {i}: {field}")
        
        # Get the form (words) from values[1]
        words_tuple = sentence.values[1]
        if isinstance(words_tuple, tuple) and len(words_tuple) == 1:
            words = words_tuple[0].split()
        else:
            words = words_tuple
        
        # Get the head indices from values[6]
        heads_list = sentence.values[6]
        if isinstance(heads_list, list):
            heads = heads_list
        else:
            heads = [heads_list]
            
        # Get the dependency relations from values[7]
        rels_list = sentence.values[7]
        if isinstance(rels_list, list):
            rels = rels_list
        else:
            rels = [rels_list]
        
        if self.verbose:
            logger.info("\nProcessed parser fields:")
            logger.info(f"Words: {words}")
            logger.info(f"Head indices: {heads}")
            logger.info(f"Relations: {rels}")
            
        return words, heads, rels

    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        trees = []
        for sentence in sentences:
            # Get prediction dataset
            dataset = self.model.predict([sentence])
            # Process the prediction output
            words, heads, rels = self._process_prediction(dataset)
            
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
                if head_idx == 0:  # Root node
                    root = nodes[idx]
                else:
                    parent = nodes[head_idx - 1]  # diaparser uses 1-based indices
                    parent.add_child(nodes[idx], dep_label)
            
            trees.append(DependencyTree(root))
        
        return trees
    
    def parse_single(self, sentence: str) -> DependencyTree:
        return self.parse_batch([sentence])[0]

    def _convert_to_tree(self, sentence: str, parse_result: Any) -> DependencyTree:
        """Convert parser output to tree structure"""
        # Debug log inputs
        if self.verbose:
            logger.info("\nConverting to tree:")
            logger.info(f"Words: {parse_result.words}")
            logger.info(f"Lemmas: {parse_result.lemmas}")
            logger.info(f"POS tags: {parse_result.pos_tags}")

        # Create nodes
        nodes = [
            Node(
                word=word,
                lemma=lemma,
                pos_tag=pos,
                idx=idx,
                features={'original_text': word}
            )
            for idx, (word, lemma, pos) 
            in enumerate(zip(parse_result.words, 
                             parse_result.lemmas,
                             parse_result.pos_tags))
        ]
        
        # Connect nodes
        root = None

        if self.verbose:
            logger.info("\nConnecting nodes:")
            logger.info(f"Head indices: {parse_result.head_indices}")
            logger.info(f"Dep labels: {parse_result.dep_labels}")

        for idx, (head_idx, dep_label) in enumerate(zip(parse_result.head_indices,
                                                       parse_result.dep_labels)):
            if head_idx == 0:  # Root node
                root = nodes[idx]
                if self.verbose:
                    logger.info(f"Found root: {root.word}")
            else:
                parent = nodes[head_idx - 1]  # diaparser uses 1-based indices
                parent.add_child(nodes[idx], dep_label)
                if self.verbose:
                    logger.info(f"Added {nodes[idx].word} as child of {parent.word} with label {dep_label}")

        tree = DependencyTree(root)
    
        # Debug final tree
        if self.verbose:
            logger.info("\nFinal tree structure:")
            for node in tree.root.get_subtree_nodes():
                logger.info(f"Node: {node.word}")
                if node.parent:
                    logger.info(f"  Parent: {node.parent.word}")
                logger.info(f"  Children: {[child[0].word for child in node.children]}")
                
        return tree


    def _create_node_features(self, node: Node) -> np.ndarray:
        from ..utils.feature_utils import FeatureExtractor
        extractor = FeatureExtractor(self.config)
        features = extractor.create_node_features(
            node, 
            self.config.get('feature_extraction', {})
        )
        return features.numpy()
    
    def _create_edge_features(self, dependency_type: str) -> np.ndarray:
        from ..utils.feature_utils import FeatureExtractor
        extractor = FeatureExtractor(self.config)
        features = extractor.create_edge_features(dependency_type)
        return features.numpy()
