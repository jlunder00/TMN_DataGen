# TMN_DataGen/TMN_DataGen/parsers/multi_parser.py
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig
from .base_parser import BaseTreeParser
from .diaparser_impl import DiaParserTreeParser
from .spacy_impl import SpacyTreeParser
from ..tree.dependency_tree import DependencyTree

class MultiParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None):
        self.parsers = {}
        super().__init__(config)
        
        # Initialize requested parsers
        parser_configs = self.config.get('parser', {}).get('parsers', {})
        
        if parser_configs.get('diaparser', {}).get('enabled', True):
            self.parsers['diaparser'] = DiaParserTreeParser(
                self.config
            )
        
        if parser_configs.get('spacy', {}).get('enabled', True):
            self.parsers['spacy'] = SpacyTreeParser(
                self.config
            )
        
        # Configure feature sources
        self.feature_sources = self.config.get('parser', {}).get('feature_sources', {
            'tree_structure': 'diaparser',
            'pos_tags': 'spacy',
            'morph': 'spacy',
            'lemmas': 'spacy'
        })
        
        self.initialized = True

    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        # Get parses from all enabled parsers
        parser_results = {}
        for name, parser in self.parsers.items():
            parser_results[name] = parser.parse_batch(sentences)
        
        # Combine results into final trees
        combined_trees = []
        for i in range(len(sentences)):
            # Start with the base tree structure from preferred parser
            base_parser = self.feature_sources['tree_structure']
            if base_parser not in parser_results:
                raise ValueError(f"Base parser {base_parser} not available")
            
            base_tree = parser_results[base_parser][i]
            base_tree.config = self.config  # Propagate config
            
            # Enhance with features from other parsers
            self._enhance_tree(
                base_tree,
                {name: results[i] for name, results in parser_results.items()}
            )
            
            combined_trees.append(base_tree)
        
        return combined_trees

    def _enhance_tree(self, base_tree: DependencyTree, parser_trees: Dict[str, DependencyTree]):
        base_nodes = {node.idx: node for node in base_tree.root.get_subtree_nodes()}
        
        for feature, parser_name in self.feature_sources.items():
            if parser_name not in parser_trees:
                continue
                
            other_tree = parser_trees[parser_name]
            other_nodes = {node.idx: node for node in other_tree.root.get_subtree_nodes()}
            
            for idx, base_node in base_nodes.items():
                if idx in other_nodes:
                    other_node = other_nodes[idx]
                    if feature == 'pos_tags':
                        base_node.pos_tag = other_node.pos_tag
                    elif feature == 'lemmas':
                        base_node.lemma = other_node.lemma
                    elif feature == 'morph':
                        # Ensure morph_features exists in features dict
                        if 'morph_features' not in base_node.features:
                            base_node.features['morph_features'] = {}
                        # Copy morph features from spacy
                        base_node.features['morph_features'].update(other_node.features.get('morph_features', {}))

    def _enhance_tree(self, base_tree: DependencyTree, 
                     parser_trees: Dict[str, DependencyTree]):
        """Enhance base tree with features from other parsers"""
        # Map nodes by their position in sentence
        base_nodes = {node.idx: node for node in base_tree.root.get_subtree_nodes()}
        
        # Add features from each parser based on configuration
        for feature, parser_name in self.feature_sources.items():
            if parser_name not in parser_trees:
                continue
                
            other_tree = parser_trees[parser_name]
            other_nodes = {node.idx: node for node in other_tree.root.get_subtree_nodes()}
            
            # Copy features
            for idx, base_node in base_nodes.items():
                if idx in other_nodes:
                    other_node = other_nodes[idx]
                    if feature == 'pos_tags':
                        base_node.pos_tag = other_node.pos_tag
                    elif feature == 'lemmas':
                        base_node.lemma = other_node.lemma
                    elif feature == 'morph':
                        base_node.features.update(other_node.features)
    
    def parse_single(self, sentence: str) -> DependencyTree:
        return self.parse_batch([sentence])[0]
