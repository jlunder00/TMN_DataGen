# TMN_DataGen/TMN_DataGen/parsers/multi_parser.py
from typing import List, Dict, Any, Optional
from omegaconf import DictConfig
from .base_parser import BaseTreeParser
from .diaparser_impl import DiaParserTreeParser
from .spacy_impl import SpacyTreeParser
from ..tree.dependency_tree import DependencyTree
from ..tree.node import Node
import yaml
from pathlib import Path

class MultiParser(BaseTreeParser):
    def __init__(self, config=None, pkg_config=None, logger=None):
        # Load package capabilities config
        self.capabilities = pkg_config['parsers'] if pkg_config else {}

        # Initialize base class
        super().__init__(config, pkg_config, logger)

        # Validate user's feature source config against capabilities
        self._validate_feature_sources()

        # Initialize parsers
        self.parsers = {}
        for parser_name, parser_config in self.config.parser.parsers.items():
            if parser_config.get('enabled', False):
                parser_class = self._get_parser_class(parser_name)
                self.parsers[parser_name] = parser_class(self.config)
        
        self.initialized = True

    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        self.logger.debug("Begin processing with multi parser")

        # First do preprocessing once
        processed_sentences = []
        for sentence in sentences:
            tokens = self.preprocess_and_tokenize(sentence)
            processed_text = ' '.join(tokens)
            processed_sentences.append(processed_text)
            
            self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")

        # Get parses from all enabled parsers
        parser_results = {}
        for name, parser in self.parsers.items():
            parser_results[name] = parser.parse_batch(processed_sentences)
        
        # Combine results into final trees
        combined_trees = []
        for i in range(len(processed_sentences)):
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

            self.logger.debug(f"\nProcessed sentence {i+1}/{len(processed_sentences)}")
            self.logger.debug(f"Combined features from {len(parser_results)} parsers")

        
        return combined_trees

    def _validate_feature_sources(self):
        """Validate that assigned feature sources are capable"""
        for feature, parser in self.config.parser.feature_sources.items():
            if parser not in self.capabilities['parsers']:
                raise ValueError(f"Unknown parser: {parser}")
            if feature not in self.capabilities['parsers'][parser]['capabilities']:
                raise ValueError(
                    f"Parser {parser} cannot provide feature {feature}"
                )

    def _enhance_tree(self, base_tree: DependencyTree, parser_trees: Dict[str, DependencyTree]):
        """Enhanced version that respects capabilities"""
        base_nodes = {node.idx: node for node in base_tree.root.get_subtree_nodes()}
        
        for feature, parser_name in self.config.parser.feature_sources.items():
            if parser_name not in parser_trees:
                continue

            # Skip if parser isn't capable of this feature
            if feature not in self.capabilities['parsers'][parser_name]['capabilities']:
                continue
                
            other_tree = parser_trees[parser_name]
            other_nodes = {node.idx: node for node in other_tree.root.get_subtree_nodes()}
            
            for idx, base_node in base_nodes.items():
                if idx in other_nodes:
                    self._copy_feature(feature, other_nodes[idx], base_node)

    def _copy_feature(self, feature: str, src_node: Node, dst_node: Node):
        """Copy a specific feature from src to dst node"""
        if feature == 'pos_tags':
            dst_node.pos_tag = src_node.pos_tag
        elif feature == 'lemmas':
            dst_node.lemma = src_node.lemma
        elif feature == 'dependency_labels':
            dst_node.dependency_to_parent = src_node.dependency_to_parent
        elif feature == 'morph_features':
            if 'morph_features' not in dst_node.features:
                dst_node.features['morph_features'] = {}
            dst_node.features['morph_features'].update(
                src_node.features.get('morph_features', {})
            )
    
    def parse_single(self, sentence: str) -> DependencyTree:
        return self.parse_batch([sentence])[0]

    def _get_parser_class(self, parser_name: str):
        """Get parser class based on name"""
        parser_map = {
            'diaparser': DiaParserTreeParser,
            'spacy': SpacyTreeParser
        }
        if parser_name not in parser_map:
            raise ValueError(f"Unknown parser type: {parser_name}")
        return parser_map[parser_name]
