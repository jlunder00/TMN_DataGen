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
        valid_indices = []
        for i, sentence in enumerate(sentences):
            tokens = self.preprocess_and_tokenize(sentence)
            if not tokens:
                self.logger.debug(f"Skipping sentence {i} due to no tokens after preprocessing: {sentence}")
                continue
            if len(tokens) < 3 or len(tokens) > 15:
                self.logger.debug(f"Skipping sentence {i} due too too many or too few tokens after preprocessing: {sentence}")
            processed_text = ' '.join(tokens)
            processed_sentences.append(processed_text)
            valid_indices.append(i)
            self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")

        if not processed_sentences:
            self.logger.warning("No valid sentences after preprocessing")
            return []

        # Get parses from all enabled parsers
        # parser_results = {}
        # for name, parser in self.parsers.items():
        #     try:
        #         results = parser.parse_batch(processed_sentences)
        #         if len(results) != len(processed_sentences):
        #             self.logger.error(f"Parser {name} returned misaligned results")
        #             return []
        #         parser_results[name] = results
        #     except Exception as e:
        #         self.logger.error(f"Parser {name} failed: {e}")
        #         return []
        
        # Get parses from all enabled parsers
        parser_results = {}
        for name, parser in self.parsers.items():
            try:
                results = parser.parse_batch(processed_sentences)
                # Initialize with None for any skipped sentences
                full_results = [None] * len(sentences)
                for proc_idx, orig_idx in enumerate(valid_indices):
                    if proc_idx < len(results):
                        tree = results[proc_idx]
                        # Filter by node count
                        if tree is not None and not self._is_valid_tree(tree):
                            tree = None
                        full_results[orig_idx] = tree
                parser_results[name] = full_results
            except Exception as e:
                self.logger.error(f"Parser {name} failed: {e}")
                return [None] * len(sentences)
        # Combine results into final trees
        # combined_trees = []
        base_parser = self.config.parser.feature_sources['tree_structure']

        final_trees = [None] * len(sentences)

        # for proc_idx, orig_idx in enumerate(valid_indices):
        #     try:
        #         base_tree = parser_results[base_parser][proc_idx]
        #         base_tree.config = self.config
        #         self._enhance_tree(base_tree, {name: results[proc_idx] for name, results in parser_results.items()})

        #         final_trees[orig_idx] = base_tree

        #         self.logger.debug(f"\nProcessed sentence {proc_idx+1}/{len(processed_sentences)}")
        #         self.logger.debug(f"Combined features from {len(parser_results)} parsers")

        #     except Exception as e:
        #         self.logger.error(f"Failed to combine featurees for sentence {proc_idx}: {e}")
        #         continue

        # combined_trees = [t for t in final_trees if t is not None]
        for i in range(len(sentences)):
            try:
                # Skip if base parser didn't produce a tree
                if not parser_results[base_parser][i]:
                    continue
                    
                # Skip if any parser failed for this sentence
                if any(results[i] is None for results in parser_results.values()):
                    continue
                    
                base_tree = parser_results[base_parser][i]
                base_tree.config = self.config
                
                # Enhance with features from other parsers
                self._enhance_tree(
                    base_tree,
                    {name: results[i] for name, results in parser_results.items()}
                )
                if not self._is_valid_contents(base_tree):
                    continue
                
                final_trees[i] = base_tree
                
            except Exception as e:
                self.logger.error(f"Failed to combine features for sentence {i}: {e}")
                continue

        # if not combined_trees:
        #     self.logger.warning("No valid trees produced")
        # elif len(combined_trees) != len(sentences):
        #     self.logger.warning(f"Only {len(combined_trees)}/{len(sentences)} sentences successfully parsed")
        

        # for i in range(len(processed_sentences)):
        #     # Start with the base tree structure from preferred parser
        #     base_parser = self.config.parser.feature_sources['tree_structure']
        #     if base_parser not in parser_results:
        #         raise ValueError(f"Base parser {base_parser} not available")
        #     
        #     base_tree = parser_results[base_parser][i]
        #     base_tree.config = self.config  # Propagate config
        #     
        #     # Enhance with features from other parsers
        #     self._enhance_tree(
        #         base_tree,
        #         {name: results[i] for name, results in parser_results.items()}
        #     )
        #     
        #     combined_trees.append(base_tree)

        #     self.logger.debug(f"\nProcessed sentence {i+1}/{len(processed_sentences)}")
        #     self.logger.debug(f"Combined features from {len(parser_results)} parsers")

        
        # return combined_trees
        return final_trees

    def _is_valid_contents(self, tree):
        for node in tree.root.get_subtree_nodes():
                    # Check for missing POS tags
            if node.pos_tag == '_' or not node.pos_tag:
                self.logger.debug(f"Invalid POS tag '_' found in tree")
                return False

            if node != tree.root and (not node.dependency_to_parent or node.dependency_to_parent == '_'):
                self.logger.debug(f"Invalid DEP tag found in tree")
                return False
        return True
                


    def _is_valid_tree(self, tree: DependencyTree) -> bool:
        """Check if tree meets validity criteria"""
        for node in tree.root.get_subtree_nodes():
            # Check node count
            node_count = len(tree.root.get_subtree_nodes())
            if node_count < 3 or node_count > 15:
                self.logger.debug(f"Tree has {node_count} nodes - outside valid range [3,15]")
                return False
                
        return True

    def _validate_feature_sources(self):
        """Validate that assigned feature sources are capable"""
        for feature, parser in self.config.parser.feature_sources.items():
            if parser not in self.capabilities:
                raise ValueError(f"Unknown parser: {parser}")
            if feature not in self.capabilities[parser]['capabilities']:
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
            if feature not in self.capabilities[parser_name]['capabilities']:
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
