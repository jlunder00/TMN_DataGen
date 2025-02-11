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

    def parse_batch(self, sentence_groups: List[List[str]]) -> List[List[DependencyTree]]:
        self.logger.debug("Begin processing with multi parser")


        parser_results = {}

        base_parser = self.config.parser.feature_sources['tree_structure']
        
        # First get base parser results
        try:
            base_results = self.parsers[base_parser].parse_batch(sentence_groups)
            if not base_results:
                return []
        except Exception as e:
            self.logger.error(f"Base parser {base_parser} failed: {e}")
            return []

        # Then get results from other parsers
        for name, parser in self.parsers.items():
            if name == base_parser:
                continue
                
            try:
                parser_results[name] = parser.parse_batch(sentence_groups)
            except Exception as e:
                self.logger.error(f"Parser {name} failed: {e}")
                return []

        # First do preprocessing once
        # processed_sentence_groups = []
        final_groups = [[None] for sentences in sentence_groups]
        valid_group_indices = []
        valid_inner_indices = {i:[] for i in range(len(sentence_groups))}
        for i, sentence_group in enumerate(sentence_groups):
            # processed_sentences = []
            valid = []
            for j, sentence in enumerate(sentence_group):
                tokens = self.preprocess_and_tokenize(sentence)
                if not tokens:
                    self.logger.debug(f"Skipping sentence ({i}, {j}) due to no tokens after preprocessing: {sentence}")
                    valid.append(None)
                    continue
                if len(tokens) < 3 or len(tokens) > 15:
                    self.logger.debug(f"Skipping sentence ({i}, {j}) due too too many or too few tokens after preprocessing: {sentence}")
                    valid.append(None)
                    continue
                # processed_text = ' '.join(tokens)
                # processed_sentences.append(processed_text)
                valid.append(j)
                # self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")
            # processed_sentence_groups.append(processed_sentences)
            if len(valid) > 0:
                valid_group_indices.append(i)
            else:
                valid_group_indices.append(None)

            valid_inner_indices[i] = valid


            

        if not valid_group_indices:
            self.logger.warning("No valid sentences after preprocessing")
            return final_groups

        # Get parses from all enabled parsers
        parser_results = {}
        for name, parser in self.parsers.items():
            try:
                results = parser.parse_batch(sentence_groups)
                # Initialize with None for any skipped sentences
                this_results = [[None] for sentences in sentence_groups]
                for orig_idx_group in valid_group_indices:
                    if orig_idx_group is None:
                        tree_group = [None]
                    if orig_idx_group < len(results):
                        tree_group = results[orig_idx_group]
                        # Filter by node count
                        for orig_idx in valid_inner_indices[orig_idx_group]:
                            if orig_idx is None:
                                tree = None
                            else:
                                tree = None
                                if orig_idx < len(tree_group):
                                    tree = tree_group[orig_idx]
                                if tree is None or (tree is not None and not self._is_valid_tree(tree)):
                                    tree = None 
                            tree_group[orig_idx] = tree
                    else:
                        tree_group = [None]
                    this_results[orig_idx_group] = tree_group

                parser_results[name] = this_results
            except Exception as e:
                self.logger.error(f"Parser {name} failed: {e}")
                return final_groups
        # Combine results into final trees
        # combined_trees = []
        base_parser = self.config.parser.feature_sources['tree_structure']


        for i, final_group_trees in enumerate(final_groups):
            base_parser_result_group = parser_results[base_parser][i]
            for j, base_parser_result in enumerate(base_parser_result_group):
                try:
                    # Skip if base parser didn't produce a tree
                    if base_parser_result is None:
                        continue
                        
                    # Skip if any parser failed for this sentence
                    if any(results[i][j] is None for results in parser_results.values()):
                        continue
                        
                    base_tree = base_parser_result
                    base_tree.config = self.config
                    
                    # Enhance with features from other parsers
                    self._enhance_tree(
                        base_tree,
                        {name: results[i][j] for name, results in parser_results.items()}
                    )
                    if not self._is_valid_contents(base_tree):
                        continue
                    
                    final_group_trees[j] = base_tree
                    
                except Exception as e:
                    self.logger.error(f"Failed to combine features for sentence ({i}, {j}), {sentence_groups[i][j]}: {e}")
                    continue
            final_groups[i] = final_group_trees
        # return combined_trees
        return final_groups

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
    
    def parse_single(self, sentence_group: List[str]) -> List[DependencyTree]:
        return self.parse_batch([sentence_group])[0]

    def _get_parser_class(self, parser_name: str):
        """Get parser class based on name"""
        parser_map = {
            'diaparser': DiaParserTreeParser,
            'spacy': SpacyTreeParser
        }
        if parser_name not in parser_map:
            raise ValueError(f"Unknown parser type: {parser_name}")
        return parser_map[parser_name]
