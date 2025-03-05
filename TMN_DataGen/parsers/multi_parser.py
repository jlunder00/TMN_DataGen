# TMN_DataGen/TMN_DataGen/parsers/multi_parser.py
import time
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

    # def parse_batch(self, sentence_groups: List[List[str]]) -> List[List[DependencyTree]]:
    #     self.logger.debug("Begin processing with multi parser")


    #     parser_results = {}

    #     base_parser = self.config.parser.feature_sources['tree_structure']
    #     
    #     max_trees = 4
    #     final_groups = [[None] for sentences in sentence_groups]
    #     valid_group_indices = []
    #     valid_inner_indices = {i:[] for i in range(len(sentence_groups))}
    #     for i, sentence_group in enumerate(sentence_groups):
    #         # processed_sentences = []
    #         token_group = sorted([self.preprocess_and_tokenize(sentence) for sentence in sentence_group], key=lambda x: len(x), reverse=True)
    #         valid = []
    #         for j, tokens in enumerate(token_group):
    #             if not tokens:
    #                 self.logger.debug(f"Skipping sentence ({i}, {j}) due to no tokens after preprocessing: {sentence_group[j]}")
    #                 valid.append(None)
    #                 continue
    #             if len(tokens) < 3 or len(tokens) > 20:
    #                 self.logger.debug(f"Skipping sentence ({i}, {j}) due too too many or too few tokens after preprocessing: {sentence_group[j]}")
    #                 valid.append(None)
    #                 continue
    #             if max_trees > 0 and j > max_trees-1:
    #                 self.logger.debug(f"Skipping sentence ({i}, {j}) due to maximum sentences per group cutoff")
    #                 valid.append(None)
    #                 continue
    #             # processed_text = ' '.join(tokens)
    #             # processed_sentences.append(processed_text)
    #             valid.append(j)
    #             # self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")
    #         # processed_sentence_groups.append(processed_sentences)
    #         if len(valid) > 0:
    #             valid_group_indices.append(i)
    #         else:
    #             valid_group_indices.append(None)

    #         valid_inner_indices[i] = valid


    #         

    #     if not valid_group_indices:
    #         self.logger.warning("No valid sentences after preprocessing")
    #         return final_groups

    #     # Get parses from all enabled parsers
    #     parser_results = {}
    #     for name, parser in self.parsers.items():
    #         try:
    #             results = parser.parse_batch(sentence_groups)
    #             # Initialize with None for any skipped sentences
    #             this_results = [[None] for sentences in sentence_groups]
    #             for orig_idx_group, orig_idx_group_validity in enumerate(valid_group_indices):
    #                 if orig_idx_group_validity is None:
    #                     tree_group = [None]
    #                 if orig_idx_group < len(results):
    #                     tree_group = results[orig_idx_group]
    #                     # Filter by node count
    #                     for orig_idx, orig_idx_validity in enumerate(valid_inner_indices[orig_idx_group]):
    #                         if orig_idx_validity is None:
    #                             tree = None
    #                         else:
    #                             tree = None
    #                             if orig_idx < len(tree_group):
    #                                 tree = tree_group[orig_idx]
    #                             if tree is None or (tree is not None and not self._is_valid_tree(tree)):
    #                                 tree = None 
    #                         tree_group[orig_idx] = tree
    #                 else:
    #                     tree_group = [None]
    #                 this_results[orig_idx_group] = tree_group

    #             parser_results[name] = this_results
    #         except Exception as e:
    #             self.logger.error(f"Parser {name} failed: {e}")
    #             return final_groups
    #     # Combine results into final trees
    #     # combined_trees = []
    #     base_parser = self.config.parser.feature_sources['tree_structure']


    #     for i, final_group_trees in enumerate(final_groups):
    #         base_parser_result_group = parser_results[base_parser][i]
    #         new_final_group_trees = []
    #         for j, base_parser_result in enumerate(base_parser_result_group):
    #             try:
    #                 # Skip if base parser didn't produce a tree
    #                 if base_parser_result is None:
    #                     continue
    #                     
    #                 # Skip if any parser failed for this sentence
    #                 if any(results[i][j] is None for results in parser_results.values()):
    #                     continue
    #                     
    #                 base_tree = base_parser_result
    #                 base_tree.config = self.config
    #                 
    #                 # Enhance with features from other parsers
    #                 self._enhance_tree(
    #                     base_tree,
    #                     {name: results[i][j] for name, results in parser_results.items()}
    #                 )
    #                 if not self._is_valid_contents(base_tree):
    #                     continue
    #                 
    #                 new_final_group_trees.append(base_tree)
    #                 
    #             except Exception as e:
    #                 self.logger.error(f"Failed to combine features for sentence ({i}, {j}), {sentence_groups[i][j]}: {e}")
    #                 continue
    #         
    #         if len(new_final_group_trees) > 0:
    #             final_groups[i] = new_final_group_trees
    #     # return combined_trees
    #     return final_groups

    def parse_batch(self, sentence_groups: List[List[str]], num_workers: int = 1) -> List[List[DependencyTree]]:
        self.logger.debug("Begin processing with multi parser")
        max_trees = 10 # maximum valid sentences per group

        # --- STEP 1: Flatten input and perform validity checks ---
        flat_sentences = []       # list of all sentences (original strings)
        index_map = []            # list of tuples: (group_index, sentence_index)
        flatten_time_start = time.time()
        for group_index, group in enumerate(sentence_groups):
            for sentence_index, sentence in enumerate(group):
                flat_sentences.append(sentence)
                index_map.append((group_index, sentence_index))
        flatten_time_end = time.time()
        self.logger.info(f"flatten time in multiparser took: {flatten_time_end-flatten_time_start} seconds")

    
        if not flat_sentences:
            self.logger.warning("No sentences to process")
            return [[None] for _ in sentence_groups]

        tokenize_time = time.time()
        token_lists = self.parallel_preprocess_tokenize(flat_sentences, num_workers=num_workers)
        self.logger.info(f"tokenize time in multiparser took: {time.time() - tokenize_time}")


        validity_time = time.time()
        valid_flags = []          # True if the sentence is valid, False otherwise
        valid_count = {}          # per-group counter for valid sentences
        processed_texts =[]
        processed_tokens = []

        for group_index, sentence_group in enumerate(sentence_groups):
            valid_count[group_index] = 0


        for i, tokens in enumerate(token_lists):
            group_index, sentence_index = index_map[i]
            is_valid = True

            if not tokens:
                self.logger.debug(
                    f"Skipping sentence ({group_index}, {sentence_index}) due to no tokens after preprocessing: {sentence}"
                )
                is_valid = False
            elif len(tokens) < 4 or len(tokens) > 325:
                self.logger.debug(
                    f"Skipping sentence ({group_index}, {sentence_index}) due to too few or too many tokens after preprocessing: {sentence}"
                )
                is_valid = False

            # Enforce max_trees per group
            if is_valid and max_trees > 0:
                if valid_count[group_index] >= max_trees:
                    self.logger.debug(
                        f"Skipping sentence ({group_index}, {sentence_index}) due to maximum sentences per group cutoff"
                    )
                    is_valid = False
                else:
                    valid_count[group_index] += 1

            valid_flags.append(is_valid)
            processed_texts.append(" ".join(tokens) if is_valid else None)
            processed_tokens.append(tokens if is_valid else None)

        self.logger.info(f"validity time in multiparser took: {time.time() - validity_time} seconds")


        # --- STEP 2: Process the flat list with each parser ---
        parser_results = {}
        for name, parser in self.parsers.items():
            parser_time = time.time()
            try:
                # Each parser now processes the entire flat list at once.
                results_flat = parser.parse_batch_flat(flat_sentences, processed_texts, processed_tokens, num_workers=num_workers)
                processed_results = []
                # Ensure we keep the same structure: if a sentence was marked invalid,
                # or if the parser returned a tree that isn’t valid, force it to None.
                for idx, tree in enumerate(results_flat):
                    if (not valid_flags[idx]) or tree is None or (tree and not self._is_valid_tree(tree)):
                        processed_results.append(None)
                    else:
                        processed_results.append(tree)
                parser_results[name] = processed_results
            except Exception as e:
                self.logger.error(f"Parser {name} failed: {e}")
                # If a parser fails, fill its results with None so indexing stays intact.
                parser_results[name] = [None] * len(flat_sentences)
            self.logger.info(f"parser: {name} took {time.time() - parser_time} seconds in multiparser")
        # --- STEP 3: Combine results using the base parser and reassemble groups ---
        base_parser_name = self.config.parser.feature_sources['tree_structure']
        base_results = parser_results.get(base_parser_name, [None] * len(flat_sentences))
        final_results_flat = [None] * len(flat_sentences)

        enhancement_time = time.time()
        for idx, (group_index, sentence_index) in enumerate(index_map):
            base_tree = base_results[idx]
            # Skip if base parser returned nothing
            if base_tree is None:
                continue
            # Skip if any parser failed for this sentence
            if any(parser_results[name][idx] is None for name in self.parsers):
                continue

            base_tree.config = self.config
            # Enhance the base tree with features from all parsers
            features = {name: parser_results[name][idx] for name in self.parsers}
            self._enhance_tree(base_tree, features)
            if not self._is_valid_contents(base_tree):
                continue

            final_results_flat[idx] = base_tree

        self.logger.info(f"enhancement time in multiparser took: {time.time() - enhancement_time} seconds")

        reassembly_time = time.time()
        # Reassemble the final results back into the original grouping/shape.
        final_groups = [ [None] * len(group) for group in sentence_groups ]
        for idx, (group_index, sentence_index) in enumerate(index_map):
            final_groups[group_index][sentence_index] = final_results_flat[idx]
        self.logger.info(f"reassembly time in multiparser took: {time.time() - reassembly_time} seconds")

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
            if node_count < 3 or node_count > 325:
                self.logger.debug(f"Tree has {node_count} nodes - outside valid range [3,325]")
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
