# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# TMN_DataGen/TMN_DataGen/parsers/multi_parser.py
import time
import copy
import torch
import gc
from typing import List, Dict, Any, Optional, Tuple, List
from omegaconf import DictConfig
from .base_parser import BaseTreeParser
from .diaparser_impl import DiaParserTreeParser
from .spacy_impl import SpacyTreeParser
from ..tree.dependency_tree import DependencyTree
from ..tree.node import Node
import yaml
from pathlib import Path
from ..utils.parallel_framework import ParallelizationMixin, batch_parallel_process, _multiparser_validate_token_worker, _multiparser_enhance_tree_worker
from concurrent.futures import ProcessPoolExecutor

class MultiParser(BaseTreeParser, ParallelizationMixin):
    def __init__(self, config=None, pkg_config=None, vocabs=[set({})], logger=None, max_concurrent=1, num_workers=1):
        # Load package capabilities config
        self.capabilities = pkg_config['parsers'] if pkg_config else {}

        # Initialize base class
        super().__init__(config, pkg_config, vocabs, logger, max_concurrent, num_workers)

        ParallelizationMixin.__init__(self)

        # Validate user's feature source config against capabilities
        self._validate_feature_sources()

        # Initialize parsers
        self.parsers = {}
        for parser_name, parser_config in self.config.parser.parsers.items():
            if parser_config.get('enabled', False):
                parser_class = self._get_parser_class(parser_name)
                p_config = copy.deepcopy(self.config)
                p_config['model_name'] = parser_config['model_name']
                self.parsers[parser_name] = parser_class(p_config, max_concurrent=max_concurrent, num_workers=num_workers)
        
        self.initialized = True

    def _validate_token_batch(self, validation_data_batch: List[Tuple]) -> List[Tuple]:
        """Process a batch of token validation data"""
        results = []
        
        for i, tokens, group_index, sentence_index, sentence, max_trees, valid_count in validation_data_batch:
            is_valid = True
            
            if not tokens:
                self.logger.debug(
                    f"Skipping sentence ({group_index}, {sentence_index}) due to no tokens: {sentence}"
                )
                is_valid = False
            elif len(tokens) < self.config.parser.min_tokens or len(tokens) > self.config.parser.max_tokens:
                self.logger.debug(
                    f"Skipping sentence ({group_index}, {sentence_index}) " 
                    f"due to token count ({len(tokens)}): {sentence}"
                )
                is_valid = False
            
            # Note: max_trees enforcement needs to be done sequentially due to shared state
            processed_text = " ".join(tokens) if is_valid else None
            processed_tokens = tokens if is_valid else None
            
            results.append((i, is_valid, processed_text, processed_tokens, group_index))
        
        return results
    
    def _enhance_tree_batch(self, enhancement_data_batch: List[Tuple]) -> List[Tuple]:
        """Process a batch of tree enhancement operations"""
        results = []
        
        for idx, base_tree, parser_results_for_idx, group_index, sentence_index in enhancement_data_batch:
            try:
                if base_tree is None:
                    results.append((idx, None))
                    continue
                
                # Skip if any parser failed for this sentence
                if any(parser_results_for_idx[name] is None for name in self.parsers):
                    results.append((idx, None))
                    continue
                
                # Create a copy of the tree for processing
                enhanced_tree = copy.deepcopy(base_tree)
                enhanced_tree.config = self.config
                
                # Enhance the tree with features from all parsers
                features = {name: parser_results_for_idx[name] for name in self.parsers}
                self._enhance_tree(enhanced_tree, features)
                
                if not self._is_valid_contents(enhanced_tree):
                    results.append((idx, None))
                    continue
                
                results.append((idx, enhanced_tree))
                
            except Exception as e:
                self.logger.error(f"Error enhancing tree at index {idx}: {e}")
                results.append((idx, None))
        
        return results

    def parse_batch(self, sentence_groups: List[List[str]]) -> List[List[DependencyTree]]:
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
        token_lists = self.parallel_preprocess_tokenize(flat_sentences)
        self.logger.info(f"tokenize time in multiparser took: {time.time() - tokenize_time}")


        validity_time = time.time()
        valid_count = {group_index: 0 for group_index, _ in enumerate(sentence_groups)} # per-group counter for valid sentences

        if self.parallel_config.get('validity_checking', True) and len(token_lists) >= self._get_min_items_for_parallel() and self.num_workers > 1:
            # Parallel
            self.logger.info("Using parallel validity checking")
            
            # Prepare validation data
            validation_data = []
            for i, tokens in enumerate(token_lists):
                group_index, sentence_index = index_map[i]
                sentence = flat_sentences[i]
                validation_data.append((
                    i, tokens, group_index, sentence_index, sentence, 
                    max_trees, valid_count[group_index]
                ))
            
            chunk_size = self._get_chunk_size('validity_checking', 100, len(validation_data))

            validation_args = [(validation_data[i:i+chunk_size], self.config) 
                      for i in range(0, len(validation_data), chunk_size)]

            
            # Process validation in parallel
            validation_results = batch_parallel_process(
                validation_args,
                _multiparser_validate_token_worker,
                # lambda batch: self._validate_token_batch(batch) if isinstance(batch, list) else [self._validate_token_batch([batch])[0]],
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                maintain_order=True,
                min_items=self._get_min_items_for_parallel()
            )
            
            # Flatten results and apply max_trees constraint sequentially
            valid_flags = [False] * len(token_lists)
            processed_texts = [None] * len(token_lists)
            processed_tokens = [None] * len(token_lists)
            
            # Reset valid_count for sequential processing
            valid_count = {group_index: 0 for group_index, _ in enumerate(sentence_groups)}
            
            for result_batch in validation_results:
                if not isinstance(result_batch, list):
                    result_batch = [result_batch]
                    
                for i, is_valid, processed_text, processed_token, group_index in result_batch:
                    # Apply max_trees constraint
                    if is_valid and max_trees > 0:
                        if valid_count[group_index] >= max_trees:
                            is_valid = False
                        else:
                            valid_count[group_index] += 1
                    
                    valid_flags[i] = is_valid
                    processed_texts[i] = processed_text
                    processed_tokens[i] = processed_token
                    
        else:
            # Sequential
            valid_flags = []          # True if the sentence is valid, False otherwise
            processed_texts =[]
            processed_tokens = []

            for i, tokens in enumerate(token_lists):
                group_index, sentence_index = index_map[i]
                is_valid = True

                if not tokens:
                    self.logger.debug(
                        f"Skipping sentence ({group_index}, {sentence_index}) due to no tokens after preprocessing: {sentence}"
                    )
                    is_valid = False
                elif len(tokens) < self.config.parser.min_tokens or len(tokens) > self.config.parser.max_tokens:
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
                results_flat = parser.parse_batch_flat(flat_sentences, processed_texts, processed_tokens)
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

        if self.parallel_config.get('enhancement', True) and len(index_map) >= self._get_min_items_for_parallel() and self.num_workers > 1:
            # Parallel
            self.logger.info("Using parallel tree enhancement")
            
            # Prepare enhancement data
            enhancement_data = []
            for idx, (group_index, sentence_index) in enumerate(index_map):
                base_tree = base_results[idx]
                parser_results_for_idx = {name: parser_results[name][idx] for name in self.parsers}
                enhancement_data.append((idx, base_tree, parser_results_for_idx, group_index, sentence_index))
            
            
            chunk_size = self._get_chunk_size('enhancement', 30, len(enhancement_data))
            
            enhancement_args = [(enhancement_data[i:i+chunk_size], self.config) for i in range(0, len(enhancement_data), chunk_size)]

            # Process enhancement in parallel
            enhancement_results = batch_parallel_process(
                enhancement_args,
                _multiparser_enhance_tree_worker,
                # lambda batch: self._enhance_tree_batch(batch) if isinstance(batch, list) else [self._enhance_tree_batch([batch])[0]],
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                maintain_order=True,
                min_items = self._get_min_items_for_parallel()
            )
            
            # Apply results
            for result_batch in enhancement_results:
                if not isinstance(result_batch, list):
                    result_batch = [result_batch]
                    
                for idx, enhanced_tree in result_batch:
                    final_results_flat[idx] = enhanced_tree
            
        else:
            # Sequential
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
        if self.parallel_config.get('reassembly', True) and len(sentence_groups) >= self._get_min_items_for_parallel() and self.num_workers > 1:
            self.logger.info("Using parallel reassembly")
            
            # Create output structure
            final_groups = [[None] * len(group) for group in sentence_groups]
            
            # Prepare reassembly data
            reassembly_data = [(idx, group_index, sentence_index, final_results_flat[idx]) 
                              for idx, (group_index, sentence_index) in enumerate(index_map)]
            
            def reassembly_worker(batch):
                assignments = []
                for idx, group_index, sentence_index, tree in batch:
                    assignments.append((group_index, sentence_index, tree))
                return assignments
            
            chunk_size = self._get_chunk_size('reassembly', 200, len(reassembly_data))

            # Process reassembly in parallel
            assignment_batches = batch_parallel_process(
                reassembly_data,
                reassembly_worker,
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                maintain_order=False,  # Order doesn't matter for assignment
                min_items=self._get_min_items_for_parallel()
            )
            
            # Apply assignments
            for assignment_batch in assignment_batches:
                for group_index, sentence_index, tree in assignment_batch:
                    final_groups[group_index][sentence_index] = tree
            
        else:
            final_groups = [ [None] * len(group) for group in sentence_groups ]
            for idx, (group_index, sentence_index) in enumerate(index_map):
                final_groups[group_index][sentence_index] = final_results_flat[idx]
        self.logger.info(f"reassembly time in multiparser took: {time.time() - reassembly_time} seconds")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
            if node_count < self.config.parser.min_nodes or node_count > self.config.parser.max_nodes:
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
