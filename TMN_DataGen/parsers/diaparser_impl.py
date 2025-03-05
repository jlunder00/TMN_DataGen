#diaparser_impl.py
import time
from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
from diaparser.parsers import Parser
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import DictConfig
import numpy as np

class DiaParserTreeParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None, pkg_config=None, vocabs=[set({})], logger=None):
        super().__init__(config, pkg_config, vocabs, logger)
        if not hasattr(self, 'model'):
            model_name = self.config.get('model_name', 'en_ewt.electra-base')
            self.model = Parser.load(model_name)
    
    def _process_prediction(self, dataset) -> List[Dict[str, List]]:
        """
        Process diaparser output into aligned lists of token information.
        DiaParser provides output in CoNLL-X format with these indices:
        1: FORM - Word form/token
        2: LEMMA - Lemma
        3: UPOS - Universal POS tag
        6: HEAD - Head token id
        7: DEPREL - Dependency relation

        Returns:
            List of Dict with keys: words, lemmas, pos_tags, heads, rels
            All lists are aligned by token position
        """
        token_data_group = []
        for i in range(len(dataset.sentences)):
            sentence = dataset.sentences[i]
            
            self.logger.debug(f"Processing CoNLL format sentence:")
            self.logger.debug(f"Raw values: {sentence.values}")
            
            def ensure_list(val) -> List[str]:
                """Convert various input formats to list"""
                if isinstance(val, str):
                    return val.split()
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

                token_data_group.append(token_data)
                
            except Exception as e:
                self.logger.error(f"Error processing parser output: {e}")
                self.logger.debug(f"Values: {sentence.values}")
                raise
        return token_data_group
        
    def parse_batch_flat(self, flat_sentences, processed_texts: List[str], processed_tokens: List[List[str]], num_workers: int = 1) -> List[List[DependencyTree]]:
        
        trees_flat = []
        valid_token_list_indices = [i for i, tokens in enumerate(processed_tokens) if tokens]
        valid_token_lists = [tokens for tokens in processed_tokens if tokens]

        parse_time = time.time()
        valid_dataset = self.model.predict(valid_token_lists, batch_size=self.diaparser_batch_size)
        valid_token_data = self._process_prediction(valid_dataset)

        token_data_flat = [None]*len(processed_tokens)
        for i, token_data in zip(valid_token_list_indices, valid_token_data):
            token_data_flat[i] = token_data
        self.logger.info(f"parsing in diaparser parser took: {time.time()-parse_time}")
            
        build_time = time.time()
        for token_data, sentence in zip(token_data_flat, flat_sentences):
            try:
                if token_data and sentence:
                    tree = self._build_tree(token_data, sentence)
                    if tree:
                        trees_flat.append(tree)
                    else:
                        trees_flat.append(None)
                else:
                    trees_flat.append(None)
            except Exception as e:
                self.logger.error(f"Error while building tree from diaparser data for: {sentence}: {e}")
                trees_flat.append(None)
        self.logger.info(f"tree building in diaparser took: {time.time()-build_time}")

        return trees_flat
    
    def parse_batch(self, sentence_groups: List[List[str]], num_workers: int = 1) -> List[List[DependencyTree]]:
        self.logger.debug(f"Parsing batch of {len(sentence_groups)} sentence groups")
        tree_groups = [self.parse_single(group, num_workers=num_workers) for group in sentence_groups]
        if len(tree_groups) < 1:
            self.logger.warning("No valid trees produced from batch")
            tree_groups = [[None for _ in group] for group in sentence_groups]
        return tree_groups
    
    def parse_single(self, sentences: List[str], num_workers: int = 1) -> List[DependencyTree]:
        """Parse a single sentence group into a dependency tree group"""
        self.logger.debug(f"Parsing group of {len(sentences)} sentences")
        # return self.parse_batch([sentence])[0]
        tokenized_sentences = []
        trees = []
        valid_sentences = []

        token_lists = self.parallel_preprocess_tokenize(sentences, num_workers)

        for tokens, sentence in zip(token_lists, sentences):

            if not tokens:
                if sentence != '?' and sentence != '.':
                    self.logger.debug(f"\nNo tokens after tokenize in {sentence}, skipping")
                trees.append(None)
                continue

            tokenized_sentences.append(tokens)
            valid_sentences.append(sentence)

        try:
            #diaparser can take in multiple token sets at once
            dataset = self.model.predict(tokenized_sentences)
            token_data_group = self._process_prediction(dataset)
        except Exception as e:
            self.logger.error(f"Error while making/processing diaparser prediction for {valid_sentences}: {e}")
            return [None for _ in sentences]
        for token_data, sentence in zip(token_data_group, valid_sentences):
            try:
                tree = self._build_tree(token_data, sentence)
                if tree:
                    trees.append(tree)
                else:
                    trees.append(None)
            except Exception as e:
                self.logger.error(f"Error while building tree from diaparser data for: {sentence}: {e}")
                trees.append(None)
        return trees

    def _build_tree(self, token_data: Dict, sentence: str) -> DependencyTree:
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

        tree = DependencyTree(sentence, root, config=self.config)
        
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

        if self.verbose == 'debug':
            from ..utils.viz_utils import print_tree_text
            self.logger.info("\nDiaparser parsed tree structure:")
            self.logger.info(print_tree_text(tree, self.config))
        return tree




