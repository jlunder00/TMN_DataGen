#spacy_impl.py
from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
import spacy
from typing import List, Any, Optional
from omegaconf import DictConfig

class SpacyTreeParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None, pkg_config = None, logger = None):
        super().__init__(config, pkg_config, logger)
        if not hasattr(self, 'model'):
            model_name = self.config.get('model_name', 'en_core_web_sm')
            self.model = spacy.load(model_name)
    
    def parse_batch(self, sentence_groups: List[List[str]]) -> List[List[DependencyTree]]:
        self.logger.info("Begin Spacy batch processing")
        tree_groups = [self.parse_single(group) for group in sentence_groups]

        if len(tree_groups) < 1:
            self.logger.warning("No valid trees produced from batch")
            tree_groups = [[None for sentence in group] for group in sentence_groups]
        return tree_groups

    def parse_single(self, sentences: List[str]) -> List[DependencyTree]:
        self.logger.info("Begin Spacy single processing")
        # Preprocess
        trees = []
        processed_sentences = []
        self.logger.info(f"Processing sentence group {sentences} with Spacy")

        for sentence in sentences:
            try:
                if self.verbose == 'info' or self.verbose == 'debug':
                    self.logger.info(f"Processing {sentence} with Spacy")
                tokens = self.preprocess_and_tokenize(sentence)
                if not tokens:
                    self.logger.warning(f"No tokens after processing: {sentence}")
                    continue
                processed_text = ' '.join(tokens)
                self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")
                processed_sentences.append(processed_text)
                doc = self.model(processed_text)
                tree = self._convert_to_tree(sentence, doc)
                if tree:
                    trees.append(tree)
                else:
                    trees.append(None)
            except Exception as e:
                self.logger.error(f'Error processing sentence {sentence}: {e}')
                trees.append(None)
                continue

        if len(trees) < 1:
            self.logger.warning("No valid trees produced from batch")
            trees = [None]
        if self.verbose == 'debug':
            for sent, tree in zip(processed_sentences, trees):
                self.logger.debug(f"\nSpacy processed sentence: {sent}")
                self.logger.debug(f"Generated Spacy tree with {len(tree.root.get_subtree_nodes())} nodes")
                
        return trees

    
    def _convert_to_tree(self, sentence, doc: Any) -> DependencyTree:
        try:
            nodes = [
                Node(
                    word=token.text,
                    lemma=token.lemma_,
                    pos_tag=token.pos_,
                    idx=token.i,
                    features={
                        'original_text': token.text,
                        'morph_features': dict(feature.split('=') 
                                             for feature in str(token.morph).split('|')
                                             if feature != '')  # Handle empty morph case
                    }
                )
                for token in doc
            ]
            
            if not nodes:
                raise ValueError("No valid tokens in document")
            # Connect nodes
            root = None
            for token in doc:
                if token.dep_ == 'ROOT':
                    root = nodes[token.i]
                else:
                    parent = nodes[token.head.i]
                    parent.add_child(nodes[token.i], token.dep_)

            if not root:
                raise ValueError("No root node found in parse")
            
            tree = DependencyTree(sentence, root, self.config)
            self.logger.debug("Tree structure created successfully")

            return tree
        except Exception as e:
            self.logger.error(f"Failed to convert doc to tree: {e}")
            raise
