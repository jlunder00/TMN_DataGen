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
    
    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        self.logger.info("Begin Spacy batch processing")
        processed_sentences = []
        trees = []
        for sentence in sentences:
            try:
                if self.verbose == 'info' or self.verbose == 'debug':
                    self.logger.info(f"Processing {sentence} with Spacy")
                # Preprocess first
                tokens = self.preprocess_and_tokenize(sentence)

                if not tokens:
                    self.logger.warning(f"No tokens after processing: {sentence}")
                    continue

                # Join tokens for spaCy - it expects a text string
                processed_text = ' '.join(tokens)
                processed_sentences.append(processed_text)

                doc = self.model(processed_text)
                tree = self._convert_to_tree(sentence, doc)
                trees.append(tree)
                
                self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")
                

            except Exception as e:
                self.logger.error(f"Error processing sentence {sentence}: {e}")
                continue

        if not trees:
            self.logger.warning("No valid trees produced from batch")
        # docs = self.model.pipe(processed_sentences)
        # trees = [self._convert_to_tree(doc) for doc in docs]
        if self.verbose == 'debug':
            for sent, tree in zip(processed_sentences, trees):
                self.logger.debug(f"\nSpacy processed sentence: {sent}")
                self.logger.debug(f"Generated Spacy tree with {len(tree.root.get_subtree_nodes())} nodes")
        return trees

    def parse_single(self, sentence: str) -> DependencyTree:
        self.logger.info("Begin Spacy single processing")
        # Preprocess
        self.logger.info(f"Processing {sentence} with Spacy")
        tokens = self.preprocess_and_tokenize(sentence)
        processed_text = ' '.join(tokens)

        self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")
        doc = self.model(processed_text)
        return self._convert_to_tree(sentence, doc)
    
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
