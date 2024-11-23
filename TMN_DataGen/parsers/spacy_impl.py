#spacy_impl.py

from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
import spacy
from typing import List, Any, Optional
from omegaconf import DictConfig

class SpacyTreeParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None):
        super().__init__(config)
        if not hasattr(self, 'model'):
            model_name = self.config.get('model_name', 'en_core_web_sm')
            self.model = spacy.load(model_name)
    
    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        docs = self.model.pipe(sentences)
        return [self._convert_to_tree(doc) for doc in docs]
    
    def parse_single(self, sentence: str) -> DependencyTree:
        doc = self.model(sentence)
        return self._convert_to_tree(doc)
    
    def _convert_to_tree(self, doc: Any) -> DependencyTree:
        # Create nodes
        nodes = [
            Node(
                word=token.text,
                lemma=token.lemma_,
                pos_tag=token.pos_,
                idx=token.i,
                features={'original_text': token.text}
            )
            for token in doc
        ]
        
        # Connect nodes
        root = None
        for token in doc:
            if token.dep_ == 'ROOT':
                root = nodes[token.i]
            else:
                parent = nodes[token.head.i]
                parent.add_child(nodes[token.i], token.dep_)
        
        return DependencyTree(root)
