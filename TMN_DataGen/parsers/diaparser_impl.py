from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
from diaparser import Parser
from typing import List, Any, Optional
from omegaconf import DictConfig

class DiaParserTreeParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None):
        super().__init__(config)
        if not hasattr(self, 'model'):
            model_name = self.config.get('model_name', 'en')
            self.model = Parser.load(model_name)
    
    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        parse_results = self.model.parse(sentences)
        return [self._convert_to_tree(sent, result) 
                for sent, result in zip(sentences, parse_results)]
    
    def parse_single(self, sentence: str) -> DependencyTree:
        result = self.model.parse([sentence])[0]
        return self._convert_to_tree(sentence, result)
    
    def _convert_to_tree(self, sentence: str, parse_result: Any) -> DependencyTree:
        # Create nodes
        nodes = [
            Node(
                word=word,
                lemma=lemma,
                pos_tag=pos,
                idx=idx,
                features={'original_text': word}
            )
            for idx, (word, lemma, pos) 
            in enumerate(zip(parse_result.words, 
                           parse_result.lemmas,
                           parse_result.pos_tags))
        ]
        
        # Connect nodes
        root = None
        for idx, (head_idx, dep_label) in enumerate(zip(parse_result.head_indices,
                                                       parse_result.dep_labels)):
            if head_idx == 0:  # Root node
                root = nodes[idx]
            else:
                parent = nodes[head_idx - 1]  # diaparser uses 1-based indices
                parent.add_child(nodes[idx], dep_label)
        
        return DependencyTree(root)
