from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
from diaparser.parsers import Parser
from typing import List, Any, Optional, Tuple
from omegaconf import DictConfig

class DiaParserTreeParser(BaseTreeParser):
    def __init__(self, config: Optional[DictConfig] = None):
        super().__init__(config)
        if not hasattr(self, 'model'):
            model_name = self.config.get('model_name', 'en_ewt.electra-base')
            self.model = Parser.load(model_name)
    
    def _process_prediction(self, dataset) -> Tuple[List[str], List[int], List[str]]:
        """Process diaparser prediction output"""
        sentence = dataset.sentences[0]  # CoNLLSentence object
        
        # Access the values directly from the CoNLLSentence object
        # The sentence.values contains tuples for each field in order:
        # [id, form, lemma, upos, xpos, feats, head, deprel, deps, misc]
        words = []
        heads = []
        rels = []
        
        # Get the form (words) from values[1]
        words_tuple = sentence.values[1]
        if isinstance(words_tuple, tuple) and len(words_tuple) == 1:
            # Split the sentence into words if it's a single string
            words = words_tuple[0].split()
        
        # Get the head indices from values[6]
        heads_list = sentence.values[6]
        if isinstance(heads_list, list):
            heads = heads_list
        
        # Get the dependency relations from values[7]
        rels_list = sentence.values[7]
        if isinstance(rels_list, list):
            rels = rels_list
        
        return words, heads, rels
    
    def parse_batch(self, sentences: List[str]) -> List[DependencyTree]:
        trees = []
        for sentence in sentences:
            # Get prediction dataset
            dataset = self.model.predict([sentence])
            # Process the prediction output
            words, heads, rels = self._process_prediction(dataset)
            
            # Create nodes
            nodes = [
                Node(
                    word=word,
                    lemma=word.lower(),  # Simple lemmatization for now
                    pos_tag="",  # We could add POS tags if needed
                    idx=idx,
                    features={'original_text': word}
                )
                for idx, word in enumerate(words)
            ]
            
            # Connect nodes
            root = None
            for idx, (head_idx, dep_label) in enumerate(zip(heads, rels)):
                if head_idx == 0:  # Root node
                    root = nodes[idx]
                else:
                    parent = nodes[head_idx - 1]  # diaparser uses 1-based indices
                    parent.add_child(nodes[idx], dep_label)
            
            trees.append(DependencyTree(root))
        
        return trees
    
    def parse_single(self, sentence: str) -> DependencyTree:
        return self.parse_batch([sentence])[0]

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
