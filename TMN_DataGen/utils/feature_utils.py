import torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel

class FeatureExtractor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FeatureExtractor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.pos_tag_map = self._create_pos_tag_map()
            self.dep_type_map = self._create_dep_type_map()
            self.initialized = True
    
    def _create_pos_tag_map(self) -> Dict[str, int]:
        """Create mapping for POS tags"""
        common_pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'ADP', 'NUM', 
                          'PRON', 'CONJ', 'PART', 'PUNCT']
        return {tag: idx for idx, tag in enumerate(common_pos_tags)}
    
    def _create_dep_type_map(self) -> Dict[str, int]:
        """Create mapping for dependency types"""
        common_deps = ['nsubj', 'obj', 'iobj', 'det', 'nmod', 'amod', 'advmod',
                      'nummod', 'appos', 'conj', 'cc', 'punct']
        return {dep: idx for idx, dep in enumerate(common_deps)}
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        """Get BERT embedding for a word"""
        inputs = self.tokenizer(word, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use mean of all token embeddings for the word
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    def get_pos_embedding(self, pos_tag: str) -> torch.Tensor:
        """Convert POS tag to one-hot embedding"""
        idx = self.pos_tag_map.get(pos_tag, len(self.pos_tag_map))
        embedding = torch.zeros(len(self.pos_tag_map) + 1)
        embedding[idx] = 1.0
        return embedding
    
    def get_dep_embedding(self, dep_type: str) -> torch.Tensor:
        """Convert dependency type to one-hot embedding"""
        idx = self.dep_type_map.get(dep_type, len(self.dep_type_map))
        embedding = torch.zeros(len(self.dep_type_map) + 1)
        embedding[idx] = 1.0
        return embedding

def create_node_features(node) -> torch.Tensor:
    """Create feature vector for a node"""
    extractor = FeatureExtractor()
    word_emb = extractor.get_word_embedding(node.word)
    pos_emb = extractor.get_pos_embedding(node.pos_tag)
    return torch.cat([word_emb, pos_emb])

def create_edge_features(dep_type: str) -> torch.Tensor:
    """Create feature vector for an edge"""
    extractor = FeatureExtractor()
    return extractor.get_dep_embedding(dep_type)
