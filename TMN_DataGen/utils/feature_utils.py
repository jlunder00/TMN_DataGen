# TMN_DataGen/TMN_DataGen/utils/feature_utils.py
import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel
from omegaconf import DictConfig

class FeatureExtractor:
    _instance = None
    
    def __new__(cls, config: Optional[DictConfig] = None):
        if cls._instance is None:
            cls._instance = super(FeatureExtractor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[DictConfig] = None):
        if not hasattr(self, 'initialized'):
            self.config = config or {}
            
            # Load model configurations
            model_name = self.config.get('feature_extraction', {}).get(
                'word_embedding_model', 'bert-base-uncased')
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Load feature mappings from config
            self.feature_mappings = self.config.get('feature_mappings', {
                'pos_tags': self._default_pos_tags(),
                'dep_types': self._default_dep_types(),
                'morph_features': self._default_morph_features()
            })
            
            self.initialized = True
    
    def _default_pos_tags(self) -> List[str]:
        return ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'ADP', 'NUM', 
                'PRON', 'CONJ', 'PART', 'PUNCT']
    
    def _default_dep_types(self) -> List[str]:
        return ['nsubj', 'obj', 'iobj', 'det', 'nmod', 'amod', 'advmod',
                'nummod', 'appos', 'conj', 'cc', 'punct']
    
    def _default_morph_features(self) -> List[str]:
        return ['Number', 'Person', 'Tense', 'VerbForm', 'Case']
    
    def get_word_embedding(self, word: str) -> torch.Tensor:
        inputs = self.tokenizer(word, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    def get_feature_embedding(self, feature_value: str, feature_type: str) -> torch.Tensor:
        """Generic one-hot embedding for any feature type"""
        feature_list = self.feature_mappings.get(feature_type, [])
        idx = feature_list.index(feature_value) if feature_value in feature_list else len(feature_list)
        embedding = torch.zeros(len(feature_list) + 1)
        embedding[idx] = 1.0
        return embedding
    
    def create_node_features(self, node, feature_config: Dict) -> torch.Tensor:
        """Create complete feature vector for a node based on config"""
        features = []
        
        if feature_config.get('use_word_embeddings', True):
            features.append(self.get_word_embedding(node.word))
        
        if feature_config.get('use_pos_tags', True):
            features.append(self.get_feature_embedding(node.pos_tag, 'pos_tags'))
        
        if feature_config.get('use_morph_features', False):
            for morph_feat in self.feature_mappings['morph_features']:
                feat_value = node.features.get(morph_feat, '')
                features.append(self.get_feature_embedding(feat_value, f'morph_{morph_feat}'))
        
        return torch.cat(features)
    
    def create_edge_features(self, dep_type: str) -> torch.Tensor:
        """Create feature vector for an edge"""
        return self.get_feature_embedding(dep_type, 'dep_types')

