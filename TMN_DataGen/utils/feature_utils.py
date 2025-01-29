# TMN_DataGen/TMN_DataGen/utils/feature_utils.py

import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Set
from transformers import AutoTokenizer, AutoModel
from omegaconf import DictConfig
import logging
from pathlib import Path
import json
import numpy as np
from ..utils.logging_config import setup_logger
from ..utils.embedding_cache import ParallelEmbeddingCache

class FeatureExtractor:
    _instance = None
    _logger = None

    @classmethod
    def _get_logger(cls, verbosity='normal'):
        if cls._logger is None:
            cls._logger = setup_logger('FeatureExtractor', verbosity)
        return cls._logger
    
    def __new__(cls, config: Optional[DictConfig] = None):
        if cls._instance is None:
            verbosity = config.get('verbose', 'normal') if config else 'normal'
            cls._logger = cls._get_logger(verbosity)
            cls._logger.debug("Creating new FeatureExtractor instance")
            cls._instance = super(FeatureExtractor, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[DictConfig] = None):
        if not hasattr(self, 'initialized'):
            self.config = config or {}
            self.logger = self._logger

            self.logger.info("Initializing Feature Extractor")
            
            # 1. Initialize feature mappings first
            self.feature_mappings = self._initialize_feature_mappings()
            
            # 2. Then calculate dimensions
            self.morph_dim = self._calculate_morph_dim()
            self.pos_dim = len(self.feature_mappings['pos_tags']) + 1
            self.dep_dim = len(self.feature_mappings['dep_types']) + 1
            
            # 3. Load model/tokenizer after mappings are ready
            model_cfg = self.config.get('feature_extraction', {})
            self.logger.info(f"feature extractor config loaded: {model_cfg}")
            self.logger.info(f"overall config: {self.config}")
            self.do_not_store_word_embeddings = model_cfg.get("do_not_store_word_embeddings", False)
            self.use_lemma = model_cfg.get("use_lemma", False)
            self.is_runtime = model_cfg.get("is_runtime", False)

            self.model_name = model_cfg.get('word_embedding_model', 'bert-base-uncased')
            self.use_gpu = model_cfg.get('use_gpu', True) and torch.cuda.is_available()
            self.cache_embeddings = model_cfg.get('cache_embeddings', True)
            self.embedding_cache_dir = Path(model_cfg.get('embedding_cache_dir', 'embedding_cache'))
            
            if self.cache_embeddings:
                self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
                self.embedding_cache = ParallelEmbeddingCache(
                    cache_dir = self.embedding_cache_dir,
                    shard_size = model_cfg.get("shard_size", 10000),
                    num_workers = model_cfg.get("num_workers", None),
                    config = model_cfg
                )
                # self._load_embedding_cache()
                self.embedding_cache.load()

            self.logger.info("setting device")
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')
            self.logger.info("device set")
            
            try:
                self.logger.info(f"Loading tokenizer and model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                if self.use_gpu:
                    self.model = self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size if not self.do_not_store_word_embeddings else 0
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise

            self.logger.info(f"Feature dimensions - Embedding: {self.embedding_dim}, "
                           f"POS: {self.pos_dim}, Dep: {self.dep_dim}, "
                           f"Morph: {self.morph_dim}")
            
            self.initialized = True

    def _initialize_feature_mappings(self) -> Dict[str, List[str]]:
        """Initialize all feature mappings with defaults or from config"""
        mappings = {
            'pos_tags': self._default_pos_tags(),
            'dep_types': self._default_dep_types(),
            'morph_features': self._default_morph_features()
        }
        
        # Override from config if provided
        cfg_mappings = self.config.get('feature_mappings', {})
        for key in mappings:
            if key in cfg_mappings:
                mappings[key] = cfg_mappings[key]
                
        return mappings

    def _default_pos_tags(self) -> List[str]:
        """Default Universal Dependencies POS tags"""
        return ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 
                'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 
                'VERB', 'X']

    def _default_dep_types(self) -> List[str]:
        """Default Universal Dependencies relation types"""
        return ['acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case',
                'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj',
                'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed',
                'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj',
                'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct',
                'reparandum', 'root', 'vocative', 'xcomp']

    def _default_morph_features(self) -> List[str]:
        """Default morphological features to track"""
        return ['Number', 'Person', 'Tense', 'VerbForm', 'Case', 'Gender', 
                'Mood', 'Voice', 'Aspect']

    def _calculate_morph_dim(self) -> int:
        """Calculate total dimension of morphological features"""
        return len(self.feature_mappings['morph_features']) * 2

    def _load_embedding_cache(self):
        """Load cached embeddings if they exist"""
        cache_file = self.embedding_cache_dir / "embedding_cache.npz"
        if cache_file.exists():
            self.logger.info("Loading embedding cache")
            cache_data = np.load(cache_file, allow_pickle=True)
            self.logger.info("Embedding cache load started")
            self.logger.info(f"Embedding cache contains {len(cache_data.values())} items")
            self.embedding_cache = {}
            for word, emb in tqdm(cache_data.items()):
                self.embedding_cache[word] = torch.from_numpy(emb)
        else:
            self.embedding_cache = {}

    def _save_embedding_cache(self):
        """Save cached embeddings"""
        if not self.cache_embeddings:
            self.logger.info("NOT SAVING EMBEDDINGS TO CACHE")
            return
        self.embedding_cache.save() 
        # cache_file = self.embedding_cache_dir / "embedding_cache.npz"
        # np.savez(
        #     cache_file, 
        #     **{word: emb.numpy() for word, emb in self.embedding_cache.items()}
        # )

    def get_word_embedding(self, word: str) -> torch.Tensor:
        """Get BERT embedding for a word with caching"""
        if self.cache_embeddings and word in self.embedding_cache:
            if self.do_not_store_word_embeddings and not self.is_runtime:
                return None  # No need to spend time doing retrieval
            return self.embedding_cache[word]

        inputs = self.tokenizer(
            word,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to GPU if available
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean of all subword tokens if word is split
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                
            # Move back to CPU and convert to numpy
            embedding = embedding.cpu()
            
            if self.cache_embeddings:
                self.embedding_cache[word] = embedding
                if len(self.embedding_cache) % 100 == 0:  # Periodic saving
                    self._save_embedding_cache()
                    
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting embedding for word '{word}': {e}")
            # Return zero vector as fallback
            return torch.zeros(self.embedding_dim)

    def get_feature_embedding(self, feature_value: str, feature_type: str) -> torch.Tensor:
        """One-hot embedding for categorical features"""
        if feature_type not in self.feature_mappings:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
        feature_list = self.feature_mappings[feature_type]
        # Handle unknown values
        if feature_value not in feature_list:
            self.logger.debug(f"Unknown {feature_type} value: {feature_value}")
            idx = len(feature_list)  # Use last index for unknown
        else:
            idx = feature_list.index(feature_value)
            
        embedding = torch.zeros(len(feature_list) + 1)
        embedding[idx] = 1.0
        return embedding

    def create_node_features(self, node) -> torch.Tensor:
        """Create complete feature vector for a node"""
        features = []
        # Word embedding (largest part)
        word_emb = self.get_word_embedding(node.word)
        if not self.do_not_store_word_embeddings:
            features.append(word_emb)
        
        # POS tag one-hot
        pos_emb = self.get_feature_embedding(node.pos_tag, 'pos_tags')
        features.append(pos_emb)
        
        # Morphological features
        morph_features = torch.zeros(self.morph_dim)
        if node.features and 'morph_features' in node.features:
            idx = 0
            for feat in self.feature_mappings['morph_features']:
                if feat in node.features['morph_features']:
                    morph_features[idx:idx+2] = torch.tensor([1., 0.])
                else:
                    morph_features[idx:idx+2] = torch.tensor([0., 1.])
                idx += 2
        features.append(morph_features)
        
        try:
            return torch.cat(features)
        except Exception as e:
            self.logger.error(f"Error concatenating features for node {node.word}: {e}")
            self.logger.debug(f"Feature shapes: {[f.shape for f in features]}")
            raise

    def prepend_node_features_with_word_emb(self, word, features) -> torch.Tensor:
        new_features = []
        word_emb = self.get_word_embedding(word)
        new_features.append(word_emb)
        new_features.append(torch.tensor(features))
        try:
            return torch.cat(new_features)
        except Exception as e:
            self.logger.error(f"Error concatenating prepended word embedding to node features for node {word}: {e}")
            self.logger.debug(f"Feature shapes: {[f.shape for f in features]}")
            self.logger.debug(f"word embedding shape: {new_features[0].shape}, other features shape: {new_features[1].shape}")
            raise

    def prepend_tree_features_with_word_emb(self, tree_json) -> Dict:
        for i, node_features in enumerate(tree_json['node_features']):
            word, lemma = tree_json['node_texts']
            tree_json['node_features'][i] = self.prepend_node_features_with_word_emb(lemma if self.use_lemma else word, node_features)
        return tree_json


    def create_edge_features(self, dep_type: str) -> torch.Tensor:
        """Create feature vector for an edge based on dependency type"""
        return self.get_feature_embedding(dep_type, 'dep_types')

    def get_feature_dim(self) -> Dict[str, int]:
        """Get dimensions of all features"""
        return {
            'node': self.embedding_dim + self.pos_dim + self.morph_dim,
            'edge': self.dep_dim
        }

    def __del__(self):
        """Save cache when object is destroyed"""
        if hasattr(self, 'cache_embeddings') and self.cache_embeddings:
            self._save_embedding_cache()
