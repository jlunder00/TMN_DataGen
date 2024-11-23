# TMN_DataGen/tests/test_feature_extraction.py
import pytest
import torch
from TMN_DataGen.utils.feature_utils import FeatureExtractor
from omegaconf import OmegaConf

def test_feature_extraction():
    config = OmegaConf.load('configs/multi_parser_config.yaml')
    extractor = FeatureExtractor(config)
    
    # Test word embedding
    word_emb = extractor.get_word_embedding("cat")
    assert isinstance(word_emb, torch.Tensor)
    
    # Test feature embedding
    pos_emb = extractor.get_feature_embedding("NOUN", "pos_tags")
    assert isinstance(pos_emb, torch.Tensor)
    assert pos_emb.sum() == 1.0  # One-hot encoding
