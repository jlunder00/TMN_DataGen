# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# tests/test_feature_extraction.py
import pytest
import torch
from TMN_DataGen.utils import FeatureExtractor
from TMN_DataGen.tree import Node

@pytest.fixture
def feature_extractor(default_config):
    config, _ = default_config
    return FeatureExtractor(config)

def test_word_embeddings(feature_extractor):
    """Test word embedding generation"""
    word = "test"
    embedding = feature_extractor.get_word_embedding(word)
    
    assert isinstance(embedding, torch.Tensor)
    assert embedding.dim() == 1
    assert embedding.shape[0] == feature_extractor.embedding_dim

def test_node_features(feature_extractor):
    """Test complete node feature generation"""
    node = Node(
        word="test",
        lemma="test",
        pos_tag="NOUN",
        idx=0,
        features={
            'morph_features': {
                'Number': 'Sing',
                'Case': 'Nom'
            }
        }
    )
    
    features = feature_extractor.create_node_features(node)
    
    expected_dim = feature_extractor.get_feature_dim()['node']
    assert features.shape[0] == expected_dim
    
def test_edge_features(feature_extractor):
    """Test edge feature generation"""
    dep_type = "nsubj"
    features = feature_extractor.create_edge_features(dep_type)
    
    expected_dim = feature_extractor.get_feature_dim()['edge']
    assert features.shape[0] == expected_dim

def test_unknown_features(feature_extractor):
    """Test handling of unknown feature values"""
    # Unknown POS tag
    pos_emb = feature_extractor.get_feature_embedding("UNKNOWN_POS", "pos_tags")
    assert torch.argmax(pos_emb).item() == len(feature_extractor.feature_mappings['pos_tags'])
    
    # Unknown dependency type
    dep_emb = feature_extractor.get_feature_embedding("UNKNOWN_DEP", "dep_types")
    assert torch.argmax(dep_emb).item() == len(feature_extractor.feature_mappings['dep_types'])

def test_feature_dimensions_consistency(feature_extractor):
    """Test that all feature dimensions add up correctly"""
    node = Node(
        word="test",
        lemma="test",
        pos_tag="NOUN",
        idx=0,
        features={
            'morph_features': {
                'Number': 'Sing',
                'Case': 'Nom'
            }
        }
    )
    
    node_features = feature_extractor.create_node_features(node)
    edge_features = feature_extractor.create_edge_features("nsubj")
    
    dims = feature_extractor.get_feature_dim()
    
    assert node_features.shape[0] == dims['node']
    assert edge_features.shape[0] == dims['edge']

@pytest.mark.parametrize("word", [
    "test",
    "supercalifragilisticexpialidocious",  # Long word
    "test-word",  # Hyphenated
    "こんにちは",  # Non-English
])
def test_word_embedding_robustness(feature_extractor, word):
    """Test word embedding generation for various types of input"""
    embedding = feature_extractor.get_word_embedding(word)
    assert isinstance(embedding, torch.Tensor)
    assert not torch.isnan(embedding).any()
