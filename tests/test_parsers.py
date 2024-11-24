# TMN_DataGen/tests/test_parsers.py
import pytest
from TMN_DataGen.utils.logging_config import get_logger
from TMN_DataGen.parsers import DiaParserTreeParser, SpacyTreeParser, MultiParser
from TMN_DataGen.tree import Node, DependencyTree 
from omegaconf import OmegaConf

@pytest.fixture
def base_config():
    return OmegaConf.create({
        'verbose': 'debug',
        'visualization': {
            'show_features': True
        }
    })

def test_diaparser(base_config):
    config = OmegaConf.merge(base_config, {
        'parser': {
            'type': 'diaparser' 
        }
    })
    parser = DiaParserTreeParser(config)
    
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    # Check tree structure
    assert tree.root is not None
    assert tree.root.word == "chases"
    
    # Verify all nodes were extracted
    nodes = tree.root.get_subtree_nodes()
    assert len(nodes) == 5  # should have 5 nodes
    words = set(node.word for node in nodes)
    assert words == {"The", "cat", "chases", "the", "mouse"}

def test_multi_parser(base_config):
    config = OmegaConf.merge(base_config, {
        'parser': {
            'type': 'multi',
            'parsers': {
                'diaparser': {
                    'enabled': True,
                    'model_name': 'en_ewt.electra-base'
                },
                'spacy': {
                    'enabled': True,
                    'model_name': 'en_core_web_sm'
                }
            }
        }
    })
    
    parser = MultiParser(config)
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    # Check full tree structure
    nodes = tree.root.get_subtree_nodes()
    assert len(nodes) == 5
    
    # Check if features from both parsers are present
    for node in nodes:
        assert isinstance(node.pos_tag, str)
        assert node.pos_tag != ""
        assert 'morph_features' in node.features

class TestMultiParser:
    @pytest.fixture
    def multi_parser_config(self, base_config):
        return OmegaConf.merge(base_config, {
            "parser": {
                "type": "multi",
                "batch_size": 32,
                "parsers": {
                    "diaparser": {
                        "enabled": True,
                        "model_name": "en_ewt.electra-base"
                    },
                    "spacy": {
                        "enabled": True,
                        "model_name": "en_core_web_sm"
                    }
                },
                "feature_sources": {
                    "tree_structure": "diaparser",
                    "pos_tags": "spacy",
                    "morph": "spacy",
                    "lemmas": "spacy"
                }
            }
        })
    
    def test_multi_parser_feature_combination(self, multi_parser_config):
        parser = MultiParser(multi_parser_config)
        sentence = "The big cat chases the small mouse."
        tree = parser.parse_single(sentence)
        
        # Check tree structure and features
        nodes = tree.root.get_subtree_nodes()
        assert len(nodes) == 7  # All words present
        
        for node in nodes:
            assert hasattr(node, 'word')
            assert hasattr(node, 'lemma')
            assert hasattr(node, 'pos_tag')
            
            if node != tree.root:
                assert node.dependency_to_parent is not None
    
    def test_multi_parser_batch_processing(self, multi_parser_config):
        parser = MultiParser(multi_parser_config)
        sentences = [
            "The cat chases the mouse.",
            "The dog barks loudly.",
            "Birds fly in the sky."
        ]
        
        trees = parser.parse_batch(sentences)
        assert len(trees) == 3
        
        # Check all trees are properly formed
        for tree in trees:
            nodes = tree.root.get_subtree_nodes()
            assert len(nodes) > 0
            assert all(isinstance(node.pos_tag, str) for node in nodes)
            assert all('morph_features' in node.features for node in nodes)

    def test_verbosity_levels(self, multi_parser_config, caplog):
        # Test debug level
        debug_config = OmegaConf.merge(multi_parser_config, {'verbose': 'debug'})
        parser = MultiParser(debug_config)
        sentence = "The cat chases the mouse."
        tree = parser.parse_single(sentence)
        
        assert any("Processing node" in msg for msg in caplog.messages)
        
        caplog.clear()
        
        # Test normal level
        normal_config = OmegaConf.merge(multi_parser_config, {'verbose': 'normal'})
        parser = MultiParser(normal_config)
        tree = parser.parse_single(sentence)
        
        assert any("Parsed tree structure" in msg for msg in caplog.messages)
        assert not any("Processing node" in msg for msg in caplog.messages)
