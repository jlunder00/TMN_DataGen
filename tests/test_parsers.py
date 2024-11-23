# TMN_DataGen/tests/test_parsers.py
import pytest
from TMN_DataGen.parsers import DiaParserTreeParser, SpacyTreeParser, MultiParser
from TMN_DataGen.tree import Node, DependencyTree 
from omegaconf import OmegaConf

def test_diaparser():
    config = OmegaConf.create({'parser': {'type': 'diaparser'}})
    parser = DiaParserTreeParser(config)
    
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    assert tree.root is not None
    assert len(tree.root.get_subtree_nodes()) > 0

def test_multi_parser():
    config = OmegaConf.load('configs/multi_parser_config.yaml')
    parser = MultiParser(config)
    
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    # Check if features from both parsers are present
    nodes = tree.root.get_subtree_nodes()
    for node in nodes:
        assert isinstance(node.pos_tag, str)  # Should have POS tags from spaCy
        assert node.pos_tag != ""
        assert 'morph_features' in node.features  # Should have morph features from spaCy

class TestMultiParser:
    @pytest.fixture
    def multi_parser_config(self):
        return OmegaConf.create({
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
            },
            "feature_extraction": {
                "word_embedding_model": "bert-base-uncased",
                "use_word_embeddings": True,
                "use_pos_tags": True,
                "use_morph_features": True
            }
        })
    
    def test_multi_parser_initialization(self, multi_parser_config):
        parser = MultiParser(multi_parser_config)
        assert len(parser.parsers) == 2
        assert "diaparser" in parser.parsers
        assert "spacy" in parser.parsers
    
    def test_multi_parser_feature_combination(self, multi_parser_config):
        parser = MultiParser(multi_parser_config)
        sentence = "The big cat chases the small mouse."
        tree = parser.parse_single(sentence)
        
        # Check if features from both parsers are present
        nodes = tree.root.get_subtree_nodes()
        for node in nodes:
            # Check basic node attributes
            assert hasattr(node, 'word')
            assert hasattr(node, 'lemma')
            assert hasattr(node, 'pos_tag')
            assert hasattr(node, 'dependency_to_parent')
            
            # Check feature presence
            assert isinstance(node.pos_tag, str)
            assert node.pos_tag != ""
            assert isinstance(node.features, dict)
            assert 'morph_features' in node.features
            
            # If node isn't root, it should have dependency information
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
        
        for tree in trees:
            assert tree.root is not None
            nodes = tree.root.get_subtree_nodes()
            assert len(nodes) > 0
            
            for node in nodes:
                assert isinstance(node.pos_tag, str)
                assert node.pos_tag != ""
                assert isinstance(node.features, dict)
                assert 'morph_features' in node.features
                if node != tree.root:
                    assert node.dependency_to_parent is not None

