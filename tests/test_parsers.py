# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# tests/test_parsers.py
import pytest
from TMN_DataGen.parsers import MultiParser, DiaParserTreeParser

def test_diaparser_basic(default_config):
    """Test basic DiaParser functionality"""
    config, _ = default_config
    parser = DiaParserTreeParser(config)
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    assert tree.root is not None
    nodes = tree.root.get_subtree_nodes()
    assert len(nodes) == 5

def test_multi_parser(default_config):
    """Test MultiParser feature combination"""
    config, pkg_config = default_config
    parser = MultiParser(config, pkg_config)
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    nodes = tree.root.get_subtree_nodes()
    for node in nodes:
        assert node.pos_tag is not None  # From spaCy
        assert hasattr(node, 'dependency_to_parent')  # From DiaParser

def test_preprocessing(default_config):
    """Test preprocessing pipeline"""
    config, pkg_config = default_config
    parser = MultiParser(config, pkg_config)
    text = "Hello,   World! "
    processed = parser.preprocess_and_tokenize(text)
    assert processed == ['hello', 'world']

