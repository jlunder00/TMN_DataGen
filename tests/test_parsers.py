# tests/test_parsers.py
import pytest
from TMN_DataGen.parsers import MultiParser, DiaParserTreeParser

def test_diaparser_basic():
    """Test basic DiaParser functionality"""
    parser = DiaParserTreeParser() 
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    # Verify tree structure
    assert tree.root is not None
    nodes = tree.root.get_subtree_nodes()
    assert len(nodes) == 5
    
    # Verify dependency labels
    root_node = tree.root
    assert root_node.word == 'chases'
    assert root_node.dependency_to_parent is None

def test_multi_parser():
    """Test MultiParser feature combination"""
    parser = MultiParser()
    sentence = "The cat chases the mouse."
    tree = parser.parse_single(sentence)
    
    nodes = tree.root.get_subtree_nodes()
    
    # Verify features from both parsers
    for node in nodes:
        # From spaCy
        assert node.pos_tag is not None
        assert node.lemma is not None
        
        # From DiaParser
        assert hasattr(node, 'dependency_to_parent')

def test_preprocessing():
    """Test preprocessing pipeline"""
    parser = MultiParser()
    text = "Hello,   World! "
    processed = parser.preprocess_and_tokenize(text)
    assert processed == ['hello', 'world']
