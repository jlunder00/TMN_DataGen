# TMN_DataGen/tests/test_viz_utils.py

import pytest
from TMN_DataGen.utils.viz_utils import *
from TMN_DataGen.tree.node import Node
from TMN_DataGen.tree.dependency_tree import DependencyTree
from TMN_DataGen.utils.logging_config import logger

class TestVizUtils:
    @pytest.fixture
    def sample_tree(self):
        # Create a simple tree: "The cat chases the mouse"
        root = Node("chases", "chase", "VERB", 2)
        det1 = Node("The", "the", "DET", 0)
        noun1 = Node("cat", "cat", "NOUN", 1)
        det2 = Node("the", "the", "DET", 3)
        noun2 = Node("mouse", "mouse", "NOUN", 4)
        
        noun1.add_child(det1, "det")
        noun2.add_child(det2, "det")
        root.add_child(noun1, "nsubj")
        root.add_child(noun2, "obj")
        
        return DependencyTree(root)
    
    def test_print_tree_text(self, sample_tree):
        text = print_tree_text(sample_tree)
        logger.info(text)
        assert "chases" in text
        assert "cat" in text
        assert "mouse" in text
        assert "--nsubj--" in text
        assert "--obj--" in text
        
    def test_print_tree_text_with_features(self, sample_tree):
        # Add some features
        sample_tree.root.features = {"tense": "present"}
        text = print_tree_text(sample_tree, show_features=True)
        logger.info(text)
        assert "tense: present" in text
        
    def test_visualize_tree_graphviz(self, sample_tree):
        dot = visualize_tree_graphviz(sample_tree)
        assert isinstance(dot, graphviz.Digraph)
        assert "chases" in dot.source
        assert "cat" in dot.source
        assert "mouse" in dot.source
        
    def test_format_tree_pair(self, sample_tree):
        # Create a second tree for testing pairs
        root2 = Node("is", "be", "VERB", 2)
        noun = Node("mouse", "mouse", "NOUN", 1)
        adj = Node("chased", "chase", "ADJ", 3)
        root2.add_child(noun, "nsubj")
        root2.add_child(adj, "acomp")
        tree2 = DependencyTree(root2)
        
        text = format_tree_pair(sample_tree, tree2, "entails")
        assert "Premise:" in text
        assert "Hypothesis:" in text
        assert "Relationship: entails" in text
