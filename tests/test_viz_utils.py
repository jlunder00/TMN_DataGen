# TMN_DataGen/tests/test_viz_utils.py
import pytest
from TMN_DataGen.utils.viz_utils import *
from TMN_DataGen.tree.node import Node
from TMN_DataGen.tree.dependency_tree import DependencyTree
from omegaconf import OmegaConf

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
    
    @pytest.fixture
    def config(self):
        return OmegaConf.create({
            'visualization': {
                'show_features': True
            },
            'verbose': 'normal'
        })
    
    def test_print_tree_text_with_features(self, sample_tree, config):
        # Add some features
        sample_tree.root.features = {"tense": "present"}
        text = print_tree_text(sample_tree, config)
        assert "tense: present" in text
        assert "chases (VERB)" in text
        assert "--nsubj-->" in text
        
    def test_print_tree_text_without_features(self, sample_tree):
        config = OmegaConf.create({'visualization': {'show_features': False}})
        text = print_tree_text(sample_tree, config)
        assert "tense: present" not in text
        assert "chases (VERB)" in text


