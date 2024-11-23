# TMN_DataGen/tests/test_tree.py
import pytest
from TMN_DataGen.tree import Node, DependencyTree
from omegaconf import OmegaConf
import numpy as np

class TestNode:
    @pytest.fixture
    def simple_node(self):
        return Node(
            word="cat",
            lemma="cat",
            pos_tag="NOUN",
            idx=0,
            features={"morph": "Number=Sing"}
        )
    
    @pytest.fixture
    def simple_tree(self):
        root = Node("chases", "chase", "VERB", 1)
        subj = Node("cat", "cat", "NOUN", 0)
        obj = Node("mouse", "mouse", "NOUN", 2)
        root.add_child(subj, "nsubj")
        root.add_child(obj, "obj")
        return DependencyTree(root)
    
    def test_node_creation(self, simple_node):
        assert simple_node.word == "cat"
        assert simple_node.lemma == "cat"
        assert simple_node.pos_tag == "NOUN"
        assert simple_node.idx == 0
        assert simple_node.features["morph"] == "Number=Sing"
        
    def test_node_children(self, simple_tree):
        root = simple_tree.root
        assert len(root.children) == 2
        assert root.children[0][1] == "nsubj"  # dependency type
        assert root.children[1][1] == "obj"
        
    def test_node_traversal(self, simple_tree):
        nodes = list(simple_tree.root.traverse_preorder())
        assert len(nodes) == 3
        assert [node.word for node in nodes] == ["chases", "cat", "mouse"]
        
        nodes = list(simple_tree.root.traverse_postorder())
        assert [node.word for node in nodes] == ["cat", "mouse", "chases"]
        
        nodes = list(simple_tree.root.traverse_levelorder())
        assert [node.word for node in nodes] == ["chases", "cat", "mouse"]
    
    def test_node_modification(self, simple_tree):
        root = simple_tree.root
        new_node = Node("dog", "dog", "NOUN", 3)
        
        # Test replacing a child
        root.replace_child(root.children[0][0], new_node, "nsubj")
        assert root.children[0][0].word == "dog"
        assert root.children[0][1] == "nsubj"
        
        # Test removing a child
        root.remove_child(new_node)
        assert len(root.children) == 1
        assert root.children[0][0].word == "mouse"

class TestDependencyTree:
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

    @pytest.fixture
    def complex_tree(self, multi_parser_config):
        # Create a more complex tree for testing
        root = Node("saw", "see", "VERB", 2)
        det1 = Node("the", "the", "DET", 0)
        noun1 = Node("cat", "cat", "NOUN", 1)
        det2 = Node("the", "the", "DET", 3)
        noun2 = Node("mouse", "mouse", "NOUN", 4)
        
        noun1.add_child(det1, "det")
        noun2.add_child(det2, "det")
        root.add_child(noun1, "nsubj")
        root.add_child(noun2, "obj")
        
        return DependencyTree(root, multi_parser_config)
    
    def test_tree_modification(self, complex_tree):
        # Test modifying nodes based on condition
        def is_det(node):
            return node.pos_tag == "DET"
        
        def remove_node(node):
            if node.parent:
                node.parent.remove_child(node)
        
        complex_tree.modify_subtree(is_det, remove_node)
        nodes = complex_tree.root.get_subtree_nodes()
        assert len(nodes) == 3
        assert all(node.pos_tag != "DET" for node in nodes)
    
    def test_tree_swap_nodes(self, complex_tree):
        nodes = complex_tree.root.get_subtree_nodes()
        subj = [n for n in nodes if n.word == "cat"][0]
        obj = [n for n in nodes if n.word == "mouse"][0]
        
        complex_tree.swap_nodes(subj, obj)
        
        # Check if swap was successful
        new_subj = [n for n in complex_tree.root.children if n[1] == "nsubj"][0][0]
        new_obj = [n for n in complex_tree.root.children if n[1] == "obj"][0][0]
        
        assert new_subj.word == "mouse"
        assert new_obj.word == "cat"
    
    def test_tree_to_graph_data(self, complex_tree):
        graph_data = complex_tree.to_graph_data()
        
        assert "node_features" in graph_data
        assert "edge_features" in graph_data
        assert "from_idx" in graph_data
        assert "to_idx" in graph_data
        
        assert isinstance(graph_data["node_features"], np.ndarray)
        assert isinstance(graph_data["edge_features"], np.ndarray)
        assert len(graph_data["from_idx"]) == len(graph_data["to_idx"])


# # TMN_DataGen/tests/test_tree.py
# import pytest
# from TMN_DataGen.tree import Node, DependencyTree
# import numpy as np

# class TestNode:
#     @pytest.fixture
#     def simple_node(self):
#         return Node(
#             word="cat",
#             lemma="cat",
#             pos_tag="NOUN",
#             idx=0,
#             features={"morph": "Number=Sing"}
#         )
#     
#     @pytest.fixture
#     def simple_tree(self):
#         root = Node("chases", "chase", "VERB", 1)
#         subj = Node("cat", "cat", "NOUN", 0)
#         obj = Node("mouse", "mouse", "NOUN", 2)
#         root.add_child(subj, "nsubj")
#         root.add_child(obj, "obj")
#         return DependencyTree(root)
#     
#     def test_node_creation(self, simple_node):
#         assert simple_node.word == "cat"
#         assert simple_node.lemma == "cat"
#         assert simple_node.pos_tag == "NOUN"
#         assert simple_node.idx == 0
#         assert simple_node.features["morph"] == "Number=Sing"
#         
#     def test_node_children(self, simple_tree):
#         root = simple_tree.root
#         assert len(root.children) == 2
#         assert root.children[0][1] == "nsubj"  # dependency type
#         assert root.children[1][1] == "obj"
#         
#     def test_node_traversal(self, simple_tree):
#         nodes = list(simple_tree.root.traverse_preorder())
#         assert len(nodes) == 3
#         assert [node.word for node in nodes] == ["chases", "cat", "mouse"]
#         
#         nodes = list(simple_tree.root.traverse_postorder())
#         assert [node.word for node in nodes] == ["cat", "mouse", "chases"]
#         
#         nodes = list(simple_tree.root.traverse_levelorder())
#         assert [node.word for node in nodes] == ["chases", "cat", "mouse"]
#     
#     def test_node_modification(self, simple_tree):
#         root = simple_tree.root
#         new_node = Node("dog", "dog", "NOUN", 3)
#         
#         # Test replacing a child
#         root.replace_child(root.children[0][0], new_node, "nsubj")
#         assert root.children[0][0].word == "dog"
#         assert root.children[0][1] == "nsubj"
#         
#         # Test removing a child
#         root.remove_child(new_node)
#         assert len(root.children) == 1
#         assert root.children[0][0].word == "mouse"

# class TestDependencyTree:
#     @pytest.fixture
#     def complex_tree(self):
#         # Create a more complex tree for testing
#         root = Node("saw", "see", "VERB", 2)
#         det1 = Node("the", "the", "DET", 0)
#         noun1 = Node("cat", "cat", "NOUN", 1)
#         det2 = Node("the", "the", "DET", 3)
#         noun2 = Node("mouse", "mouse", "NOUN", 4)
#         
#         noun1.add_child(det1, "det")
#         noun2.add_child(det2, "det")
#         root.add_child(noun1, "nsubj")
#         root.add_child(noun2, "obj")
#         
#         return DependencyTree(root)
#     
#     def test_tree_modification(self, complex_tree):
#         # Test modifying nodes based on condition
#         def is_det(node):
#             return node.pos_tag == "DET"
#         
#         def remove_node(node):
#             if node.parent:
#                 node.parent.remove_child(node)
#         
#         complex_tree.modify_subtree(is_det, remove_node)
#         nodes = complex_tree.root.get_subtree_nodes()
#         assert len(nodes) == 3
#         assert all(node.pos_tag != "DET" for node in nodes)
#     
#     def test_tree_swap_nodes(self, complex_tree):
#         nodes = complex_tree.root.get_subtree_nodes()
#         subj = [n for n in nodes if n.word == "cat"][0]
#         obj = [n for n in nodes if n.word == "mouse"][0]
#         
#         complex_tree.swap_nodes(subj, obj)
#         
#         # Check if swap was successful
#         new_subj = [n for n in complex_tree.root.children if n[1] == "nsubj"][0][0]
#         new_obj = [n for n in complex_tree.root.children if n[1] == "obj"][0][0]
#         
#         assert new_subj.word == "mouse"
#         assert new_obj.word == "cat"
#     
#     def test_tree_to_graph_data(self, complex_tree):
#         graph_data = complex_tree.to_graph_data()
#         
#         assert "node_features" in graph_data
#         assert "edge_features" in graph_data
#         assert "from_idx" in graph_data
#         assert "to_idx" in graph_data
#         
#         assert isinstance(graph_data["node_features"], np.ndarray)
#         assert isinstance(graph_data["edge_features"], np.ndarray)
#         assert len(graph_data["from_idx"]) == len(graph_data["to_idx"])
