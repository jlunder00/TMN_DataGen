# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# tests/test_tree.py
from TMN_DataGen.tree import Node, DependencyTree

def test_tree_basics():
    """Test basic tree functionality"""
    # Create simple tree
    root = Node("chases", "chase", "VERB", 0)
    n1 = Node("cat", "cat", "NOUN", 1)
    n2 = Node("mouse", "mouse", "NOUN", 2)
    
    root.add_child(n1, "nsubj")
    root.add_child(n2, "dobj")
    
    tree = DependencyTree(root)
    
    # Test structure
    assert len(root.children) == 2
    assert n1.parent == root
    assert n2.parent == root
    
    # Test traversal
    nodes = tree.root.get_subtree_nodes()
    assert len(nodes) == 3
    
def test_tree_modification():
    """Test tree modification operations"""
    root = Node("root", "root", "VERB", 0)
    n1 = Node("n1", "n1", "NOUN", 1) 
    n2 = Node("n2", "n2", "NOUN", 2)
    
    root.add_child(n1, "dep1")
    root.add_child(n2, "dep2")
    
    tree = DependencyTree(root)
    
    # Test node swap
    tree.swap_nodes(n1, n2)
    assert n2.dependency_to_parent == "dep1"
    assert n1.dependency_to_parent == "dep2"
