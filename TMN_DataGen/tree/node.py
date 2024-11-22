from typing import List, Tuple, Optional, Dict, Any
import numpy as np

class Node:
    def __init__(self, 
                 word: str,
                 lemma: str,
                 pos_tag: str,
                 idx: int,
                 features: Dict[str, Any] = None):
        self.word = word
        self.lemma = lemma
        self.pos_tag = pos_tag
        self.idx = idx
        self.features = features or {}
        
        # Tree structure
        self.children: List[Tuple['Node', str]] = []  # List of (node, dependency_type)
        self.parent: Optional['Node'] = None
        self.dependency_to_parent: Optional[str] = None
    
    def add_child(self, child_node: 'Node', dependency_type: str):
        """Add a child node with its dependency relationship"""
        self.children.append((child_node, dependency_type))
        child_node.parent = self
        child_node.dependency_to_parent = dependency_type
    
    def remove_child(self, child_node: 'Node'):
        """Remove a child node"""
        self.children = [(node, dep) for node, dep in self.children 
                        if node != child_node]
        child_node.parent = None
        child_node.dependency_to_parent = None
    
    def replace_child(self, old_child: 'Node', new_child: 'Node', 
                     dependency_type: Optional[str] = None):
        """Replace a child node with another node"""
        for i, (node, dep) in enumerate(self.children):
            if node == old_child:
                self.children[i] = (new_child, dependency_type or dep)
                old_child.parent = None
                old_child.dependency_to_parent = None
                new_child.parent = self
                new_child.dependency_to_parent = dependency_type or dep
                break
    
    def traverse_preorder(self):
        """Depth-first pre-order traversal"""
        yield self
        for child, _ in self.children:
            yield from child.traverse_preorder()
            
    def traverse_postorder(self):
        """Depth-first post-order traversal"""
        for child, _ in self.children:
            yield from child.traverse_postorder()
        yield self
    
    def traverse_levelorder(self):
        """Breadth-first traversal"""
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(child for child, _ in node.children)
    
    def get_subtree_nodes(self):
        """Get all nodes in the subtree rooted at this node"""
        return list(self.traverse_preorder())
    
    def to_dict(self) -> Dict:
        """Convert node to dictionary representation"""
        return {
            'word': self.word,
            'lemma': self.lemma,
            'pos_tag': self.pos_tag,
            'idx': self.idx,
            'features': self.features,
            'dependency_to_parent': self.dependency_to_parent
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node':
        """Create node from dictionary representation"""
        node = cls(
            word=data['word'],
            lemma=data['lemma'],
            pos_tag=data['pos_tag'],
            idx=data['idx'],
            features=data['features']
        )
        node.dependency_to_parent = data['dependency_to_parent']
        return node
