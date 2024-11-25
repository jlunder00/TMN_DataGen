#dependency_tree.py
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from omegaconf import DictConfig

try:
    from .node import Node
except ImportError:
    from node import Node

class DependencyTree:
    def __init__(self, root: Node, config: Optional[DictConfig] = None):
        self.root = root
        self.config = config or {}
        # Verify tree structure
        if not self.root.verify_tree_structure():
            raise ValueError("Invalid tree structure detected")
    
    def modify_subtree(self, condition_fn, modification_fn):
        """Apply modification to nodes that meet condition"""
        for node in self.root.traverse_preorder():
            if condition_fn(node):
                modification_fn(node)
    
    def swap_nodes(self, node1: Node, node2: Node):
        """Swap two nodes while preserving tree structure"""
        # Save parent relationships
        parent1 = node1.parent
        parent2 = node2.parent
        dep1 = node1.dependency_to_parent
        dep2 = node2.dependency_to_parent
        
        # Save children lists
        children1 = node1.children.copy()
        children2 = node2.children.copy()
        
        # Remove from current parents
        if parent1:
            parent1.remove_child(node1)
        if parent2:
            parent2.remove_child(node2)
        
        # Add to new parents
        if parent1:
            parent1.add_child(node2, dep1)
        if parent2:
            parent2.add_child(node1, dep2)
        
        # Update children
        node1.children = children2
        node2.children = children1
        
        # Update parent references in children
        for child, _ in node1.children:
            child.parent = node1
        for child, _ in node2.children:
            child.parent = node2
        
        # Update root if needed
        if self.root == node1:
            self.root = node2
        elif self.root == node2:
            self.root = node1
    
    def to_graph_data(self) -> Dict[str, np.ndarray]:
        """Convert to format needed for GMN"""
        nodes = self.root.get_subtree_nodes()
        
        # Create node features
        node_features = []
        for node in nodes:
            node_feat = self._create_node_features(node)  
            node_features.append(node_feat)
            
        # Create edge list and features
        from_idx = []
        to_idx = []
        edge_features = []
        
        for i, node in enumerate(nodes):
            for child, dep_type in node.children:
                child_idx = nodes.index(child)
                from_idx.append(i)
                to_idx.append(child_idx)
                edge_feat = self._create_edge_features(dep_type)
                edge_features.append(edge_feat)
                
        return {
            'node_features': np.array(node_features),
            'edge_features': np.array(edge_features),
            'from_idx': np.array(from_idx),
            'to_idx': np.array(to_idx)
        }
    
    def _create_node_features(self, node: Node) -> np.ndarray:
        from ..utils.feature_utils import FeatureExtractor
        extractor = FeatureExtractor(self.config)
        features = extractor.create_node_features(
            node, 
            self.config.get('feature_extraction', {})
        )
        return features.numpy()
    
    def _create_edge_features(self, dependency_type: str) -> np.ndarray:
        from ..utils.feature_utils import FeatureExtractor
        extractor = FeatureExtractor(self.config)
        features = extractor.create_edge_features(dependency_type)
        return features.numpy()
    
    def to_dict(self) -> Dict:
        """Convert tree to dictionary representation"""
        def node_to_dict(node: Node) -> Dict:
            node_dict = node.to_dict()
            node_dict['children'] = [
                (child.to_dict(), dep_type) 
                for child, dep_type in node.children
            ]
            return node_dict
        
        return {'root': node_to_dict(self.root)}
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DependencyTree':
        """Create tree from dictionary representation"""
        def dict_to_node(node_dict: Dict) -> Node:
            node = Node.from_dict(node_dict)
            for child_dict, dep_type in node_dict['children']:
                child = dict_to_node(child_dict)
                node.add_child(child, dep_type)
            return node
        
        root = dict_to_node(data['root'])
        return cls(root)


