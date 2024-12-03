#dependency_tree.py
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from omegaconf import DictConfig

try:
    from .node import Node
except ImportError:
    from node import Node

class DependencyTree:
    def __init__(self, text: str, root: Node, config: Optional[DictConfig] = None):
        self.root = root
        self.config = config or {}
        # Cache feature dimensions
        self._feature_dims = None
        # Verify tree structure
        self.text = text
        if not self.root.verify_tree_structure():
            raise ValueError("Invalid tree structure detected")
    
    @property
    def feature_dims(self) -> Dict[str, int]:
        """Get or cache feature dimensions"""
        if self._feature_dims is None:
            from ..utils.feature_utils import FeatureExtractor
            extractor = FeatureExtractor(self.config)
            self._feature_dims = extractor.get_feature_dim()
        return self._feature_dims

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
    
    def to_graph_data(self) -> Dict[str, Any]:
        """Convert to format needed for GMN"""
        from ..utils.feature_utils import FeatureExtractor
        extractor = FeatureExtractor(self.config)
        nodes = self.root.get_subtree_nodes()
        
        # Get feature dimensions
        dims = self.feature_dims
        
        # Pre-allocate tensors for efficiency
        node_features = torch.zeros(len(nodes), dims['node'])
        edge_indices = []  # Will convert to tensor after collecting all
        edge_features = []  # Will stack after collecting all
        
        # Create node features
        for i, node in enumerate(nodes):
            try:
                node_features[i] = extractor.create_node_features(node)
            except Exception as e:
                raise ValueError(f"Failed to create features for node {node}: {e}")
            
        # Create edge list and features
        for i, node in enumerate(nodes):
            for child, dep_type in node.children:
                try:
                    child_idx = nodes.index(child)
                    edge_indices.append((i, child_idx))
                    edge_features.append(extractor.create_edge_features(dep_type))
                except Exception as e:
                    raise ValueError(f"Failed to process edge {node}->{child}: {e}")
                
        # Convert to final format
        edge_indices = torch.tensor(edge_indices).t() if edge_indices else torch.zeros((2, 0))
        edge_features = torch.stack(edge_features) if edge_features else torch.zeros((0, dims['edge']))
        
        # Convert to lists for JSON serialization
        return {
            'node_features': node_features.numpy().tolist(),
            'edge_features': edge_features.numpy().tolist(),
            'from_idx': edge_indices[0].numpy().tolist(),
            'to_idx': edge_indices[1].numpy().tolist(),
            'graph_idx': [0] * len(nodes),
            'n_graphs': 1,
            'text': self.text
        }
    
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
