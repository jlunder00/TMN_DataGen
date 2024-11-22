from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    from .node import Node
except ImportError:
    from node import Node


class DependencyTree:
    def __init__(self, root: Node):
        self.root = root
    
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
        
        # Swap in parents' children lists
        if parent1:
            parent1.replace_child(node1, node2, dep1)
        if parent2:
            parent2.replace_child(node2, node1, dep2)
            
        # Handle case where one is parent of the other
        if node1 in [child for child, _ in node2.children]:
            idx = [child for child, _ in node2.children].index(node1)
            node2.children[idx] = (node1, node2.children[idx][1])
        elif node2 in [child for child, _ in node1.children]:
            idx = [child for child, _ in node1.children].index(node2)
            node1.children[idx] = (node2, node1.children[idx][1])
            
        # Swap children lists
        node1.children, node2.children = node2.children, node1.children
        
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
        """Convert node to feature vector - implement in subclass"""
        raise NotImplementedError
    
    def _create_edge_features(self, dependency_type: str) -> np.ndarray:
        """Convert dependency type to feature vector - implement in subclass"""
        raise NotImplementedError
    
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


