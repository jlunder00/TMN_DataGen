# TMN_DataGen/TMN_DataGen/utils/viz_utils.py

from typing import Optional, Dict, Any, Tuple
from ..tree.node import Node 
from ..tree.dependency_tree import DependencyTree
from omegaconf import DictConfig
import graphviz

def print_tree_text(tree: DependencyTree, config: Optional[DictConfig] = None) -> str:
    """
    Create a text visualization of a dependency tree.
    
    Args:
        tree: DependencyTree to visualize
        config: Configuration dict, used for visualization settings
    
    Returns:
        String representation of the tree
        
    Example:
        >>> config = {'visualization': {'show_features': True}}
        >>> print(print_tree_text(tree, config))
        chases (VERB)
        ├── cat (NOUN) --nsubj--> [Number=Sing]
        └── mouse (NOUN) --obj--> [Number=Sing]
    """
    show_features = config.get('visualization', {}).get('show_features', False) if config else False
    
    def _build_lines(node: Node, prefix: str = "", is_last: bool = True) -> list[str]:
        lines = []
        
        # Create node text
        node_text = f"{node.word} ({node.pos_tag})"
        if node.dependency_to_parent:
            node_text = f"{node_text} --{node.dependency_to_parent}-->"
            
        if show_features and node.features:
            feat_str = ", ".join(f"{k}: {v}" for k, v in node.features.items())
            node_text = f"{node_text} [{feat_str}]"
        
        # Add node to output with proper prefixing
        conn = "└── " if is_last else "├── "
        lines.append(prefix + conn + node_text)
        
        # Handle children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for idx, (child, _) in enumerate(node.children):
            child_lines = _build_lines(
                child,
                prefix=child_prefix,
                is_last=(idx == len(node.children) - 1)
            )
            lines.extend(child_lines)
        
        return lines

    if not tree or not tree.root:
        return "<empty tree>"
        
    return "\n".join(_build_lines(tree.root))

def visualize_tree_graphviz(
    tree: DependencyTree, 
    config: Optional[DictConfig] = None,
    filename: Optional[str] = None
) -> graphviz.Digraph:
    """
    Create a graphical visualization of a dependency tree using graphviz.
    
    Args:
        tree: DependencyTree to visualize
        config: Configuration dict, used for visualization settings
        filename: If provided, save the visualization to this file
        
    Returns:
        Graphviz digraph object
    """
    show_features = config.get('visualization', {}).get('show_features', False) if config else False
    
    dot = graphviz.Digraph(comment='Dependency Tree')
    dot.attr(rankdir='TB')

    def _add_nodes_edges(node: Node):
        # Create node label
        node_label = f"{node.word}\n{node.pos_tag}"
        if show_features and node.features:
            feat_str = "\n".join(f"{k}: {v}" for k,v in node.features.items())
            node_label = f"{node_label}\n{feat_str}"
            
        # Add node
        node_id = str(node.idx)
        dot.node(node_id, node_label)
        
        # Add edges to children
        for child, dep_type in node.children:
            child_id = str(child.idx)
            dot.edge(node_id, child_id, label=dep_type)
            _add_nodes_edges(child)

    if tree and tree.root:
        _add_nodes_edges(tree.root)
        
    if filename:
        dot.render(filename, view=True)
        
    return dot

def format_tree_pair(
    tree1: DependencyTree, 
    tree2: DependencyTree,
    label: Optional[str] = None,
    config: Optional[DictConfig] = None
) -> str:
    """
    Format a pair of trees and their relationship as a string.
    
    Args:
        tree1: First tree (premise)
        tree2: Second tree (hypothesis)
        label: Relationship label
        config: Configuration dict, used for visualization settings
        
    Returns:
        Formatted string showing both trees and their relationship
    """
    s1 = print_tree_text(tree1, config)
    s2 = print_tree_text(tree2, config)
    
    # Split into lines and get max width
    s1_lines = s1.split('\n')
    s2_lines = s2.split('\n')
    max_len = max(len(line) for line in s1_lines)
    
    # Add padding between trees
    pad = 4
    max_len += pad
    
    # Combine lines side by side
    result = []
    result.append("Premise:" + " " * (max_len - 8) + "Hypothesis:")
    result.append("-" * (max_len + max(len(line) for line in s2_lines)))
    
    for i in range(max(len(s1_lines), len(s2_lines))):
        line1 = s1_lines[i] if i < len(s1_lines) else ""
        line2 = s2_lines[i] if i < len(s2_lines) else ""
        result.append(f"{line1:<{max_len}}{line2}")
        
    if label:
        result.append("\nRelationship: " + label)
        
    return "\n".join(result)
