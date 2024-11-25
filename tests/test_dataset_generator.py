# tests/test_dataset_generator.py
import pytest
from TMN_DataGen import DatasetGenerator
import pickle

def test_dataset_generation(sample_data, default_config, tmp_path):
    """Test basic dataset generation workflow"""
    config, pkg_config = default_config
    
    # Initialize generator
    generator = DatasetGenerator()
    output_path = tmp_path / "test_dataset.pkl"
    
    # Generate dataset
    generator.generate_dataset(
        sentence_pairs=sample_data['sentence_pairs'],
        labels=sample_data['labels'],
        output_path=str(output_path),
        parser_config=config,
        verbosity='normal'
    )
    
    assert output_path.exists()

def test_config_override(sample_data, default_config, tmp_path):
    """Test config override functionality"""
    config, pkg_config = default_config
    
    # Modify config
    config.parser.feature_sources.update({
        'tree_structure': 'diaparser',
        'pos_tags': 'spacy'
    })
    
    generator = DatasetGenerator()
    output_path = tmp_path / "test_dataset.pkl"
    
    generator.generate_dataset(
        sentence_pairs=sample_data['sentence_pairs'],
        labels=sample_data['labels'], 
        output_path=str(output_path),
        parser_config=config,
        verbosity='debug'
    )

def test_dataset_content(sample_data, default_config, tmp_path):
    """Test that generated dataset has correct structure and content"""
    config, pkg_config = default_config
    generator = DatasetGenerator()
    output_path = tmp_path / "test_dataset.json"
    
    generator.generate_dataset(
        sentence_pairs=sample_data['sentence_pairs'],
        labels=sample_data['labels'],
        output_path=str(output_path),
        parser_config=config
    )

    # Load and verify dataset
    with open(output_path) as f:
        data = json.load(f)
    
    # Check basic structure
    assert 'graph_pairs' in data
    assert 'labels' in data
    assert len(data['graph_pairs']) == len(data['labels'])
    
    # Check graph structure
    graph1 = data['graph_pairs'][0][0]  # First graph of first pair
    required_keys = ['node_features', 'edge_features', 'from_idx', 'to_idx', 
                     'graph_idx', 'n_graphs']
    for key in required_keys:
        assert key in graph1
        
    # Check feature dimensions
    n_nodes = len(graph1['node_features'])
    n_edges = len(graph1['edge_features'])
    assert len(graph1['from_idx']) == n_edges
    assert len(graph1['to_idx']) == n_edges
    assert len(graph1['graph_idx']) == n_nodes
    
    # Check label encoding
    label_map = {'entails': 1, 'contradicts': -1, 'neutral': 0}
    for label, encoded in zip(sample_data['labels'], data['labels']):
        assert encoded == label_map[label]


