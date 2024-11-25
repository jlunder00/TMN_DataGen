# tests/test_dataset_generator.py
import pytest
from TMN_DataGen import DatasetGenerator
import pickle

def test_dataset_generation(sample_data, tmp_path):
    """Test basic dataset generation workflow"""
    # Initialize generator
    generator = DatasetGenerator()
    output_path = tmp_path / "test_dataset.pkl"
    
    # Generate dataset
    generator.generate_dataset(
        sentence_pairs=sample_data['sentence_pairs'],
        labels=sample_data['labels'],
        output_path=str(output_path)
    )
    
    # Verify output
    assert output_path.exists()
    with open(output_path, 'rb') as f:
        dataset = pickle.load(f)
    assert 'graph_pairs' in dataset
    assert 'labels' in dataset
    assert len(dataset['graph_pairs']) == len(sample_data['labels'])

def test_config_override(sample_data, tmp_path):
    """Test config override functionality"""
    generator = DatasetGenerator()
    output_path = tmp_path / "test_dataset.pkl"
    
    parser_config = {
        'parser': {
            'type': 'multi',
            'feature_sources': {
                'tree_structure': 'diaparser',
                'pos_tags': 'spacy'
            }
        }
    }
    
    # Should work with custom config
    generator.generate_dataset(
        sentence_pairs=sample_data['sentence_pairs'],
        labels=sample_data['labels'],
        output_path=str(output_path),
        parser_config=parser_config,
        verbosity='debug'
    )
