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


