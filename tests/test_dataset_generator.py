# TMN_DataGen/tests/test_dataset_generator.py
import pytest
from TMN_DataGen import DatasetGenerator
from omegaconf import OmegaConf
import os
import pickle

class TestDatasetGenerator:
    @pytest.fixture
    def sample_data(self):
        return {
            "sentence_pairs": [
                ("The cat chases the mouse.", "The mouse is being chased by the cat."),
                ("The dog barks.", "The cat meows."),
                ("Birds fly in the sky.", "The birds are flying.")
            ],
            "labels": ["entails", "neutral", "entails"]
        }
    
    @pytest.fixture
    def config(self):
        return OmegaConf.load('configs/multi_parser_config.yaml')
    
    def test_dataset_generation(self, sample_data, config, tmp_path):
        generator = DatasetGenerator(config)
        output_path = tmp_path / "test_dataset.pkl"
        
        generator.generate_dataset(
            sentence_pairs=sample_data["sentence_pairs"],
            labels=sample_data["labels"],
            output_path=str(output_path)
        )
        
        # Check if file was created
        assert output_path.exists()
        
        # Load and verify dataset
        with open(output_path, 'rb') as f:
            dataset = pickle.load(f)
        
        assert 'graph_pairs' in dataset
        assert 'labels' in dataset
        assert len(dataset['graph_pairs']) == len(sample_data["labels"])
        
        # Check graph structure
        for graph_pair in dataset['graph_pairs']:
            graph1, graph2 = graph_pair
            assert all(key in graph1 for key in ['node_features', 'edge_features', 'from_idx', 'to_idx'])
            assert all(key in graph2 for key in ['node_features', 'edge_features', 'from_idx', 'to_idx'])
    
    def test_label_mapping(self, sample_data, config):
        generator = DatasetGenerator(config)
        
        # Test if labels are properly mapped
        assert generator.label_map["entails"] == 1
        assert generator.label_map["contradicts"] == -1
        assert generator.label_map["neutral"] == 0
