# TMN_DataGen/tests/test_dataset_generator.py
import pytest
from TMN_DataGen import DatasetGenerator
from omegaconf import OmegaConf
import os
import pickle

class TestDatasetGenerator:
    @pytest.fixture
    def base_config(self):
        return OmegaConf.create({
            'verbose': 'normal',
            'visualization': {
                'show_features': True
            }
        })
    
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
    
    def test_dataset_generation(self, sample_data, base_config, tmp_path):
        config = OmegaConf.merge(base_config, {
            'parser': {
                'type': 'multi'
            }
        })
        
        generator = DatasetGenerator(config)
        output_path = tmp_path / "test_dataset.pkl"
        
        generator.generate_dataset(
            sentence_pairs=sample_data["sentence_pairs"],
            labels=sample_data["labels"],
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        with open(output_path, 'rb') as f:
            dataset = pickle.load(f)
        
        assert 'graph_pairs' in dataset
        assert 'labels' in dataset
        assert len(dataset['graph_pairs']) == len(sample_data["labels"])
        
        for graph_pair in dataset['graph_pairs']:
            graph1, graph2 = graph_pair
            assert all(key in graph1 for key in ['node_features', 'edge_features', 'from_idx', 'to_idx'])
            assert all(key in graph2 for key in ['node_features', 'edge_features', 'from_idx', 'to_idx'])
    
    def test_verbosity_levels(self, sample_data, base_config, tmp_path, caplog):
        # Test debug verbosity
        debug_config = OmegaConf.merge(base_config, {'verbose': 'debug'})
        generator = DatasetGenerator(debug_config)
        debug_path = tmp_path / "debug_dataset.pkl"
        
        generator.generate_dataset(
            sentence_pairs=[sample_data["sentence_pairs"][0]],
            labels=[sample_data["labels"][0]],
            output_path=str(debug_path)
        )
        
        assert any("Raw parser output" in msg for msg in caplog.messages)
        
        caplog.clear()
        
        # Test normal verbosity
        normal_config = OmegaConf.merge(base_config, {'verbose': 'normal'})
        generator = DatasetGenerator(normal_config)
        normal_path = tmp_path / "normal_dataset.pkl"
        
        generator.generate_dataset(
            sentence_pairs=[sample_data["sentence_pairs"][0]],
            labels=[sample_data["labels"][0]],
            output_path=str(normal_path)
        )
        
        assert any("Processing" in msg for msg in caplog.messages)
        assert not any("Raw parser output" in msg for msg in caplog.messages)

