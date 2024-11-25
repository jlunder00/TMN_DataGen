# tests/conftest.py
import pytest
import yaml
from pathlib import Path
from omegaconf import OmegaConf

@pytest.fixture
def default_config():
    """Load all default configs for testing"""
    config_dir = Path(__file__).parent.parent / 'TMN_DataGen' / 'configs'
    
    # Load package config
    with open(config_dir / 'default_package_config.yaml') as f:
        pkg_config = yaml.safe_load(f)
        
    # Load and merge parser and preprocessing configs
    with open(config_dir / 'default_parser_config.yaml') as f:
        config = yaml.safe_load(f)
        
    with open(config_dir / 'default_preprocessing_config.yaml') as f:
        preproc = yaml.safe_load(f)
        config.update(preproc)
        
    return OmegaConf.create(config), pkg_config

@pytest.fixture
def sample_data():
    """Sample text data for testing"""
    return {
        'sentence_pairs': [
            ('The cat chases the mouse.', 'The mouse is being chased by the cat.'),
            ('The dog barks.', 'The cat meows.')
        ],
        'labels': ['entails', 'neutral']
    }

@pytest.fixture
def unicode_data():
    return {
        'sentence_pairs': [
            ('The café is nice!', 'It is a coffee shop.'),
            ('こんにちは世界', 'Hello world')  
        ],
        'labels': ['entails', 'neutral']
    }
