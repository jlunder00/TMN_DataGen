# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# tests/conftest.py
import pytest
import yaml
from pathlib import Path
from omegaconf import OmegaConf

@pytest.fixture
def default_config():
    """Load all default configs for testing"""
    config_dir = Path(__file__).parent.parent / 'TMN_DataGen' / 'configs'
    
    # Load configs
    with open(config_dir / 'default_package_config.yaml') as f:
        pkg_config = yaml.safe_load(f)
        
    with open(config_dir / 'default_parser_config.yaml') as f:
        config = yaml.safe_load(f)
        
    with open(config_dir / 'default_preprocessing_config.yaml') as f:
        preproc = yaml.safe_load(f)
        
    with open(config_dir / 'default_feature_config.yaml') as f:
        feature_config = yaml.safe_load(f)

    # Merge configs
    config.update(preproc)
    config.update(feature_config)
    
    # # Override some settings for testing
    # config['feature_extraction'].update({
    #     'use_gpu': False,
    #     'cache_embeddings': False  
    # })
    
    return OmegaConf.create(config), pkg_config

@pytest.fixture
def sample_data():
    """Sample text data for testing"""
    return {
        'sentence_pairs': [
            ('The cat chases the mouse.', 'The mouse is being chased by the cat.'),
            ('The dog barks.', 'The cat meows.')
        ],
        'labels': ['entailment', 'neutral']
    }

@pytest.fixture
def unicode_data():
    return {
        'sentence_pairs': [
            ('The café is nice!', 'It is a coffee shop.'),
            ('こんにちは世界', 'Hello world')  
        ],
        'labels': ['entailment', 'neutral']
    }
