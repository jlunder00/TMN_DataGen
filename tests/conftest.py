# tests/conftest.py
import pytest
import yaml
from pathlib import Path

# @pytest.fixture
# def base_config():
#     """Load default configs for testing"""
#     config_dir = Path(__file__).parent.parent / 'TMN_DataGen' / 'configs'
#     
#     with open(config_dir / 'default_parser_config.yaml') as f:
#         config = yaml.safe_load(f)
#         
#     with open(config_dir / 'default_preprocessing_config.yaml') as f:
#         config.update(yaml.safe_load(f))
#         
#     config['verbose'] = 'debug' # Use debug for tests
#     return config

@pytest.fixture 
def sample_data():
    """Sample text data for testing"""
    return {
        'sentence_pairs': [
            ('The cat chases the mouse.', 'The mouse is being chased by the cat.'),
            ('The dog barks.', 'The cat meows.'),
            ('Birds fly in the sky.', 'The birds are on the ground.')
        ],
        'labels': ['entails', 'neutral', 'contradicts']
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
