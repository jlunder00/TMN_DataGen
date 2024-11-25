# tests/test_preprocessing.py
from TMN_DataGen.utils import BasePreprocessor
from TMN_DataGen.utils import RegexTokenizer, StanzaTokenizer
import pytest

def test_basic_preprocessing():
    """Test basic preprocessing features"""
    config = {
        'preprocessing': {
            'strictness_level': 1,
            'tokenizer': 'regex',
            'preserve_case': False
        }
    }
    
    preprocessor = BasePreprocessor(config)
    assert preprocessor.preprocess("Hello,   World! ") == "hello world"
    
def test_unicode_handling():
    """Test unicode normalization"""
    config = {
        'preprocessing': {
            'strictness_level': 2,
            'normalize_unicode': True
        }
    }
    
    preprocessor = BasePreprocessor(config)
    assert preprocessor.preprocess("café") == "cafe"

@pytest.mark.skipif(not pytest.importorskip("stanza"), reason="stanza not installed")
def test_stanza_tokenizer():
    """Test Stanza tokenizer if available"""
    config = {
        'preprocessing': {
            'tokenizer': 'stanza',
            'language': 'en'
        }
    }
    
    tokenizer = StanzaTokenizer(config)
    tokens = tokenizer.tokenize("Hello, world!")
    assert tokens == ['Hello', ',', 'world', '!']
