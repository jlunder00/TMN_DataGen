# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# tests/test_preprocessing.py
from TMN_DataGen.utils import BasePreprocessor
from TMN_DataGen.utils import RegexTokenizer, StanzaTokenizer
import pytest

def test_basic_preprocessing(default_config):
    """Test basic preprocessing features"""
    config, _ = default_config
    preprocessor = BasePreprocessor(config)
    assert preprocessor.preprocess("Hello,   World! ") == "hello world"

def test_unicode_handling(default_config):
    """Test unicode normalization"""
    config, _ = default_config
    config.preprocessing.strictness_level = 2
    config.preprocessing.normalize_unicode = True
    preprocessor = BasePreprocessor(config)
    assert preprocessor.preprocess("café") == "cafe"

@pytest.mark.skipif(not pytest.importorskip("stanza"), reason="stanza not installed")  
def test_stanza_tokenizer(default_config):
    """Test Stanza tokenizer if available"""
    config, _ = default_config
    config.preprocessing.tokenizer = 'stanza'
    tokenizer = StanzaTokenizer(config)
    tokens = tokenizer.tokenize("Hello, world!")
    assert tokens == ['Hello', ',', 'world', '!']
