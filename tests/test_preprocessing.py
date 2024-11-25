#tests/test_preprocessing.py
import pytest
from TMN_DataGen.utils.text_preprocessing import BasePreprocessor
from TMN_DataGen.utils.tokenizers import RegexTokenizer, StanzaTokenizer
from omegaconf import OmegaConf

class TestPreprocessing:
    @pytest.fixture
    def config(self):
        return OmegaConf.create({
            'preprocessing': {
                'strictness_level': 1,
                'tokenizer': 'regex',
                'language': 'en',
                'preserve_case': False,
                'remove_punctuation': True,
                'normalize_unicode': True,
                'remove_numbers': False,
                'max_token_length': 50,
                'min_token_length': 1
            }
        })

    def test_strictness_levels(self, config):
        # Test STRICTNESS_NONE
        config.preprocessing.strictness_level = 0
        preprocessor = BasePreprocessor(config)
        text = "Hello, World! こんにちは"
        assert preprocessor.preprocess(text) == text

        # Test STRICTNESS_BASIC
        config.preprocessing.strictness_level = 1
        preprocessor = BasePreprocessor(config)
        result = preprocessor.preprocess("Hello,   World! ")
        assert result == "Hello World"

        # Test STRICTNESS_MEDIUM
        config.preprocessing.strictness_level = 2
        preprocessor = BasePreprocessor(config)
        result = preprocessor.preprocess("Hello, World! こんにちは")
        assert result == "Hello World"

        # Test STRICTNESS_STRICT
        config.preprocessing.strictness_level = 3
        preprocessor = BasePreprocessor(config)
        result = preprocessor.preprocess("Héllo, Wörld!")
        assert result == "hello world"

    def test_regex_tokenizer(self, config):
        config.preprocessing.tokenizer = 'regex'
        tokenizer = RegexTokenizer(config)
        
        # Test basic tokenization
        text = "The quick brown fox"
        tokens = tokenizer.tokenize(text)
        assert tokens == ['The', 'quick', 'brown', 'fox']

        # Test length filters
        config.preprocessing.min_token_length = 4
        tokenizer = RegexTokenizer(config)
        tokens = tokenizer.tokenize(text)
        assert tokens == ['quick', 'brown']

    @pytest.mark.skipif(not pytest.importorskip("stanza"), reason="stanza not installed")
    def test_stanza_tokenizer(self, config):
        config.preprocessing.tokenizer = 'stanza'
        tokenizer = StanzaTokenizer(config)
        
        text = "The quick brown fox."
        tokens = tokenizer.tokenize(text)
        assert len(tokens) == 5  # includes period
        assert tokens[:-1] == ['The', 'quick', 'brown', 'fox']
