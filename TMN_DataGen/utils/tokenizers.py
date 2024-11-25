# TMN_DataGen/TMN_DataGen/utils/tokenizers.py
from abc import ABC, abstractmethod
import re
import stanza
from typing import List

class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

class RegexTokenizer(BaseTokenizer):
    def __init__(self, config):
        self.config = config
        self.min_len = config.preprocessing.min_token_length
        self.max_len = config.preprocessing.max_token_length
        
    def tokenize(self, text: str) -> List[str]:
        # Simple word boundary tokenization
        tokens = re.findall(r'\b\w+\b', text)
        # Apply length filters
        tokens = [t for t in tokens 
                 if self.min_len <= len(t) <= self.max_len]
        return tokens

class StanzaTokenizer(BaseTokenizer):
    def __init__(self, config):
        self.config = config
        try:
            self.nlp = stanza.Pipeline(
                lang=config.preprocessing.language,
                processors='tokenize',
                use_gpu=True
            )
        except Exception as e:
            raise ValueError(f"Failed to load Stanza: {e}")
            
    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        tokens = [word.text for sent in doc.sentences 
                 for word in sent.words]
        return tokens
