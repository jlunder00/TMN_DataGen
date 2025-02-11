# TMN_DataGen/TMN_DataGen/utils/text_preprocessing.py
from abc import ABC, abstractmethod
import unicodedata
import re
from typing import List

class BasePreprocessor:
    """Base text preprocessing with configurable strictness"""
    
    def __init__(self, config):
        self.config = config
        self.strictness = config.preprocessing.strictness_level
        
    def preprocess(self, text: str) -> str:
        """Apply preprocessing based on strictness level"""
        if self.strictness == 0:
            return text
            
        # Basic (level 1)
        if self.strictness >= 1:
            text = self._basic_cleanup(text)
            
        # Medium (level 2) 
        if self.strictness >= 2:
            text = self._medium_cleanup(text)
            
        # Strict (level 3)
        if self.strictness >= 3:
            text = self._strict_cleanup(text)
            
        return text
        
    def _basic_cleanup(self, text: str) -> str:
        """Basic normalization"""
        # Normalize whitespace
        # text = re.sub(r'\s+', ' ', text)
        # text = text.strip()
    
        text = ' '.join(text.split())
        
        if self.config.preprocessing.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        if not self.config.preprocessing.preserve_case:
            text = text.lower()
            
        text = ' '.join(text.split())
        return text
        
    def _medium_cleanup(self, text: str) -> str:
        """Medium level cleanup"""
        if self.config.preprocessing.normalize_unicode:
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text)
            # Remove non-ASCII
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            
        return text
        
    def _strict_cleanup(self, text: str) -> str:
        """Strict cleanup"""
        # Remove accents
        text = ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
                      
        if not self.config.preprocessing.preserve_case:
            text = text.lower()
            
        return text


class SentenceSplitter:
    """Split text into sentences based on common terminators"""
    
    def __init__(self):
        # Basic sentence terminator pattern
        self.terminators = r'[.!?]+'
        self.split_pattern = f'({self.terminators}\\s+)'
        
    def split(self, text: str) -> List[str]:
        """Split text into sentences, preserving terminators"""
        # Split on terminators but keep them
        parts = re.split(self.split_pattern, text.strip())
        
        # Combine parts into complete sentences
        sentences = []
        current = ""
        for part in parts:
            current += part
            if re.search(self.terminators, part):
                sentences.append(current.strip())
                current = ""
                
        # Handle any remaining text
        if current.strip():
            sentences.append(current.strip())
            
        return sentences
