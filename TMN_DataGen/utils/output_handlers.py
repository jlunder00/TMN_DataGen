# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#Unused: planned to be integrated in a future update to offload some of the logic of output preparation

# TMN_DataGen/utils/output_handlers.py
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from pathlib import Path

class ContrastiveLossType(Enum):
    TRIPLET = "triplet"
    INFONCE = "infonce" 
    SUPERVISED = "supervised"

class DataOrganizer(ABC):
    """Base class for organizing data into various contrastive formats"""
    def __init__(self, config):
        self.config = config
        
    @abstractmethod
    def organize_pairs(self, pairs: List[Dict]) -> Dict:
        """Organize data into format for specific loss type"""
        pass
        
    @abstractmethod
    def get_output_format(self) -> Dict:
        """Return metadata about output format"""
        pass

class TripletOrganizer(DataOrganizer):
    """Organizes data for triplet loss"""
    def organize_pairs(self, pairs: List[Dict]) -> Dict:
        organized = {
            "triplets": []
        }
        # Form triplets from pairs
        for anchor_pair in pairs:
            positives = self._get_positive_pairs(anchor_pair, pairs)
            negatives = self._get_negative_pairs(anchor_pair, pairs)
            
            for pos in positives:
                for neg in negatives:
                    organized["triplets"].append({
                        "anchor": anchor_pair["graph1"],
                        "positive": pos["graph1"], 
                        "negative": neg["graph1"]
                    })
        return organized

    def get_output_format(self) -> Dict:
        return {
            "type": "triplet",
            "format_version": "1.0",
            "requires_word_embeddings": True
        }

