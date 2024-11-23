#dataset_generator.py
from typing import List, Tuple, Optional, Dict
from omegaconf import DictConfig
from .parsers import DiaParserTreeParser, SpacyTreeParser, MultiParser
from .tree import DependencyTree
import torch
import numpy as np
from tqdm import tqdm
import json
import pickle
from pathlib import Path

class DatasetGenerator:
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or {}
        parser_type = self.config.get('parser', {}).get('type', 'diaparser')
        self.parser = {
            "diaparser": DiaParserTreeParser,
            "spacy": SpacyTreeParser,
            "multi": MultiParser
        }[parser_type]
        
        self.label_map = {
            'entails': 1,
            'contradicts': -1,
            'neutral': 0
        }
    
    def generate_dataset(self, 
                        sentence_pairs: List[Tuple[str, str]], 
                        labels: List[str],
                        output_path: str,
                        show_progress: bool = True):
        """Generate and save dataset in GMN-compatible format"""
        # Parse all sentences
        all_sentences = []
        for premise, hypothesis in sentence_pairs:
            all_sentences.extend([premise, hypothesis])
            
        print("Parsing sentences...")
        all_trees = self.parser.parse_all(all_sentences, show_progress)
        
        # Pair trees
        tree_pairs = []
        for i in range(0, len(all_trees), 2):
            tree_pairs.append((all_trees[i], all_trees[i+1]))
        
        # Convert to GMN format
        print("Converting to GMN format...")
        dataset = self._convert_to_gmn_format(tree_pairs, labels)
        
        # Save dataset
        print(f"Saving dataset to {output_path}")
        self._save_dataset(dataset, output_path)
    
    def _convert_to_gmn_format(self, 
                              tree_pairs: List[Tuple[DependencyTree, DependencyTree]],
                              labels: List[str]) -> Dict:
        """Convert tree pairs to GMN-compatible format"""
        graph_pairs = []
        numeric_labels = []
        
        for (tree1, tree2), label in zip(tree_pairs, labels):
            graph1 = tree1.to_graph_data()
            graph2 = tree2.to_graph_data()
            graph_pairs.append((graph1, graph2))
            numeric_labels.append(self.label_map[label])
        
        return {
            'graph_pairs': graph_pairs,
            'labels': numeric_labels
        }
    
    def _save_dataset(self, dataset: Dict, output_path: str):
        """Save dataset to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
