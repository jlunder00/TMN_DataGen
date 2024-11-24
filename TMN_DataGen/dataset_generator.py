# TMN_DataGen/TMN_DataGen/dataset_generator.py
from typing import List, Tuple, Optional, Dict
from omegaconf import DictConfig
from .parsers import DiaParserTreeParser, SpacyTreeParser, MultiParser
from .tree import DependencyTree
from .utils.viz_utils import format_tree_pair
from .utils.logging_config import logger
import torch
import numpy as np
from tqdm import tqdm
import pickle
from pathlib import Path


class DatasetGenerator:
    def __init__(self, config: Optional[DictConfig] = None):
        self.config = config or {}
        self.verbose = self.config.get('verbose', False)
        parser_type = self.config.get('parser', {}).get('type', 'diaparser')
        
        if 'parser' not in self.config:
            self.config['parser'] = {}
        self.config['parser']['verbose'] = self.verbose
        
        parser_class = {
            "diaparser": DiaParserTreeParser,
            "spacy": SpacyTreeParser,
            "multi": MultiParser
        }[parser_type]
        self.parser = parser_class(self.config)
        
        self.label_map = {
            'entails': 1,
            'contradicts': -1,
            'neutral': 0
        }
        if self.verbose:
            logger.info("Initialized DatasetGenerator with verbose output")
    
    def generate_dataset(self, sentence_pairs: List[Tuple[str, str]], 
                        labels: List[str], 
                        output_path: str,
                        show_progress: bool = True) -> None:
        """Generate dataset from sentence pairs and labels"""
        if self.verbose:
            logger.info("\nGenerating dataset...")
            logger.info(f"Processing {len(sentence_pairs)} sentence pairs")

        all_sentences = [s for pair in sentence_pairs for s in pair]
        
        logger.info("Parsing sentences...")
        all_trees = self.parser.parse_all(all_sentences, show_progress)
        
        # Pair up trees
        tree_pairs = [
            (all_trees[i], all_trees[i+1]) 
            for i in range(0, len(all_trees), 2)
        ]
        
        if self.verbose:
            logger.info("\nGenerated tree pairs:")
            for (tree1, tree2), label in zip(tree_pairs, labels):
                logger.info("\n" + "=" * 80)
                logger.info(format_tree_pair(tree1, tree2, label))
                logger.info("=" * 80)
        
        logger.info("Converting to GMN format...")
        dataset = self._convert_to_gmn_format(tree_pairs, labels)
        
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)

        if self.verbose:
            logger.info(f"\nDataset saved to {output_path}")
    
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
