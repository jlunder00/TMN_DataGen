# TMN_DataGen/TMN_DataGen/dataset_generator.py
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import yaml
import pickle
from omegaconf import OmegaConf
from .parsers import DiaParserTreeParser, SpacyTreeParser, MultiParser
from .tree import DependencyTree
from .utils.viz_utils import format_tree_pair
from .utils.logging_config import logger

class DatasetGenerator:
    def __init__(self):
        """Initialize dataset generator without config - config provided per method call"""
        self.label_map = {
            'entails': 1,
            'contradicts': -1,
            'neutral': 0
        }


    def _load_configs(
        self,
        parser_config: Optional[Union[str, Dict]] = None,
        preprocessing_config: Optional[Union[str, Dict]] = None, 
        verbosity: str = 'normal',
        override_pkg_config: Optional[Union[str, Dict]] = None
    ) -> Dict:
        """Load and merge configurations"""
        config_dir = Path(__file__).parent / 'configs'
        
        # Load default configs
        with open(config_dir / 'default_package_config.yaml') as f:
            pkg_config = yaml.safe_load(f)
            
        with open(config_dir / 'default_parser_config.yaml') as f:
            config = yaml.safe_load(f)
            
        with open(config_dir / 'default_preprocessing_config.yaml') as f:
            config.update(yaml.safe_load(f))
            
        # Add verbosity
        config['verbose'] = verbosity
            
        # Override package config if provided
        if override_pkg_config:
            if isinstance(override_pkg_config, str):
                with open(override_pkg_config) as f:
                    pkg_config = yaml.safe_load(f)
            else:
                pkg_config = override_pkg_config
                
        # Override parser config if provided
        if parser_config:
            if isinstance(parser_config, str):
                with open(parser_config) as f:
                    parser_config = yaml.safe_load(f)
            config['parser'].update(parser_config)
            
        # Override preprocessing config if provided
        if preprocessing_config:
            if isinstance(preprocessing_config, str):
                with open(preprocessing_config) as f:
                    preprocessing_config = yaml.safe_load(f)
            config['preprocessing'].update(preprocessing_config)

        return OmegaConf.create(config), pkg_config

    def generate_dataset(
        self,
        sentence_pairs: List[Tuple[str, str]],
        labels: List[str],
        output_path: str,
        parser_config: Optional[Union[str, Dict]] = None,
        preprocessing_config: Optional[Union[str, Dict]] = None,
        verbosity: str = 'normal',
        override_pkg_config: Optional[Union[str, Dict]] = None,
        show_progress: bool = True
    ) -> None:
        """
        Generate parsed dataset from sentence pairs

        Args:
            sentence_pairs: List of sentence pairs to parse
            labels: List of labels for each pair 
            output_path: Where to save the dataset
            parser_config: Parser config as dict or path to yaml
            preprocessing_config: Preprocessing config as dict or path
            verbosity: Logging verbosity ('quiet', 'normal', 'debug')
            override_pkg_config: Optional override for package capabilities
            show_progress: Show progress bar during processing
        """
        # Load and merge configs
        config, pkg_config = self._load_configs(
            parser_config,
            preprocessing_config,
            verbosity,
            override_pkg_config
        )
        self.logger = setup_logger(
                self.__class__.__name__,
                verbosity
                )

        # Initialize parser
        parser = MultiParser(config, pkg_config)

        # Process sentence pairs
        if verbosity != 'quiet':
            logger.info("\nGenerating dataset...")
            logger.info(f"Processing {len(sentence_pairs)} sentence pairs")

        all_sentences = [s for pair in sentence_pairs for s in pair]
        
        logger.info("Parsing sentences...")
        all_trees = parser.parse_all(all_sentences, show_progress)
        
        # Pair up trees
        tree_pairs = [
            (all_trees[i], all_trees[i+1]) 
            for i in range(0, len(all_trees), 2)
        ]

        if verbosity == 'debug':
            logger.info("\nGenerated tree pairs:")
            for (tree1, tree2), label in zip(tree_pairs, labels):
                logger.info("\n" + "=" * 80)
                logger.info(format_tree_pair(tree1, tree2, label))
                logger.info("=" * 80)

        # Convert and save
        dataset = self._convert_to_gmn_format(tree_pairs, labels)
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)

        if verbosity != 'quiet':
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
