# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

# TMN_DataGen/TMN_DataGen/dataset_generator.py
from typing import List, Tuple, Optional, Dict, Union, NamedTuple
from pathlib import Path
import yaml
import json
from omegaconf import OmegaConf
from .parsers import DiaParserTreeParser, SpacyTreeParser, MultiParser
from .tree import DependencyTree
from .utils.text_preprocessing import SentenceSplitter, BasePreprocessor
from .utils.viz_utils import format_tree_pair
from .utils.logging_config import setup_logger
from importlib.resources import files
import uuid
from english_words import get_english_words_set
from gensim.models import KeyedVectors

class TreeGroup(NamedTuple):
    """Helper class for tracking tree groups"""
    group_id: str
    original_text: str
    trees: List[DependencyTree]
    original_text_b: str
    trees_b : List[DependencyTree]
    label : str

def generate_group_id():
    """Generate unique ID for groups"""
    return str(uuid.uuid4())

class DatasetGenerator:
    def __init__(self, num_workers=1):
        """Initialize dataset generator without config - config provided per method call"""
        self.label_map = None
        self.sentence_splitter = SentenceSplitter()
        self.num_workers = num_workers
        self.vocabs = []

    def _load_configs(
        self,
        parser_config: Optional[Union[str, Dict]] = None,
        preprocessing_config: Optional[Union[str, Dict]] = None,
        feature_config: Optional[Union[str, Dict]] = None,
        output_config: Optional[Union[str, Dict]] = None,
        merge_config: Optional[Union[str, Dict]] = None,
        verbosity: str = 'normal',
        override_pkg_config: Optional[Union[str, Dict]] = None
    ) -> Dict:
        """Load and merge configurations"""
        config_dir = Path(files('TMN_DataGen').joinpath('configs'))
        # config_dir = Path(__file__).parent / 'configs'
        
        # Load default configs
        with open(config_dir / 'default_package_config.yaml') as f:
            pkg_config = yaml.safe_load(f)
            
        with open(config_dir / 'default_parser_config.yaml') as f:
            config = yaml.safe_load(f)
            
        with open(config_dir / 'default_preprocessing_config.yaml') as f:
            config.update(yaml.safe_load(f))

        with open(config_dir / 'default_feature_config.yaml') as f:
            config.update(yaml.safe_load(f))

        with open(config_dir / 'default_output_format.yaml') as f:
            config.update(yaml.safe_load(f))

        with open(config_dir / 'default_merge_config.yaml') as f:
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

        # Override feature config if provided
        if feature_config:
            if isinstance(feature_config, str):
                with open(feature_config) as f:
                    feature_config = yaml.safe_load(f)
            for key in ['feature_extraction', 'feature_mappings']:
                if key in feature_config:
                    config[key].update(feature_config[key])

        if output_config:
            if isinstance(output_config, str):
                with open(output_config) as f:
                    output_config = yaml.safe_load(f)
            config['output_format'].update(output_config)

        if merge_config:
            if isinstance(merge_config, str):
                with open(merge_config) as f:
                    merge_config = yaml.safe_load(f)
            config['merge'].update(merge_config)

        return OmegaConf.create(config), pkg_config

    def generate_dataset(
        self,
        text_pairs: List[Tuple[str, str]],
        labels: List[str],
        output_path: str,
        parser_config: Optional[Union[str, Dict]] = None,
        preprocessing_config: Optional[Union[str, Dict]] = None,
        feature_config: Optional[Union[str, Dict]] = None,
        output_config: Optional[Union[str, Dict]] = None,
        merge_config: Optional[Union[str, Dict]] = None,
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
            feature_config,
            output_config,
            merge_config,
            verbosity,
            override_pkg_config
        )
        self.logger = setup_logger(
                self.__class__.__name__,
                verbosity
                )

        self.config = config
        if self.config.output_format.label_map is not None:
            self.label_map = self.config.output_format.label_map

        if not self.vocabs and self.config.preprocessing.tokenizer == 'vocab':
            vocabs = []
            vocab_model =  KeyedVectors.load_word2vec_format(self.config.preprocessing.get('vocab_model_path', '/home/jlunder/research/data/word2vec_model/GoogleNews-vectors-negative300.bin'), binary=True, limit=self.config.preprocessing.get('vocab_limit', 500000)) #take only top n common words 
            vocabs.append(vocab_model.index_to_key)
            del vocab_model
            all_words = set()
            all_words_lower = get_english_words_set(['web2', 'gcide'], lower=True)
            all_words = all_words.union(all_words_lower)
            all_words_standard = get_english_words_set(['web2', 'gcide'])
            all_words = all_words.union(all_words_standard)
            all_words_alpha_standard = get_english_words_set(['web2', 'gcide'], alpha=True)
            all_words = all_words.union(all_words_alpha_standard)
            all_words_alpha_lower = get_english_words_set(['web2', 'gcide'], alpha=True, lower=True)
            all_words = all_words.union(all_words_alpha_lower)
            vocabs.append(all_words)
            self.vocabs = vocabs
        else:
            self.vocabs = []

        # Initialize parser
        parser = MultiParser(config, pkg_config, self.vocabs, self.logger)


        # Process sentence pairs
        if verbosity != 'quiet':
            self.logger.info(f"\nGenerating dataset...")
            self.logger.info(f"Processing {len(text_pairs)} text pairs")

        is_paired = self.config.output_format and self.config.output_format.get('paired', False)
        self_paired = self.config.output_format and self.config.output_format.get('self_paired', False)
        self.preprocessor = BasePreprocessor(self.config)

        # Split sentences and track groups
        sentence_groups = []
        group_metadata = []
        
        import time
        start = time.time()
        for i, (text1, text2) in enumerate(text_pairs):
            # Split into sentences
            text2_clean = text2
            text1_clean = self.preprocessor.preprocess(text1)
            group1 = self.sentence_splitter.split(text1_clean)
            if is_paired:
                text2_clean = self.preprocessor.preprocess(text2)
                group2 = self.sentence_splitter.split(text2_clean)
            
            # Create group metadata
            group_id = generate_group_id()
            metadata = {
                'group_id': group_id,
                'text': text1,
                'text_clean': text1_clean,
                'text_b': text2,
                'text_b_clean': text2_clean,
                'label' : labels[i]
            }
            group_metadata.append(metadata)
            
            to_add = [group1]
            if is_paired:
                to_add.append(group2)
            sentence_groups.extend(to_add)
        print(f"preprocess took {(time.time()-start)}")

        # Parse all sentences
        all_tree_groups = parser.parse_all(sentence_groups, show_progress, num_workers=self.num_workers)

        # Organize trees with groups
        tree_groups = []
        group_idx = 0
        for i, meta in enumerate(group_metadata):
            # if i + 1 >= len(all_tree_groups):
            #     continue
                
            group = TreeGroup(
                group_id=meta['group_id'],
                original_text=meta['text'],
                trees=all_tree_groups[i*2],
                original_text_b= '' if not is_paired else meta['text_b'],
                trees_b = [] if not is_paired else all_tree_groups[i*2+1],
                label = meta['label']
            )
            tree_groups.append(group)

        # Convert based on format
        if self.config.output_format.type == "infonce":
            dataset = self._convert_to_infonce_format(tree_groups, is_paired, self_paired)
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=4)
        # else:
        #     # Convert tree groups to pairs for backwards compatibility
        #     valid_pairs = []
        #     valid_labels = []
        #     for group1, group2 in tree_groups:
        #         for t1 in group1.trees:
        #             for t2 in group2.trees:
        #                 if t1 is not None and t2 is not None:
        #                     valid_pairs.append((t1, t2))
        #                     valid_labels.append(labels[len(valid_pairs)-1])
        #     dataset = self._convert_to_gmn_format(valid_pairs, valid_labels)
        #     with open(output_path, 'w') as f:
        #         json.dump(dataset, f, indent=4)


    def _convert_to_infonce_format(
        self,
        tree_groups: List[TreeGroup],
        is_paired=False,
        self_paired=False
    ) -> Dict:
        """Convert to InfoNCE format with group tracking"""
        groups = []
        
        for group in tree_groups:
            # Convert all trees to graph format
            trees1 = [
                t.to_graph_data() for t in group.trees 
                if t is not None
            ]
            if is_paired:
                trees2 = [
                    t.to_graph_data() for t in group.trees_b
                    if t is not None
                ]
            
            add = (not is_paired and trees1) or (is_paired and trees1 and trees2) or (self_paired and trees1)

            if add:  # Only add if both have valid trees
                group_data = {
                    "group_id": group.group_id,
                    "text": group.original_text,
                    "trees": trees1,
                    "label": group.label
                }
                if is_paired:
                    group_data['text_b'] = group.original_text_b
                    group_data['trees_b'] = trees2
                groups.append(group_data)

        return {
            "version": "1.0",
            "format": "infonce",
            "requires_word_embeddings": self.config.feature_extraction.do_not_store_word_embeddings,
            "groups": groups
        }

        # sentence_pairs = []
        # for pair in text_pairs:
        #     pair_groups = []
        #     for text in pair:
        #         sentences = self.sentence_splitter.split(text)
        #         pair_groups.append(sentences)
        #     # pair_groups = self.filter_groups()
        #     sentence_pairs.append((pair_groups[0], pair_groups[1]))

        # all_sentence_groups = [s for pair in sentence_pairs for s in pair]
        # 
        # self.logger.info("Parsing sentences...")
        # all_tree_groups = parser.parse_all(all_sentence_groups, show_progress)
        # 

        # valid_pairs = []
        # valid_labels = []
        # # Pair up trees
        # # tree_pairs = [
        # #     (all_trees[i], all_trees[i+1]) 
        # #     for i in range(0, len(all_trees), 2)
        # # ]
        # all_trees = []
        # for i in range(0, len(all_tree_groups), 2):
        #     if i + 1 >= len(all_tree_groups):
        #         self.logger.warning(f"Uneven number of tree groups: {len(all_tree_groups)}")
        #         break
        #         
        #     # Get the pair of trees
        #     tree_group1 = all_tree_groups[i]
        #     tree_group2 = all_tree_groups[i+1]
        #     
        #     for tree1 in tree_group1:
        #         for tree2 in tree_group2:
        #             all_trees.append(tree1)
        #             all_trees.append(tree2)

        # for i in range(0, len(all_trees), 2):
        #     # Skip if either tree is None
        #     if tree1 is None or tree2 is None:
        #         self.logger.debug(f"Skipping pair {i//2} - missing tree")
        #         continue
        #         
        #     pair_idx = i // 2
        #     if pair_idx >= len(labels):
        #         self.logger.error(f"Label index {pair_idx} out of range for {len(labels)} labels")
        #         break
        #         
        #     valid_pairs.append((tree1, tree2))
        #     valid_labels.append(labels[pair_idx])

        # if verbosity == 'debug':
        #     self.logger.info("\nGenerated tree pairs:")
        #     # for (tree1, tree2), label in zip(tree_pairs, labels):
        #     for (tree1, tree2), label in zip(valid_pairs, valid_labels):
        #         self.logger.info("\n" + "=" * 80)
        #         self.logger.info(format_tree_pair(tree1, tree2, label))
        #         self.logger.info("=" * 80)

        # if not valid_pairs:
        #     raise ValueError("No valid tree pairs produced")

        # self.logger.info(f"Generated {len(valid_pairs)} valid pairs from {len(sentence_pairs)} original pairs")
        # # Convert and save
        # # dataset = self._convert_to_gmn_format(tree_pairs, labels)
        # dataset = self._convert_to_gmn_format(valid_pairs, valid_labels)
        # with open(output_path, 'w') as f:
        #     json.dump(dataset, f, indent=4)

        # if verbosity != 'quiet':
        #     self.logger.info(f"\nDataset saved to {output_path}")

    def _convert_to_gmn_format(self, 
                              tree_pairs: List[Tuple[DependencyTree, DependencyTree]], 
                              labels: List[str]) -> Dict:
        """Convert tree pairs to GMN-compatible format"""
        graph_pairs = []
        numeric_labels = []
        
        skipped_count = 0
        for (tree1, tree2), label in zip(tree_pairs, labels):
            # Skip pairs with no majority label
            if label == '-':
                skipped_count += 1
                continue
                

            try:
                # Handle valid labels
                if self.label_map is not None and label not in self.label_map:
                    self.logger.error(f"Invalid label '{label}' encountered. Expected one of: {list(self.label_map.keys())}")
                    raise ValueError(f"Invalid label: {label}")

                label = self.label_map[label] if self.label_map is not None else label
                if self.config.output_format.normalize is not None:
                    label = (float(label) - self.config.output_format.normalize.min) / self.config.output_format.normalize.max

            except Exception as e:
                self.logger.error(f"Error while processing labels: {e}, exiting")
                raise e 
                
            graph1 = tree1.to_graph_data()
            graph2 = tree2.to_graph_data()
            graph_pairs.append((graph1, graph2))
            numeric_labels.append(label)
        
        if skipped_count > 0:
            skip_percent = (skipped_count / len(labels)) * 100
            self.logger.info(f"Skipped {skipped_count} pairs ({skip_percent:.1f}%) due to annotator disagreement")
            if skip_percent > 5:  # Arbitrary threshold
                self.logger.warning(f"High skip rate ({skip_percent:.1f}%) may indicate data quality issues")
        
        return {
            'graph_pairs': graph_pairs,
            'labels': numeric_labels
        }
