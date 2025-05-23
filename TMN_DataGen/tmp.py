# TMN_DataGen/dataset_generator.py (Updated sections)
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
from .utils.parallel_framework import ParallelizationMixin, batch_parallel_process, _process_text_pair_worker
from concurrent.futures import ProcessPoolExecutor
import time

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

class DatasetGenerator(ParallelizationMixin):
    def __init__(self, num_workers=1):
        """Initialize dataset generator without config - config provided per method call"""
        self.label_map = None
        self.sentence_splitter = SentenceSplitter()
        self.num_workers = num_workers
        self.vocabs = []
        super().__init__()

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

        # ADD: Load parallel config
        with open(config_dir / 'default_parallel_config.yaml') as f:
            config.update(yaml.safe_load(f))
            
        # Add verbosity
        config['verbose'] = verbosity
            
        # Override configs as before...
        # [Keep all existing override logic unchanged]
        
        return OmegaConf.create(config), pkg_config

    def _create_tree_group_batch(self, metadata_and_trees_batch: List[Tuple[Dict, List, int]],
                                is_paired: bool) -> List[TreeGroup]:
        """Create TreeGroup objects from metadata and parsed trees"""
        tree_groups_batch = []
        
        for metadata, all_tree_groups, i in metadata_and_trees_batch:
            if metadata is None:
                continue
                
            try:
                group = TreeGroup(
                    group_id=metadata['group_id'],
                    original_text=metadata['text'],
                    trees=all_tree_groups[i*2] if i*2 < len(all_tree_groups) else [],
                    original_text_b='' if not is_paired else metadata['text_b'],
                    trees_b=[] if not is_paired else (all_tree_groups[i*2+1] if i*2+1 < len(all_tree_groups) else []),
                    label=metadata['label']
                )
                tree_groups_batch.append(group)
            except Exception as e:
                self.logger.error(f"Error creating tree group for {metadata.get('group_id', 'unknown')}: {e}")
        
        return tree_groups_batch
    
    def _convert_tree_group_to_infonce_batch(self, tree_groups_batch: List[TreeGroup],
                                           is_paired: bool, self_paired: bool) -> List[Dict]:
        """Convert a batch of tree groups to InfoNCE format"""
        groups_batch = []
        
        for group in tree_groups_batch:
            try:
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
                else:
                    trees2 = []
                
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
                    groups_batch.append(group_data)
                    
            except Exception as e:
                self.logger.error(f"Error converting tree group {group.group_id} to InfoNCE: {e}")
        
        return groups_batch

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
        show_progress: bool = True,
        cache_dir = None,
        max_concurrent=None
    ) -> None:
        """Generate parsed dataset from sentence pairs with parallelization"""
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
        self.logger = setup_logger(self.__class__.__name__, verbosity)
        self.config = config
        config.feature_extraction.embedding_cache_dir = cache_dir if cache_dir else config.feature_extraction.embedding_cache_dir
        if self.config.output_format.label_map is not None:
            self.label_map = self.config.output_format.label_map

        # Initialize vocabs if needed
        if not self.vocabs and self.config.preprocessing.tokenizer == 'vocab':
            # [Keep existing vocab initialization logic unchanged]
            pass

        # Initialize parser
        parser = MultiParser(config, pkg_config, self.vocabs, self.logger, max_concurrent, self.num_workers)

        # Process sentence pairs
        if verbosity != 'quiet':
            self.logger.info(f"\nGenerating dataset...")
            self.logger.info(f"Processing {len(text_pairs)} text pairs")

        is_paired = self.config.output_format and self.config.output_format.get('paired', False)
        self_paired = self.config.output_format and self.config.output_format.get('self_paired', False)
        self.preprocessor = BasePreprocessor(self.config)

        # PARALLELIZED PREPROCESSING AND SENTENCE SPLITTING
        start = time.time()
        
        if (self.parallel_config.get('preprocessing', True) and 
            len(text_pairs) >= 100 and 
            self.num_workers > 1):
            
            self.logger.info("Using parallel preprocessing and sentence splitting")
            
            # Prepare data with original indices for tracking
            indexed_text_pairs = [(text1, text2, i) for i, (text1, text2) in enumerate(text_pairs)]
            
            # Get configurable chunk size
            chunk_size = self._get_chunk_size('preprocessing', 100, len(indexed_text_pairs))
            
            # Prepare arguments for workers
            chunks = [indexed_text_pairs[i:i + chunk_size] 
                     for i in range(0, len(indexed_text_pairs), chunk_size)]
            
            # Process chunks in parallel using module-level worker function
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                chunk_args = [(chunk, is_paired, labels, 
                              self.config.preprocessing, {}) 
                             for chunk in chunks]
                chunk_results = list(executor.map(_process_text_pair_worker, chunk_args))
            
            # Combine results
            sentence_groups = []
            group_metadata = []
            for sentence_groups_batch, group_metadata_batch in chunk_results:
                sentence_groups.extend(sentence_groups_batch)
                group_metadata.extend(group_metadata_batch)
            
            # Filter out None entries
            group_metadata = [meta for meta in group_metadata if meta is not None]
            
        else:
            # Sequential processing (existing code)
            sentence_groups = []
            group_metadata = []
            
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
        
        self.logger.info(f"Preprocessing took {(time.time()-start):.2f}s")

        # Parse all sentences
        all_tree_groups = parser.parse_all(sentence_groups, show_progress, num_workers=self.num_workers)

        # PARALLELIZED TREE GROUP ASSEMBLY
        assembly_start = time.time()
        
        if (self.parallel_config.get('tree_group_assembly', True) and 
            len(group_metadata) >= 100 and 
            self.num_workers > 1):
            
            self.logger.info("Using parallel tree group assembly")
            
            # Prepare data for parallel processing
            metadata_and_trees = [(meta, all_tree_groups, i) for i, meta in enumerate(group_metadata)]
            
            # Get configurable chunk size
            chunk_size = self._get_chunk_size('tree_group_assembly', 50, len(metadata_and_trees))
            
            # Process in parallel
            tree_groups = batch_parallel_process(
                metadata_and_trees,
                lambda item: TreeGroup(
                    group_id=item[0]['group_id'],
                    original_text=item[0]['text'],
                    trees=item[1][item[2]*2] if item[2]*2 < len(item[1]) else [],
                    original_text_b='' if not is_paired else item[0]['text_b'],
                    trees_b=[] if not is_paired else (item[1][item[2]*2+1] if item[2]*2+1 < len(item[1]) else []),
                    label=item[0]['label']
                ) if item[0] else None,
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                maintain_order=True
            )
            
            # Filter out None entries
            tree_groups = [tg for tg in tree_groups if tg is not None]
            
        else:
            # Sequential processing
            tree_groups = []
            for i, meta in enumerate(group_metadata):
                group = TreeGroup(
                    group_id=meta['group_id'],
                    original_text=meta['text'],
                    trees=all_tree_groups[i*2],
                    original_text_b='' if not is_paired else meta['text_b'],
                    trees_b=[] if not is_paired else all_tree_groups[i*2+1],
                    label=meta['label']
                )
                tree_groups.append(group)
        
        self.logger.info(f"Tree group assembly took {(time.time()-assembly_start):.2f}s")

        # Convert based on format
        if self.config.output_format.type == "infonce":
            dataset = self._convert_to_infonce_format(tree_groups, is_paired, self_paired)
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=4)

    def _convert_to_infonce_format(
        self,
        tree_groups: List[TreeGroup],
        is_paired=False,
        self_paired=False
    ) -> Dict:
        """Convert to InfoNCE format with group tracking - PARALLELIZED"""
        
        conversion_start = time.time()
        
        if (self.parallel_config.get('infonce_conversion', True) and 
            len(tree_groups) >= 50 and 
            self.num_workers > 1):
            
            self.logger.info("Using parallel InfoNCE conversion")
            
            # Get configurable chunk size
            chunk_size = self._get_chunk_size('infonce_conversion', 20, len(tree_groups))
            
            # Process tree groups in parallel
            groups_batches = batch_parallel_process(
                tree_groups,
                lambda group: self._convert_single_tree_group_to_infonce(group, is_paired, self_paired),
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                maintain_order=True
            )
            
            # Filter out None results
            groups = [group_data for group_data in groups_batches if group_data is not None]
            
        else:
            # Sequential processing
            groups = []
            
            for group in tree_groups:
                group_data = self._convert_single_tree_group_to_infonce(group, is_paired, self_paired)
                if group_data:
                    groups.append(group_data)
        
        self.logger.info(f"InfoNCE conversion took {(time.time()-conversion_start):.2f}s")

        return {
            "version": "1.0",
            "format": "infonce",
            "requires_word_embeddings": self.config.feature_extraction.do_not_store_word_embeddings,
            "groups": groups
        }

    def _convert_single_tree_group_to_infonce(self, group: TreeGroup, is_paired: bool, self_paired: bool) -> Optional[Dict]:
        """Convert a single tree group to InfoNCE format (helper for parallel processing)"""
        try:
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
            else:
                trees2 = []
            
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
                return group_data
                
        except Exception as e:
            self.logger.error(f"Error converting tree group {group.group_id} to InfoNCE: {e}")
        
        return None
