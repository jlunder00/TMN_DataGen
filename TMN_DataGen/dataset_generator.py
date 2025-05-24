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
from .utils.parallel_framework import ParallelizationMixin, batch_parallel_process, _preprocessing_task_worker, _infonce_conversion_worker, _tree_group_assembly_worker
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

        with open(config_dir / 'default_parallel_config.yaml') as f:
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

    def _process_text_pair_batch(self, text_pairs_batch: List[Tuple[str, str, int]], 
                                is_paired: bool, labels: List[str]) -> Tuple[List[List[str]], List[Dict]]:
        """Process a batch of text pairs for preprocessing and sentence splitting"""
        sentence_groups_batch = []
        group_metadata_batch = []
        
        for i, (text1, text2, original_idx) in text_pairs_batch:
            try:
                # Preprocess
                text1_clean = self.preprocessor.preprocess(text1)
                group1 = self.sentence_splitter.split(text1_clean)
                
                if is_paired:
                    text2_clean = self.preprocessor.preprocess(text2)
                    group2 = self.sentence_splitter.split(text2_clean)
                else:
                    text2_clean = text2
                    group2 = []
                
                # Create group metadata
                group_id = generate_group_id()
                metadata = {
                    'group_id': group_id,
                    'text': text1,
                    'text_clean': text1_clean,
                    'text_b': text2,
                    'text_b_clean': text2_clean,
                    'label': labels[original_idx]
                }
                
                group_metadata_batch.append(metadata)
                
                to_add = [group1]
                if is_paired:
                    to_add.append(group2)
                sentence_groups_batch.extend(to_add)
                
            except Exception as e:
                self.logger.error(f"Error processing text pair {original_idx}: {e}")
                # Add empty entries to maintain indexing
                group_metadata_batch.append(None)
                sentence_groups_batch.extend([[], []] if is_paired else [[]])
        
        return sentence_groups_batch, group_metadata_batch
    
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
        config.feature_extraction.embedding_cache_dir = cache_dir if cache_dir else config.feature_extraction.embedding_cache_dir
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
        parser = MultiParser(config, pkg_config, self.vocabs, self.logger, max_concurrent, self.num_workers)


        # Process sentence pairs
        if verbosity != 'quiet':
            self.logger.info(f"\nGenerating dataset...")
            self.logger.info(f"Processing {len(text_pairs)} text pairs")

        is_paired = self.config.output_format and self.config.output_format.get('paired', False)
        self_paired = self.config.output_format and self.config.output_format.get('self_paired', False)
        self.preprocessor = BasePreprocessor(self.config)

        # Split sentences and track groups
        if self.parallel_config.get('preprocessing', True) and len(text_pairs) >= self._get_min_items_for_parallel() and self.num_workers > 1:
            self.logger.info("Using parallel preprocessing and sentence splitting")
            # Prepare data with original indices for tracking
            # indexed_text_pairs = [(text1, text2, i) for i, (text1, text2) in enumerate(text_pairs)]
            preprocessing_tasks = []
            for i, (text1, text2) in enumerate(text_pairs):
                preprocessing_tasks.append({
                    'text1': text1,
                    'text2': text2,
                    'label': labels[i],
                    'is_paired': is_paired,
                    'config': self.config.preprocessing
                })
            
            chunk_size = self._get_chunk_size('preprocessing', 100, len(preprocessing_tasks))

            preprocessing_results = batch_parallel_process(
                # indexed_text_pairs,
                # lambda item: _process_single_text_pair((item, is_paired, labels, self.config.preprocessing)),
                preprocessing_tasks,
                _preprocessing_task_worker,
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                min_items=self._get_min_items_for_parallel()
            )

            # Combine results
            sentence_groups = []
            group_metadata = []
            for sentence_groups_batch, group_metadata_batch in preprocessing_results:
                sentence_groups.extend(sentence_groups_batch)
                group_metadata.extend(group_metadata_batch)
            
            # Filter out None entries
            group_metadata = [meta for meta in group_metadata if meta is not None]
        else:
            sentence_groups = []
            group_metadata = []
            
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
        self.logger.info(f"Preprocessing took {(time.time()-start):.2f}s")

        # Parse all sentences
        all_tree_groups = parser.parse_all(sentence_groups, show_progress)

        # Organize trees with groups
        assembly_start = time.time()
        if self.parallel_config.get('tree_group_assembly', True) and len(group_metadata) >= self._get_min_items_for_parallel() and self.num_workers > 1:
            # Parallel
            self.logger.info("Using parallel tree group assembly")
            
            # Prepare data for parallel processing
            # metadata_and_trees = [(meta, all_tree_groups, i) for i, meta in enumerate(group_metadata)]
            # Pre-extract the needed trees for each metadata item
            tree_assembly_data = []
            for i, meta in enumerate(group_metadata):
                trees_a = all_tree_groups[i*2] if i*2 < len(all_tree_groups) else []
                trees_b = [] if not is_paired else (all_tree_groups[i*2+1] if i*2+1 < len(all_tree_groups) else [])
                
                tree_assembly_data.append({
                    'meta': meta,
                    'trees_a': trees_a,
                    'trees_b': trees_b,
                    'is_paired': is_paired
                })

            chunk_size = self._get_chunk_size('tree_group_assembly', 50, len(tree_assembly_data))
            
            # # Process in parallel
            # tree_groups = batch_parallel_process(
            #     metadata_and_trees,
            #     lambda item: self._create_tree_group_batch([item], is_paired)[0] if item[0] else None,
            #     num_workers=self.num_workers,
            #     maintain_order=True
            # )

            tree_groups = batch_parallel_process(
                tree_assembly_data,
                _tree_group_assembly_worker,  
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                maintain_order=True,
                min_items=self._get_min_items_for_parallel()
            )
            # tree_groups = batch_parallel_process(
            #     metadata_and_trees,
            #     lambda item: TreeGroup(
            #         group_id=item[0]['group_id'],
            #         original_text=item[0]['text'],
            #         trees=item[1][item[2]*2] if item[2]*2 < len(item[1]) else [],
            #         original_text_b='' if not is_paired else item[0]['text_b'],
            #         trees_b=[] if not is_paired else (item[1][item[2]*2+1] if item[2]*2+1 < len(item[1]) else []),
            #         label=item[0]['label']
            #     ) if item[0] else None,
            #     num_workers=self.num_workers,
            #     chunk_size=chunk_size,
            #     maintain_order=True
            # )
            
            # Filter out None entries
            tree_groups = [tg for tg in tree_groups if tg is not None]
            
        else:
            # Sequential
            tree_groups = []
            group_idx = 0
            for i, meta in enumerate(group_metadata):
                # if i + 1 >= len(all_tree_groups):
                #     continue
                    
                group = TreeGroup(
                    group_id=meta['group_id'],
                    original_text=meta['text'],
                    trees=all_tree_groups[i*2] if i*2 < len(all_tree_groups) else [],
                    original_text_b= '' if not is_paired else meta['text_b'],
                    trees_b = [] if not is_paired else all_tree_groups[i*2+1] if i*2+1 < len(all_tree_groups) else [],
                    label = meta['label']
                )
                tree_groups.append(group)
        self.logger.info(f"Tree group assembly took {(time.time()-assembly_start):.2f}s")

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
        
        conversion_start = time.time()
        if self.parallel_config.get('infonce_conversion', True) and len(tree_groups) >= self._get_min_items_for_parallel() and self.num_workers > 1:
            # Parallel
            self.logger.info("Using parallel InfoNCE conversion")

            chunk_size = self._get_chunk_size('infonce_conversion', 20, len(tree_groups))

            worker_args = [(group, is_paired, self_paired) for group in tree_groups]
            
            # Process tree groups in parallel
            groups_batches = batch_parallel_process(
                worker_args,
                _infonce_conversion_worker, 
                num_workers=self.num_workers,
                chunk_size=chunk_size,  # Smaller chunks for complex operations
                maintain_order=True,
                min_items=self._get_min_items_for_parallel()
            )
            
            # Flatten results
            groups = []
            for batch in groups_batches:
                groups.extend(batch)
            
        else:
            # Sequential
            groups = []
            
            for group in tree_groups:
                group_data = self._convert_single_tree_group_to_infonce(group, is_paired, self_paired)
                if group_data:
                    groups.append(group_data)
                # # Convert all trees to graph format
                # trees1 = [
                #     t.to_graph_data() for t in group.trees 
                #     if t is not None
                # ]
                # if is_paired:
                #     trees2 = [
                #         t.to_graph_data() for t in group.trees_b
                #         if t is not None
                #     ]
                # 
                # add = (not is_paired and trees1) or (is_paired and trees1 and trees2) or (self_paired and trees1)

                # if add:  # Only add if both have valid trees
                #     group_data = {
                #         "group_id": group.group_id,
                #         "text": group.original_text,
                #         "trees": trees1,
                #         "label": group.label
                #     }
                #     if is_paired:
                #         group_data['text_b'] = group.original_text_b
                #         group_data['trees_b'] = trees2
                #     groups.append(group_data)
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
