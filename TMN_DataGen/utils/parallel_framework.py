# TMN_DataGen/utils/parallel_framework.py
import multiprocessing as mp
from functools import partial
from typing import List, Callable, Any, Optional, Dict, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor

class ParallelizationMixin:
    """Mixin to add parallelization configuration to classes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set up parallel config after other initialization
        if hasattr(self, 'config'):
            self.parallel_config = self._get_parallel_config()
        else:
            self.parallel_config = {}
            
    def _get_parallel_config(self) -> Dict[str, bool]:
        """Get parallelization configuration from config"""
        default_config = {
            'tree_group_assembly': True,
            'infonce_conversion': True,
            'preprocessing': True,
            'validity_checking': True,
            'enhancement': True,  
            'reassembly': True,
            'diaparser_processing': True,
            'tree_building': True,
            'spacy_conversion': True,
            'spacy_parsing': True,  # Now enabled since CPU-only
        }
        
        if hasattr(self, 'config') and self.config:
            return self.config.get('parallelization', default_config)
        return default_config
    
    def _get_chunk_size(self, operation_name: str, default_size: int, num_items: int) -> int:
        """Get configurable chunk size for an operation"""
        if hasattr(self, 'config') and self.config:
            chunk_sizes = self.config.get('parallelization', {}).get('chunk_sizes', {})
            if operation_name in chunk_sizes:
                return chunk_sizes[operation_name]
        
        # Auto-calculate if not configured
        if hasattr(self, 'num_workers'):
            return max(1, min(default_size, num_items // (self.num_workers * 4)))
        return default_size


def batch_parallel_process(items: List[Any], process_func: Callable, 
                          num_workers: int = None, chunk_size: int = None,
                          maintain_order: bool = True) -> List[Any]:
    """
    Standalone function for parallel processing of items.
    """
    if not items:
        return []
    
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    if num_workers <= 1 or len(items) < 50:
        # Process sequentially for small datasets or single worker
        return [process_func(item) for item in items]
    
    if chunk_size is None:
        chunk_size = max(1, len(items) // (num_workers * 4))
    
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    def process_chunk(chunk):
        return [process_func(item) for item in chunk]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(process_chunk, chunks))
    
    # Flatten results
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
    
    return results


# Worker functions for multiprocessing (must be at module level for pickling)
def _process_text_pair_worker(args):
    """Worker function for text pair processing"""
    text_pairs_batch, is_paired, labels, preprocessor_config, sentence_splitter_config = args
    
    # Recreate preprocessor and sentence splitter in worker
    from TMN_DataGen.utils.text_preprocessing import BasePreprocessor, SentenceSplitter
    from omegaconf import OmegaConf
    
    config = OmegaConf.create(preprocessor_config)
    preprocessor = BasePreprocessor(config)
    sentence_splitter = SentenceSplitter()
    
    sentence_groups_batch = []
    group_metadata_batch = []
    
    for text1, text2, original_idx in text_pairs_batch:
        try:
            # Preprocess
            text1_clean = preprocessor.preprocess(text1)
            group1 = sentence_splitter.split(text1_clean)
            
            if is_paired:
                text2_clean = preprocessor.preprocess(text2)
                group2 = sentence_splitter.split(text2_clean)
            else:
                text2_clean = text2
                group2 = []
            
            # Create group metadata
            from uuid import uuid4
            group_id = str(uuid4())
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
            # Add empty entries to maintain indexing
            group_metadata_batch.append(None)
            sentence_groups_batch.extend([[], []] if is_paired else [[]])
    
    return sentence_groups_batch, group_metadata_batch


def _spacy_parse_worker(args):
    """Worker function for SpaCy parsing"""
    texts_batch, model_name, batch_size = args
    
    # Load SpaCy model in worker process
    import spacy
    model = spacy.load(model_name)
    
    docs = []
    for text in texts_batch:
        if text is None:
            docs.append(None)
        else:
            try:
                doc = model(text)
                docs.append(doc)
            except Exception:
                docs.append(None)
    
    return docs


def _tree_conversion_worker(args):
    """Worker function for tree conversion"""
    conversion_batch, config_dict = args
    
    # Recreate necessary objects in worker
    from TMN_DataGen.tree.node import Node
    from TMN_DataGen.tree.dependency_tree import DependencyTree
    from omegaconf import OmegaConf
    
    config = OmegaConf.create(config_dict)
    results = []
    
    for orig_sentence, doc_data in conversion_batch:
        if doc_data is None:
            results.append(None)
        else:
            try:
                # Reconstruct tree from serialized doc data
                tree = _rebuild_tree_from_doc_data(orig_sentence, doc_data, config)
                results.append(tree)
            except Exception as e:
                results.append(None)
    
    return results


def _rebuild_tree_from_doc_data(sentence, doc_data, config):
    """Helper function to rebuild tree from serialized SpaCy doc data"""
    from TMN_DataGen.tree.node import Node
    from TMN_DataGen.tree.dependency_tree import DependencyTree
    
    nodes = []
    for token_data in doc_data['tokens']:
        node = Node(
            word=token_data['text'],
            lemma=token_data['lemma'],
            pos_tag=token_data['pos'],
            idx=token_data['idx'],
            features={
                'original_text': token_data['text'],
                'morph_features': token_data.get('morph_features', {})
            }
        )
        nodes.append(node)
    
    if not nodes:
        raise ValueError("No valid tokens in document")
    
    # Connect nodes
    root = None
    for i, token_data in enumerate(doc_data['tokens']):
        if token_data['dep'] == 'ROOT':
            root = nodes[i]
        else:
            parent = nodes[token_data['head_idx']]
            parent.add_child(nodes[i], token_data['dep'])
    
    if not root:
        raise ValueError("No root node found in parse")
    
    return DependencyTree(sentence, root, config)
