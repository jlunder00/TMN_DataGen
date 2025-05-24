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
            'infonce_conversion': False,
            'preprocessing': True,
            'validity_checking': True,
            'enhancement': True,  
            'reassembly': True,
            'diaparser_processing': True,
            'tree_building': True,
            'spacy_conversion': True,
            'spacy_parsing': True,
        }
        
        if hasattr(self, 'config') and self.config:
            return self.config.get('parallelization', default_config)
        return default_config

    def _get_chunk_size(self, operation_name: str, default_size: int, num_items: int) -> int:
        """Get configurable chunk size for an operation"""
        if hasattr(self, 'config') and self.config:
            parallelization_config = self.config.get('parallelization', {})
            
            # Check if auto chunk sizing is disabled
            if not parallelization_config.get('auto_chunk_size', True):
                chunk_sizes = parallelization_config.get('chunk_sizes', {})
                if operation_name in chunk_sizes:
                    return chunk_sizes[operation_name]
                return default_size
        
        # Auto-calculate if not configured or auto_chunk_size is True
        if hasattr(self, 'num_workers'):
            if operation_name in ['preprocessing', 'tree_group_assembly', 'infonce_conversion']:
                # These are worth parallelizing with larger chunks
                return max(50, min(default_size, num_items // (self.num_workers * 2)))
            else:
                # For other operations, make chunks much larger to reduce overhead
                return max(100, min(default_size, num_items // self.num_workers))
        return default_size
        # if hasattr(self, 'num_workers'):
        #     return max(1, min(default_size, num_items // (self.num_workers * 4)))
        # return default_size
    
    def _get_min_items_for_parallel(self) -> int:
        """Get minimum items threshold for parallelization"""
        if hasattr(self, 'config') and self.config:
            return self.config.get('parallelization', {}).get('min_items_for_parallel', 100)
        return 100

def batch_parallel_process(items: List[Any], process_func: Callable, 
                          num_workers: int = None, chunk_size: int = None,
                          maintain_order: bool = True, min_items: int = 50) -> List[Any]:
    """
    Standalone function for parallel processing of items.
    """
    if not items:
        return []
    
    if num_workers is None:
        num_workers = mp.cpu_count() - 1
    
    if num_workers <= 1 or len(items) < min_items:
        # Process sequentially for small datasets or single worker
        return [process_func(item) for item in items]
    
    if chunk_size is None:
        chunk_size = max(1, len(items) // (num_workers * 4))
    
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    
    # Create worker arguments - each chunk gets the process_func
    chunk_args = [(chunk, process_func) for chunk in chunks]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        chunk_results = list(executor.map(_process_chunk_worker, chunk_args))
    
    # Flatten results
    results = []
    for chunk_result in chunk_results:
        results.extend(chunk_result)
    
    return results

def _process_chunk_worker(args):
    """Worker function that processes a chunk of items"""
    chunk, process_func = args
    return [process_func(item) for item in chunk]

# ==============================================================================
# WORKER FUNCTIONS FOR SPECIFIC OPERATIONS
# ==============================================================================

def _tree_group_assembly_worker(item):
    """Worker function for tree group assembly"""
    if not item or not item.get('meta'):
        return None
    
    from TMN_DataGen.dataset_generator import TreeGroup
    
    return TreeGroup(
        group_id=item['meta']['group_id'],
        original_text=item['meta']['text'],
        trees=item['trees_a'],
        original_text_b='' if not item['is_paired'] else item['meta']['text_b'],
        trees_b=item['trees_b'],
        label=item['meta']['label']
    )

def _infonce_conversion_worker(args):
    """Worker function for InfoNCE conversion"""
    group, is_paired, self_paired = args
    
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
        # Log error but don't crash the worker
        return None
    
    return None

def _preprocessing_task_worker(task):
    """Worker function for preprocessing tasks"""
    try:
        from TMN_DataGen.utils.text_preprocessing import BasePreprocessor, SentenceSplitter
        from TMN_DataGen.dataset_generator import generate_group_id
        from omegaconf import OmegaConf
        
        # Extract task data
        text1 = task['text1']
        text2 = task['text2']
        is_paired = task['is_paired']
        
        # Recreate preprocessor and splitter in worker
        config = OmegaConf.create({'preprocessing': task['config']})
        preprocessor = BasePreprocessor(config)
        sentence_splitter = SentenceSplitter()
        
        # Process exactly like sequential version
        text1_clean = preprocessor.preprocess(text1)
        group1 = sentence_splitter.split(text1_clean)
        
        if is_paired:
            text2_clean = preprocessor.preprocess(text2)
            group2 = sentence_splitter.split(text2_clean)
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
            'label': task['label']
        }

        to_add = [group1]
        if is_paired:
            to_add.append(group2)
        
        return {
            'sentence_groups': to_add,
            'metadata': metadata
        }
        
    except Exception as e:
        return None

def _diaparser_process_prediction_worker(args):
    """Worker function for processing single DiaParser prediction"""
    i, sentence = args
    
    try:
        def ensure_list(val):
            """Convert various input formats to list"""
            if isinstance(val, str):
                return val.split()
            return list(val)

        token_data = {
            'words': ensure_list(sentence.values[1]),
            'lemmas': ensure_list(sentence.values[2]),
            'pos_tags': ensure_list(sentence.values[3]),
            'heads': [int(h) for h in ensure_list(sentence.values[6])],
            'rels': ensure_list(sentence.values[7])
        }

        # Verify all lists have same length
        list_lens = [len(lst) for lst in token_data.values()]
        if len(set(list_lens)) != 1:
            raise ValueError(f"Inconsistent token list lengths: {list_lens}")
        
        return (i, token_data)
        
    except Exception as e:
        return (i, None)

def _diaparser_build_tree_worker(args):
    """Worker function for DiaParser tree building"""
    token_data, sentence, config_dict = args
    
    if not token_data or not sentence:
        return None
    
    try:
        from TMN_DataGen.tree.node import Node
        from TMN_DataGen.tree.dependency_tree import DependencyTree
        from omegaconf import OmegaConf
        
        config = OmegaConf.create(config_dict) if config_dict else None
        
        # Step 1: Create all nodes
        nodes = []
        for i in range(len(token_data['words'])):
            node = Node(
                word=token_data['words'][i],
                lemma=token_data['lemmas'][i],
                pos_tag=token_data['pos_tags'][i],
                idx=i,
                features={
                    'original_text': token_data['words'][i]
                }
            )
            nodes.append(node)

        # Step 2: Connect nodes using head indices
        root = None
        for i, (node, head_idx, rel) in enumerate(zip(nodes, 
                                                     token_data['heads'],
                                                     token_data['rels'])):
            if head_idx == 0:  # Root node
                root = node
            else:
                # Head indices are 1-based in CoNLL format
                parent = nodes[head_idx - 1]
                parent.add_child(node, rel)
                
        # Step 3: Verify we found a root and built valid tree
        if root is None:
            raise ValueError(f"No root node found in parse: {sentence}")

        tree = DependencyTree(sentence, root, config=config)
        
        # Verify all nodes are reachable and structure is valid
        tree_nodes = tree.root.get_subtree_nodes()
        if len(tree_nodes) != len(nodes):
            raise ValueError(
                f"Tree structure incomplete: only {len(tree_nodes)} of {len(nodes)} "
                f"nodes reachable from root"
            )
            
        # Verify tree structure is valid
        if not root.verify_tree_structure():
            raise ValueError(
                f"Invalid tree structure detected for sentence: {sentence}"
            )

        return tree
        
    except Exception as e:
        return None

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

def _spacy_convert_to_tree_worker(args):
    """Worker function for SpaCy tree conversion"""
    sentence, doc_data, config_dict = args
    
    if not doc_data:
        return None
        
    try:
        from TMN_DataGen.tree.node import Node
        from TMN_DataGen.tree.dependency_tree import DependencyTree
        from omegaconf import OmegaConf
        
        config = OmegaConf.create(config_dict) if config_dict else None
        
        # Recreate nodes from serialized doc data
        if hasattr(doc_data, '__iter__') and hasattr(doc_data, '__getitem__'):
            # It's a SpaCy doc object - convert to serializable format first
            tokens_data = []
            for token in doc_data:
                tokens_data.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_,
                    'idx': token.i,
                    'dep': token.dep_,
                    'head_idx': token.head.i,
                    'morph_features': dict(feature.split('=') 
                                         for feature in str(token.morph).split('|')
                                         if feature != '')
                })
            doc_data = {'tokens': tokens_data}
        
        # Build nodes from token data
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
        
        tree = DependencyTree(sentence, root, config)
        return tree
        
    except Exception as e:
        return None

def _multiparser_validate_token_single_worker(args):
    """Worker function for single token validation"""
    i, tokens, group_index, sentence_index, sentence, config_dict = args
    
    try:
        from omegaconf import OmegaConf
        config = OmegaConf.create(config_dict) if config_dict else None
        
        is_valid = True
        
        if not tokens:
            is_valid = False
        elif (config and 
              (len(tokens) < config.get('parser', {}).get('min_tokens', 3) or 
               len(tokens) > config.get('parser', {}).get('max_tokens', 100))):
            is_valid = False
        
        processed_text = " ".join(tokens) if is_valid and tokens else None
        processed_tokens = tokens if is_valid else None
        
        return (i, is_valid, processed_text, processed_tokens, group_index)
        
    except Exception as e:
        return (i, False, None, None, group_index)

def _multiparser_enhance_tree_single_worker(args):
    """Worker function for single tree enhancement"""
    idx, base_tree, parser_results_for_idx, group_index, sentence_index, config_dict = args
    
    try:
        from omegaconf import OmegaConf
        import copy
        
        config = OmegaConf.create(config_dict) if config_dict else None
        
        if base_tree is None:
            return (idx, None)
        
        # Skip if any parser failed for this sentence
        if any(parser_results_for_idx[name] is None for name in parser_results_for_idx):
            return (idx, None)
        
        # Create a copy of the tree for processing
        enhanced_tree = copy.deepcopy(base_tree)
        enhanced_tree.config = config
        
        # Simple enhancement - just copy basic features
        if not _is_valid_tree_contents(enhanced_tree):
            return (idx, None)
        
        return (idx, enhanced_tree)
        
    except Exception as e:
        return (idx, None)

def _reassembly_worker(args):
    """Worker function for reassembly - extracts assignment info from tuples
    
    Note: This operation is so lightweight that parallelization overhead 
    is almost certainly not worth it. Consider disabling parallel reassembly.
    """
    idx, group_index, sentence_index, tree = args
    return (group_index, sentence_index, tree)

def _is_valid_tree_contents(tree):
    """Helper function to validate tree contents"""
    try:
        for node in tree.root.get_subtree_nodes():
            # Check for missing POS tags
            if node.pos_tag == '_' or not node.pos_tag:
                return False
            if node != tree.root and (not node.dependency_to_parent or node.dependency_to_parent == '_'):
                return False
        return True
    except:
        return False
