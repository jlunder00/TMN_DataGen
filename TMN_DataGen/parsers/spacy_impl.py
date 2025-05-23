# Authored by: Jason Lunder, EWUID: 01032294, Github: https://github.com/jlunder00/

#spacy_impl.py
import time
from .base_parser import BaseTreeParser
from ..tree.node import Node
from ..tree.dependency_tree import DependencyTree
import spacy
from typing import List, Any, Optional
from omegaconf import DictConfig
import torch, gc
from ..utils.parallel_framework import ParallelizationMixin, batch_parallel_process, _spacy_parse_worker
from concurrent.futures import ProcessPoolExecutor

class SpacyTreeParser(BaseTreeParser, ParallelizationMixin):
    def __init__(self, config: Optional[DictConfig] = None, pkg_config = None, vocabs = [set({})], logger = None, max_concurrent=1, num_workers=1):
        super().__init__(config, pkg_config, vocabs, logger, max_concurrent, num_workers)
        if not hasattr(self, 'model'):
            model_name = self.config.get('model_name', 'en_core_web_sm')
            self.model_name = model_name  # Store for workers
            self.model = spacy.load(model_name)
    
    def parse_batch(self, sentence_groups: List[List[str]]) -> List[List[DependencyTree]]:
        pass

    def parse_batch_flat(self, flat_sentences, processed_texts: List[str], processed_tokens: List[List[str]]) -> List[List[DependencyTree]]:
        self.logger.info("Begin Spacy batch processing")
        # Process only the valid texts using spacy.pipe.
        # For any sentence that ended up invalid (None), we’ll later fill in a None.
        docs = [None] * len(processed_texts)
        valid_texts_with_indices = [(idx, text) for idx, text in enumerate(processed_texts) if text is not None]
        
        parse_time = time.time()
        if valid_texts_with_indices:
            valid_indices, valid_texts = zip(*valid_texts_with_indices)
            if (self.parallel_config.get('spacy_parsing', True) and 
                len(valid_texts) >= 100 and 
                self.num_workers > 1):
                
                self.logger.info("Using parallel SpaCy parsing")
                
                chunk_size = self._get_chunk_size('spacy_parsing', 50, len(valid_texts))
                
                # Prepare data for parallel processing
                text_chunks = [valid_texts[i:i + chunk_size] 
                              for i in range(0, len(valid_texts), chunk_size)]
                
                # Process chunks in parallel
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    chunk_args = [(chunk, self.model_name, self.spacy_batch_size) 
                                 for chunk in text_chunks]
                    chunk_results = list(executor.map(_spacy_parse_worker, chunk_args))
                
                # Flatten results and assign to correct indices
                flat_docs = []
                for chunk_result in chunk_results:
                    flat_docs.extend(chunk_result)
                
                for idx, doc in zip(valid_indices, flat_docs):
                    docs[idx] = doc
                    
            else:
                # Sequential
                for idx, doc in zip(valid_indices, self.model.pipe(valid_texts, batch_size=self.spacy_batch_size)):
                    docs[idx] = doc
        self.logger.info(f"parsing in spacy parser took: {time.time()-parse_time}")

        # Convert each Doc to a DependencyTree.
        convert_time = time.time()
        if self.parallel_config.get('spacy_conversion', True) and len(flat_sentences) >= 50:
            # Parallel
            self.logger.info("Using parallel SpaCy tree conversion")
            
            # Prepare conversion data
            conversion_data = [(orig_sentence, doc) for orig_sentence, doc in zip(flat_sentences, docs)]
            
            chunk_size = self._get_chunk_size('spacy_conversion', 25, len(conversion_data))

            # Convert docs to trees in parallel
            # tree_results = batch_parallel_process(
            #     conversion_data,
            #     lambda batch: self._convert_docs_to_trees_batch(batch) if isinstance(batch, list) else [self._convert_docs_to_trees_batch([batch])[0]],
            #     num_workers=self.num_workers,
            #     chunk_size=25,
            #     maintain_order=True
            # )
            tree_results = batch_parallel_process(
                conversion_data,
                lambda item: self._convert_to_tree(item[0], item[1]) if item[1] is not None else None,
                num_workers=self.num_workers,
                chunk_size=chunk_size,
                maintain_order=True
            )
            
            trees_flat = tree_results
            
            # Flatten results
            # trees_flat = []
            # for result_batch in tree_results:
            #     if isinstance(result_batch, list):
            #         trees_flat.extend(result_batch)
            #     else:
            #         trees_flat.append(result_batch)
            
        else:
            # Sequential
            trees_flat = []
            for orig_sentence, doc in zip(flat_sentences, docs):
                if doc is None:
                    trees_flat.append(None)
                else:
                    try:
                        tree = self._convert_to_tree(orig_sentence, doc)
                        trees_flat.append(tree)
                    except Exception as e:
                        self.logger.error(f"Error converting doc to tree for sentence '{orig_sentence}': {e}")
                        trees_flat.append(None)
        self.logger.info(f"tree building in spacy parser took: {time.time()-convert_time}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return trees_flat


    def parse_single(self, sentences: List[str]) -> List[DependencyTree]:
        self.logger.info("Begin Spacy single processing")
        # Preprocess
        trees = []
        processed_sentences = []
        self.logger.info(f"Processing sentence group {sentences} with Spacy")

        for sentence in sentences:
            try:
                if self.verbose == 'info' or self.verbose == 'debug':
                    self.logger.info(f"Processing {sentence} with Spacy")
                tokens = self.preprocess_and_tokenize(sentence)
                if not tokens:
                    self.logger.warning(f"No tokens after processing: {sentence}")
                    trees.append(None)
                    continue
                processed_text = ' '.join(tokens)
                self.logger.debug(f"Preprocessed '{sentence}' to '{processed_text}'")
                processed_sentences.append(processed_text)
                doc = self.model(processed_text)
                tree = self._convert_to_tree(sentence, doc)
                if tree:
                    trees.append(tree)
                else:
                    trees.append(None)
            except Exception as e:
                self.logger.error(f'Error processing sentence {sentence}: {e}')
                trees.append(None)
                continue

        if len(trees) < 1:
            self.logger.warning("No valid trees produced from batch")
            trees = [None]
        if self.verbose == 'debug':
            for sent, tree in zip(processed_sentences, trees):
                self.logger.debug(f"\nSpacy processed sentence: {sent}")
                self.logger.debug(f"Generated Spacy tree with {len(tree.root.get_subtree_nodes()) if tree is not None else 0} nodes")
                
        return trees

    
    # def _convert_docs_to_trees_batch(self, doc_sentence_batch: List[Tuple[str, Any]]) -> List[DependencyTree]:
    #     """Convert a batch of SpaCy docs to dependency trees"""
    #     trees = []
    #     
    #     for orig_sentence, doc in doc_sentence_batch:
    #         if doc is None:
    #             trees.append(None)
    #         else:
    #             try:
    #                 tree = self._convert_to_tree(orig_sentence, doc)
    #                 trees.append(tree)
    #             except Exception as e:
    #                 self.logger.error(f"Error converting doc to tree for sentence '{orig_sentence}': {e}")
    #                 trees.append(None)
    #     
    #     return trees

    def _convert_to_tree(self, sentence, doc: Any) -> DependencyTree:
        try:
            nodes = [
                Node(
                    word=token.text,
                    lemma=token.lemma_,
                    pos_tag=token.pos_,
                    idx=token.i,
                    features={
                        'original_text': token.text,
                        'morph_features': dict(feature.split('=') 
                                             for feature in str(token.morph).split('|')
                                             if feature != '')  # Handle empty morph case
                    }
                )
                for token in doc
            ]
            
            if not nodes:
                raise ValueError("No valid tokens in document")
            # Connect nodes
            root = None
            for token in doc:
                if token.dep_ == 'ROOT':
                    root = nodes[token.i]
                else:
                    parent = nodes[token.head.i]
                    parent.add_child(nodes[token.i], token.dep_)

            if not root:
                raise ValueError("No root node found in parse")
            
            tree = DependencyTree(sentence, root, self.config)
            self.logger.debug("Tree structure created successfully")

            return tree
        except Exception as e:
            self.logger.error(f"Failed to convert doc to tree: {e}")
            raise
    
    def _serialize_doc_for_multiprocessing(self, doc):
        """Serialize SpaCy doc for multiprocessing"""
        if doc is None:
            return None
            
        tokens = []
        for token in doc:
            tokens.append({
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
        
        return {'tokens': tokens}
