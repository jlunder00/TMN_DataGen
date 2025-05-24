#!/usr/bin/env python3
"""
Comprehensive pytest-based test suite for parallelization functionality.
This automatically compares parallel vs sequential outputs to ensure consistency.

Usage:
    pytest test_parallelization.py -v
    pytest test_parallelization.py::test_preprocessing_consistency -v
    pytest test_parallelization.py -v --tb=short
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import patch, MagicMock
import sys
import os

# Add the package to path
current_dir = Path(__file__).parent
if current_dir.name == 'scripts':
    package_root = current_dir.parent
else:
    package_root = current_dir
sys.path.insert(0, str(package_root))

from TMN_DataGen import DatasetGenerator
from TMN_DataGen.utils.parallel_framework import (
    batch_parallel_process, 
    _preprocessing_task_worker,
    _tree_group_assembly_worker,
    _infonce_conversion_worker
)
from omegaconf import OmegaConf

class TestData:
    """Test data provider for consistent test cases"""
    
    @staticmethod
    def get_sample_text_pairs(size: int = 10) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Get sample text pairs and labels for testing"""
        base_pairs = [
            ("The cat sits on the mat.", "A feline rests on a rug."),
            ("Dogs love to play fetch.", "Canines enjoy playing with balls."),
            ("Birds can fly high.", "Avians soar through air."),
            ("Fish swim in ocean.", "Marine life inhabits water."),
            ("Sun shines brightly.", "Light illuminates the day."),
            ("Children read books.", "Kids enjoy stories."),
            ("Music brings joy.", "Songs make people happy."),
            ("Flowers bloom in spring.", "Plants grow in springtime."),
            ("Cars drive on roads.", "Vehicles travel on highways."),
            ("Students learn in school.", "Pupils study in classrooms."),
        ]
        
        labels = ["entailment", "neutral", "contradiction"]
        
        # Generate requested size by cycling through base data
        text_pairs = []
        test_labels = []
        
        for i in range(size):
            pair_idx = i % len(base_pairs)
            label_idx = i % len(labels)
            text_pairs.append(base_pairs[pair_idx])
            test_labels.append(labels[label_idx])
        
        return text_pairs, test_labels

    @staticmethod
    def get_test_config(parallel_enabled: bool = True) -> Dict:
        """Get test configuration with parallelization settings"""
        return {
            'parser': {
                'type': 'multi',
                'batch_size': 5,
                'diaparser_batch_size': 10,
                'spacy_batch_size': 10,
                'min_tokens': 2,
                'max_tokens': 50,
                'min_nodes': 2,
                'max_nodes': 50,
                'parsers': {
                    'diaparser': {
                        'enabled': True,
                        'model_name': 'en_ewt.electra-base'
                    },
                    'spacy': {
                        'enabled': True,
                        'model_name': 'en_core_web_sm'
                    }
                },
                'feature_sources': {
                    'tree_structure': 'diaparser',
                    'pos_tags': 'spacy',
                    'lemmas': 'spacy',
                    'dependency_labels': 'diaparser'
                }
            },
            'preprocessing': {
                'strictness_level': 1,
                'tokenizer': 'regex',
                'preserve_case': False,
                'remove_punctuation': False,
            },
            'feature_extraction': {
                'do_not_store_word_embeddings': True,
                'word_embedding_model': 'bert-base-uncased',
                'use_gpu': False,  # Disable GPU for testing
                'cache_embeddings': False,  # Disable caching for testing
                'batch_size': 8,
            },
            'output_format': {
                'type': 'infonce',
                'paired': True,
                'self_paired': False,
                'label_map': {
                    'entailment': 1,
                    'neutral': 0,
                    'contradiction': -1
                }
            },
            'parallelization': {
                'tree_group_assembly': parallel_enabled,
                'infonce_conversion': parallel_enabled,
                'preprocessing': parallel_enabled,
                'validity_checking': parallel_enabled,
                'enhancement': parallel_enabled,
                'reassembly': parallel_enabled,
                'diaparser_processing': parallel_enabled,
                'tree_building': parallel_enabled,
                'spacy_conversion': parallel_enabled,
                'spacy_parsing': False,  # Keep false to avoid GPU issues
                'min_items_for_parallel': 5,  # Low threshold for testing
                'auto_chunk_size': True,
            }
        }

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files"""
    with tempfile.TemporaryDirectory(prefix="tmn_test_") as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def mock_gpu_coordinator():
    """Mock the GPU coordinator to avoid initialization issues in tests"""
    with patch('TMN_DataGen.utils.gpu_coordinator.GPUCoordinator') as mock_coordinator:
        # Create a mock that can be used as a context manager
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=None)
        mock_coordinator.return_value = mock_instance
        yield mock_coordinator

@pytest.fixture
def sample_data():
    """Provide sample test data"""
    return TestData.get_sample_text_pairs(15)
        
def simple_processor(x):
    return x * 2

class TestParallelizationConsistency:
    """Test that parallel processing produces identical results to sequential"""
    
    def run_dataset_generation(self, text_pairs: List[Tuple[str, str]], 
                             labels: List[str], parallel_enabled: bool, 
                             temp_dir: Path, mock_gpu_coordinator) -> Dict:
        """Run dataset generation with specified parallelization setting"""
        
        config = TestData.get_test_config(parallel_enabled)
        output_path = temp_dir / f"test_output_{'parallel' if parallel_enabled else 'sequential'}.json"
        
        # Create generator
        generator = DatasetGenerator(num_workers=2 if parallel_enabled else 1)
        
        # Configure to use specific settings
        parser_config = config['parser']
        preprocessing_config = config['preprocessing']
        feature_config = {
            'feature_extraction': config['feature_extraction']
        }
        feature_config.update({'parallelization': config['parallelization']})
        output_config = config['output_format']
        
        try:
            generator.generate_dataset(
                text_pairs=text_pairs,
                labels=labels,
                output_path=str(output_path),
                parser_config=parser_config,
                preprocessing_config=preprocessing_config,
                feature_config=feature_config,
                output_config=output_config,
                verbosity='quiet',
                show_progress=False
            )
            
            # Load result
            with open(output_path, 'r') as f:
                result_data = json.load(f)
            
            return {
                'success': True,
                'data': result_data,
                'path': str(output_path)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def compare_results(self, sequential_result: Dict, parallel_result: Dict) -> Dict[str, Any]:
        """Compare two dataset generation results for consistency"""
        comparison = {
            'identical': True,
            'differences': [],
            'summary': {}
        }
        
        if not sequential_result['success'] or not parallel_result['success']:
            comparison['identical'] = False
            comparison['differences'].append({
                'type': 'execution_failure',
                'sequential_success': sequential_result['success'],
                'parallel_success': parallel_result['success'],
                'sequential_error': sequential_result.get('error'),
                'parallel_error': parallel_result.get('error')
            })
            return comparison
        
        seq_data = sequential_result['data']
        par_data = parallel_result['data']
        
        # Compare high-level structure
        seq_groups = seq_data.get('groups', [])
        par_groups = par_data.get('groups', [])
        
        comparison['summary'] = {
            'sequential_groups': len(seq_groups),
            'parallel_groups': len(par_groups),
            'group_count_match': len(seq_groups) == len(par_groups)
        }
        
        if len(seq_groups) != len(par_groups):
            comparison['identical'] = False
            comparison['differences'].append({
                'type': 'group_count_mismatch',
                'sequential_count': len(seq_groups),
                'parallel_count': len(par_groups)
            })
            return comparison
        
        # Compare group structure (not exact content due to potential ordering)
        seq_group_ids = set()
        par_group_ids = set()
        
        for group in seq_groups:
            if 'group_id' in group:
                seq_group_ids.add(group['group_id'])
        
        for group in par_groups:
            if 'group_id' in group:
                par_group_ids.add(group['group_id'])
        
        if len(seq_group_ids) != len(par_group_ids):
            comparison['identical'] = False
            comparison['differences'].append({
                'type': 'unique_group_id_count_mismatch',
                'sequential_unique_ids': len(seq_group_ids),
                'parallel_unique_ids': len(par_group_ids)
            })
        
        # Check for tree structure consistency
        seq_tree_counts = [len(group.get('trees', [])) for group in seq_groups]
        par_tree_counts = [len(group.get('trees', [])) for group in par_groups]
        
        if sorted(seq_tree_counts) != sorted(par_tree_counts):
            comparison['identical'] = False
            comparison['differences'].append({
                'type': 'tree_count_distribution_mismatch',
                'sequential_tree_counts': sorted(seq_tree_counts),
                'parallel_tree_counts': sorted(par_tree_counts)
            })
        
        comparison['summary']['tree_structure_consistent'] = len(comparison['differences']) == 0
        
        return comparison
    
    def test_full_pipeline_consistency(self, sample_data, temp_dir, mock_gpu_coordinator):
        """Test that the full pipeline produces consistent results"""
        text_pairs, labels = sample_data
        
        # Run sequential version
        sequential_result = self.run_dataset_generation(
            text_pairs, labels, parallel_enabled=False, 
            temp_dir=temp_dir, mock_gpu_coordinator=mock_gpu_coordinator
        )
        
        # Run parallel version
        parallel_result = self.run_dataset_generation(
            text_pairs, labels, parallel_enabled=True, 
            temp_dir=temp_dir, mock_gpu_coordinator=mock_gpu_coordinator
        )
        
        # Compare results
        comparison = self.compare_results(sequential_result, parallel_result)
        
        # Assert consistency
        if not comparison['identical']:
            failure_msg = f"Parallel and sequential results differ:\n"
            for diff in comparison['differences']:
                failure_msg += f"  - {diff['type']}: {diff}\n"
            failure_msg += f"Summary: {comparison['summary']}"
            pytest.fail(failure_msg)
        
        assert comparison['identical'], "Parallel and sequential results should be identical"
    
    def test_preprocessing_consistency(self):
        """Test preprocessing parallelization specifically"""
        # Create test tasks
        tasks = []
        for i in range(10):
            tasks.append({
                'text1': f"Test sentence {i} with some content.",
                'text2': f"Another test sentence {i} with different content.",
                'label': 'entailment' if i % 2 == 0 else 'neutral',
                'is_paired': True,
                'config': {
                    'strictness_level': 1,
                    'tokenizer': 'regex',
                    'preserve_case': False,
                    'remove_punctuation': False,
                }
            })
        
        # Process sequentially
        sequential_results = []
        for task in tasks:
            result = _preprocessing_task_worker(task)
            sequential_results.append(result)
        
        # Process in parallel
        parallel_results = batch_parallel_process(
            tasks, _preprocessing_task_worker,
            num_workers=2, min_items=5
        )
        
        # Compare results
        assert len(sequential_results) == len(parallel_results)
        
        for i, (seq_result, par_result) in enumerate(zip(sequential_results, parallel_results)):
            if seq_result is None and par_result is None:
                continue
            
            assert seq_result is not None, f"Sequential result {i} should not be None"
            assert par_result is not None, f"Parallel result {i} should not be None"
            
            # Compare structure
            assert 'sentence_groups' in seq_result and 'sentence_groups' in par_result
            assert 'metadata' in seq_result and 'metadata' in par_result
            
            # Compare sentence group lengths
            seq_groups = seq_result['sentence_groups']
            par_groups = par_result['sentence_groups']
            assert len(seq_groups) == len(par_groups), f"Result {i}: group count mismatch"
            
            # Compare metadata structure
            seq_meta = seq_result['metadata']
            par_meta = par_result['metadata']
            
            # Check that all required fields exist
            required_fields = ['group_id', 'text', 'text_clean', 'text_b', 'text_b_clean', 'label']
            for field in required_fields:
                assert field in seq_meta, f"Sequential metadata missing {field}"
                assert field in par_meta, f"Parallel metadata missing {field}"
    
    def test_batch_parallel_process_basic(self):
        """Test the core batch_parallel_process function"""
        
        
        test_data = list(range(20))
        
        # Sequential processing
        sequential_result = [simple_processor(x) for x in test_data]
        
        # Parallel processing
        parallel_result = batch_parallel_process(
            test_data, simple_processor,
            num_workers=2, min_items=10
        )
        
        assert sequential_result == parallel_result, "Basic parallel processing should match sequential"
    
    def test_batch_parallel_process_edge_cases(self):
        """Test edge cases for batch_parallel_process"""
        
        # def simple_processor(x):
        #     return x * 2
        
        # Empty input
        assert batch_parallel_process([], simple_processor, num_workers=2) == []
        
        # Small input (should use sequential)
        small_input = [1, 2, 3]
        result = batch_parallel_process(small_input, simple_processor, num_workers=2, min_items=10)
        expected = [simple_processor(x) for x in small_input]
        assert result == expected
        
        # Single worker (should use sequential)
        result = batch_parallel_process([1, 2, 3, 4, 5], simple_processor, num_workers=1)
        expected = [simple_processor(x) for x in [1, 2, 3, 4, 5]]
        assert result == expected

@pytest.mark.slow
class TestPerformanceComparison:
    """Test performance characteristics of parallel vs sequential"""
    
    def test_parallel_performance_benefit(self, sample_data, temp_dir, mock_gpu_coordinator):
        """Test that parallelization provides performance benefit for larger datasets"""
        # Use larger dataset for performance testing
        text_pairs, labels = TestData.get_sample_text_pairs(30)
        
        # Time sequential version
        start_time = time.time()
        sequential_result = TestParallelizationConsistency().run_dataset_generation(
            text_pairs, labels, parallel_enabled=False, 
            temp_dir=temp_dir, mock_gpu_coordinator=mock_gpu_coordinator
        )
        sequential_time = time.time() - start_time
        
        # Time parallel version
        start_time = time.time()
        parallel_result = TestParallelizationConsistency().run_dataset_generation(
            text_pairs, labels, parallel_enabled=True, 
            temp_dir=temp_dir, mock_gpu_coordinator=mock_gpu_coordinator
        )
        parallel_time = time.time() - start_time
        
        # Both should succeed
        assert sequential_result['success'], f"Sequential failed: {sequential_result.get('error')}"
        assert parallel_result['success'], f"Parallel failed: {parallel_result.get('error')}"
        
        print(f"\nPerformance comparison:")
        print(f"Sequential time: {sequential_time:.3f}s")
        print(f"Parallel time: {parallel_time:.3f}s")
        
        if parallel_time > 0:
            speedup = sequential_time / parallel_time
            print(f"Speedup: {speedup:.2f}x")
        
        # Note: We don't assert performance improvement here as it depends on many factors
        # This is mainly for observation and debugging

if __name__ == "__main__":
    # Allow running directly with python
    pytest.main([__file__, "-v"])
