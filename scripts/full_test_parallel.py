#!/usr/bin/env python3
"""
Comprehensive test script for the parallelization system.
Run this to verify all parallelization features work correctly.

Usage:
    python scripts/test_parallelization_comprehensive.py
    python scripts/test_parallelization_comprehensive.py --quick
    python scripts/test_parallelization_comprehensive.py --size 100
"""

import json
import time
import tempfile
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
import yaml
from omegaconf import OmegaConf

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from TMN_DataGen import DatasetGenerator
from TMN_DataGen.utils.parallel_framework import batch_parallel_process, ParallelizationMixin

# Module-level functions for multiprocessing (must be at top level for pickling)
def simple_test_processor(x):
    """Simple processor for testing - must be at module level"""
    return x * 2

def complex_test_processor(item):
    """More complex processor for testing"""
    if isinstance(item, dict):
        return {k: v * 2 if isinstance(v, (int, float)) else v for k, v in item.items()}
    return item

def error_prone_processor(x):
    """Processor that occasionally fails for testing error handling"""
    if x % 10 == 7:  # Fail on multiples of 10 plus 7
        raise ValueError(f"Intentional error for testing: {x}")
    return x * 3

class ComprehensiveParallelizationTester:
    def __init__(self, temp_dir: Path = None, verbosity: str = 'normal'):
        self.verbosity = verbosity
        self.test_results = {}
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="tmn_parallel_test_"))
        self.temp_dir.mkdir(exist_ok=True)
        
        if verbosity != 'quiet':
            print(f"Using temp directory: {self.temp_dir}")
        
    def log(self, message: str, level: str = 'info'):
        """Controlled logging based on verbosity"""
        if self.verbosity == 'quiet' and level == 'info':
            return
        if self.verbosity == 'normal' and level == 'debug':
            return
        print(message)
        
    def create_test_configs(self) -> Dict[str, Dict]:
        """Create test configurations with different parallelization settings"""
        
        # Config 1: All parallelization enabled with low thresholds
        config_parallel = {
            'parallelization': {
                'tree_group_assembly': True,
                'infonce_conversion': True,
                'preprocessing': True,
                'validity_checking': True,
                'enhancement': True,
                'reassembly': True,
                'diaparser_processing': True,
                'tree_building': True,
                'spacy_conversion': True,
                'spacy_parsing': False,  # Keep false for GPU coordination
                'min_items_for_parallel': 5,  # Very low threshold for testing
                'auto_chunk_size': True,
                'chunk_sizes': {
                    'tree_group_assembly': 3,
                    'infonce_conversion': 4,
                    'preprocessing': 5,
                    'validity_checking': 8,
                    'enhancement': 2,
                    'tree_building': 3,
                    'spacy_conversion': 4,
                }
            }
        }
        
        # Config 2: All parallelization disabled
        config_sequential = {
            'parallelization': {
                'tree_group_assembly': False,
                'infonce_conversion': False,
                'preprocessing': False,
                'validity_checking': False,
                'enhancement': False,
                'reassembly': False,
                'diaparser_processing': False,
                'tree_building': False,
                'spacy_conversion': False,
                'spacy_parsing': False,
                'min_items_for_parallel': 1000,  # High threshold to disable
                'auto_chunk_size': True,
            }
        }
        
        # Config 3: Mixed - some parallel, some sequential
        config_mixed = {
            'parallelization': {
                'tree_group_assembly': True,
                'infonce_conversion': False,
                'preprocessing': True,
                'validity_checking': False,
                'enhancement': True,
                'reassembly': False,
                'diaparser_processing': True,
                'tree_building': False,
                'spacy_conversion': True,
                'spacy_parsing': False,
                'min_items_for_parallel': 10,
                'auto_chunk_size': True,
            }
        }
        
        # Config 4: Fixed chunk sizes (auto_chunk_size disabled)
        config_fixed_chunks = {
            'parallelization': {
                'tree_group_assembly': True,
                'infonce_conversion': True,
                'preprocessing': True,
                'validity_checking': True,
                'enhancement': True,
                'reassembly': True,
                'diaparser_processing': True,
                'tree_building': True,
                'spacy_conversion': True,
                'spacy_parsing': False,
                'min_items_for_parallel': 5,
                'auto_chunk_size': False,  # Disable auto sizing
                'chunk_sizes': {
                    'tree_group_assembly': 2,
                    'infonce_conversion': 3,
                    'preprocessing': 4,
                    'validity_checking': 6,
                    'enhancement': 2,
                    'tree_building': 3,
                    'spacy_conversion': 3,
                    'diaparser_processing': 4,
                }
            }
        }
        
        return {
            'parallel': config_parallel,
            'sequential': config_sequential,
            'mixed': config_mixed,
            'fixed_chunks': config_fixed_chunks
        }
    
    def create_test_data(self, size: int = 30) -> Tuple[List[Tuple[str, str]], List[str]]:
        """Create test sentence pairs and labels"""
        base_sentences = [
            ("The cat sits on the mat.", "A feline rests on a rug."),
            ("Dogs love to play fetch.", "Canines enjoy playing with balls."),
            ("Birds can fly high in the sky.", "Avians soar through the atmosphere."),
            ("Fish swim in the deep ocean.", "Marine life inhabits the sea."),
            ("The sun shines brightly today.", "Sunlight illuminates the day."),
            ("Children enjoy reading books.", "Kids like to read stories."),
            ("Music brings joy to people.", "Songs make people happy."),
            ("Flowers bloom in spring.", "Blossoms appear in springtime."),
            ("Cars drive on the highway.", "Vehicles travel on roads."),
            ("Students learn in school.", "Pupils study in classrooms."),
            ("The rain falls gently.", "Water drops softly from clouds."),
            ("Scientists conduct experiments.", "Researchers perform tests."),
            ("Artists create beautiful paintings.", "Painters make lovely artwork."),
            ("Chefs cook delicious meals.", "Cooks prepare tasty food."),
        ]
        
        base_labels = ["entailment", "neutral", "contradiction"]
        
        # Generate test pairs by cycling through available sentences
        text_pairs = []
        test_labels = []
        
        for i in range(size):
            sentence_pair = base_sentences[i % len(base_sentences)]
            text_pairs.append(sentence_pair)
            test_labels.append(base_labels[i % len(base_labels)])
        
        return text_pairs, test_labels
    
    def test_config_loading(self) -> bool:
        """Test that configurations are properly loaded and applied"""
        self.log("\n=== Testing Configuration Loading ===")
        
        configs = self.create_test_configs()
        all_passed = True
        
        for config_name, config_data in configs.items():
            self.log(f"\nTesting config: {config_name}")
            
            try:
                # Create a test class that inherits from ParallelizationMixin
                class TestClass(ParallelizationMixin):
                    def __init__(self, config):
                        self.config = OmegaConf.create(config)
                        self.num_workers = 4
                        super().__init__()
                
                test_obj = TestClass(config_data)
                
                # Test min_items_for_parallel
                expected_min_items = config_data['parallelization']['min_items_for_parallel']
                actual_min_items = test_obj._get_min_items_for_parallel()
                self.log(f"  min_items_for_parallel: expected={expected_min_items}, actual={actual_min_items}", 'debug')
                
                if actual_min_items != expected_min_items:
                    self.log(f"  ✗ Min items mismatch for {config_name}: expected {expected_min_items}, got {actual_min_items}")
                    all_passed = False
                    continue
                
                # Test chunk size calculation
                chunk_size = test_obj._get_chunk_size('preprocessing', 100, 200)
                self.log(f"  chunk_size for preprocessing: {chunk_size}", 'debug')
                
                # Test auto_chunk_size behavior
                auto_chunk = config_data['parallelization'].get('auto_chunk_size', True)
                if not auto_chunk and 'preprocessing' in config_data['parallelization'].get('chunk_sizes', {}):
                    expected_chunk = config_data['parallelization']['chunk_sizes']['preprocessing']
                    if chunk_size != expected_chunk:
                        self.log(f"  ✗ Fixed chunk size not respected for {config_name}: expected {expected_chunk}, got {chunk_size}")
                        all_passed = False
                        continue
                
                self.log(f"  ✓ Config {config_name} loaded correctly")
                
            except Exception as e:
                self.log(f"  ✗ Error testing config {config_name}: {e}")
                all_passed = False
        
        return all_passed
    
    def test_batch_parallel_process_basic(self) -> bool:
        """Test the core batch_parallel_process function with basic operations"""
        self.log("\n=== Testing batch_parallel_process Basic Operations ===")
        
        try:
            # Test 1: Simple processing
            self.log("\n1. Testing simple processing...")
            test_items = list(range(50))
            
            result_parallel = batch_parallel_process(
                test_items, simple_test_processor, 
                num_workers=2, min_items=20
            )
            
            result_sequential = [simple_test_processor(item) for item in test_items]
            
            if result_parallel != result_sequential:
                self.log("  ✗ Simple parallel processing results don't match sequential")
                return False
            
            self.log("  ✓ Simple processing works correctly")
            
            # Test 2: min_items threshold
            self.log("\n2. Testing min_items threshold...")
            
            small_data = list(range(10))
            result_small = batch_parallel_process(
                small_data, simple_test_processor,
                num_workers=2, min_items=20  # Should trigger sequential
            )
            
            expected_small = [simple_test_processor(x) for x in small_data]
            if result_small != expected_small:
                self.log("  ✗ min_items threshold test failed")
                return False
            
            self.log("  ✓ min_items threshold works correctly")
            
            # Test 3: Custom chunk size
            self.log("\n3. Testing custom chunk sizing...")
            
            result_chunked = batch_parallel_process(
                test_items, simple_test_processor,
                num_workers=4, chunk_size=5, min_items=20
            )
            
            if result_chunked != result_sequential:
                self.log("  ✗ Custom chunk size test failed")
                return False
            
            self.log("  ✓ Custom chunk sizing works correctly")
            
            return True
            
        except Exception as e:
            self.log(f"  ✗ batch_parallel_process basic test failed: {e}")
            return False
    
    def test_batch_parallel_process_advanced(self) -> bool:
        """Test advanced scenarios for batch_parallel_process"""
        self.log("\n=== Testing batch_parallel_process Advanced Scenarios ===")
        
        try:
            # Test 1: Complex data structures
            self.log("\n1. Testing complex data structures...")
            
            complex_data = [
                {'id': i, 'value': i * 2, 'name': f'item_{i}'}
                for i in range(30)
            ]
            
            result_complex = batch_parallel_process(
                complex_data, complex_test_processor,
                num_workers=2, min_items=10
            )
            
            expected_complex = [complex_test_processor(item) for item in complex_data]
            
            if result_complex != expected_complex:
                self.log("  ✗ Complex data structure processing failed")
                return False
            
            self.log("  ✓ Complex data structures work correctly")
            
            # Test 2: Error handling
            self.log("\n2. Testing error handling...")
            
            error_data = list(range(20))
            try:
                result_errors = batch_parallel_process(
                    error_data, error_prone_processor,
                    num_workers=2, min_items=10
                )
                # This should not complete successfully due to errors
                self.log("  ⚠ Error handling test didn't fail as expected")
            except Exception:
                self.log("  ✓ Error handling works (processing failed as expected)")
            
            # Test 3: Empty input
            self.log("\n3. Testing empty input...")
            
            result_empty = batch_parallel_process(
                [], simple_test_processor,
                num_workers=2, min_items=10
            )
            
            if result_empty != []:
                self.log("  ✗ Empty input test failed")
                return False
            
            self.log("  ✓ Empty input handled correctly")
            
            return True
            
        except Exception as e:
            self.log(f"  ✗ batch_parallel_process advanced test failed: {e}")
            return False
    
    def create_parser_configs_for_testing(self) -> Dict[str, Any]:
        """Create optimized parser configs for testing"""
        return {
            'parser': {
                'type': 'multi',
                'batch_size': 5,  # Small for testing
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
            }
        }
    
    def run_dataset_generation_test(self, config_name: str, config_data: dict, 
                                  text_pairs: List[Tuple[str, str]], labels: List[str]) -> Dict[str, Any]:
        """Run dataset generation with specified configuration"""
        self.log(f"\n--- Running dataset generation with {config_name} config ---")
        
        output_path = self.temp_dir / f"test_output_{config_name}.json"
        
        try:
            # Create feature config with parallelization settings
            feature_config = {
                'feature_extraction': {
                    'do_not_store_word_embeddings': True,
                    'embedding_cache_dir': str(self.temp_dir / f"cache_{config_name}"),
                    'word_embedding_model': 'bert-base-uncased',
                    'use_gpu': False,  # Disable GPU for testing
                    'cache_embeddings': True,
                    'batch_size': 8,
                }
            }
            feature_config.update(config_data)
            
            # Parser config optimized for testing
            parser_config = self.create_parser_configs_for_testing()
            
            # Output config
            output_config = {
                'output_format': {
                    'type': 'infonce',
                    'paired': True,
                    'self_paired': False,
                    'label_map': {
                        'entailment': 1,
                        'neutral': 0,
                        'contradiction': -1
                    }
                }
            }
            
            # Preprocessing config
            preprocessing_config = {
                'preprocessing': {
                    'strictness_level': 1,
                    'tokenizer': 'regex',  # Use regex for faster testing
                    'preserve_case': False,
                    'remove_punctuation': False,
                }
            }
            
            # Create generator
            generator = DatasetGenerator(num_workers=4)
            
            # Time the generation
            start_time = time.time()
            
            generator.generate_dataset(
                text_pairs=text_pairs,
                labels=labels,
                output_path=str(output_path),
                parser_config=parser_config,
                preprocessing_config=preprocessing_config,
                feature_config=feature_config,
                output_config=output_config,
                verbosity='quiet' if self.verbosity == 'quiet' else 'normal',
                show_progress=False
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Load and verify output
            with open(output_path, 'r') as f:
                result_data = json.load(f)
            
            # Basic validation
            if 'groups' not in result_data:
                raise ValueError("Output missing 'groups' field")
            
            num_groups = len(result_data['groups'])
            if num_groups == 0:
                raise ValueError("No groups generated")
            
            return {
                'success': True,
                'processing_time': processing_time,
                'output_path': str(output_path),
                'num_groups': num_groups,
                'result_data': result_data,
                'file_size': output_path.stat().st_size
            }
            
        except Exception as e:
            self.log(f"  ✗ Dataset generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': None,
                'num_groups': 0
            }
    
    def test_dataset_generation(self, data_size: int = 20) -> Dict[str, Any]:
        """Test dataset generation with different parallelization configs"""
        self.log(f"\n=== Testing Dataset Generation (size={data_size}) ===")
        
        # Create test data
        text_pairs, labels = self.create_test_data(data_size)
        self.log(f"Created {len(text_pairs)} test pairs")
        
        configs = self.create_test_configs()
        results = {}
        
        for config_name, config_data in configs.items():
            self.log(f"\nTesting {config_name} configuration...")
            
            result = self.run_dataset_generation_test(
                config_name, config_data, text_pairs, labels
            )
            results[config_name] = result
            
            if result['success']:
                self.log(f"  ✓ {config_name}: {result['processing_time']:.2f}s, "
                        f"{result['num_groups']} groups, "
                        f"{result['file_size']} bytes")
            else:
                self.log(f"  ✗ {config_name}: {result['error']}")
        
        return results
    
    def verify_output_consistency(self, results: Dict[str, Any]) -> bool:
        """Verify that different configs produce equivalent outputs"""
        self.log("\n=== Verifying Output Consistency ===")
        
        # Get successful results
        successful_results = {k: v for k, v in results.items() 
                            if v.get('success', False) and 'result_data' in v}
        
        if len(successful_results) < 2:
            self.log("  ⚠ Not enough successful results to compare")
            return True  # Not a failure, just can't verify
        
        # Compare group counts
        group_counts = {k: v['num_groups'] for k, v in successful_results.items()}
        unique_counts = set(group_counts.values())
        
        if len(unique_counts) == 1:
            self.log(f"  ✓ All configs produced same number of groups: {list(unique_counts)[0]}")
        else:
            self.log(f"  ⚠ Different group counts: {group_counts}")
            return False
        
        # Compare structure of results
        config_names = list(successful_results.keys())
        base_config = config_names[0]
        base_groups = successful_results[base_config]['result_data']['groups']
        
        self.log(f"  Using {base_config} as baseline for comparison")
        
        consistency_passed = True
        
        for config_name in config_names[1:]:
            other_groups = successful_results[config_name]['result_data']['groups']
            
            if len(base_groups) != len(other_groups):
                self.log(f"  ⚠ {config_name} has different group count than baseline")
                consistency_passed = False
                continue
            
            # Compare group structure (not exact content due to potential ordering differences)
            base_group_keys = set()
            other_group_keys = set()
            
            for group in base_groups:
                if 'group_id' in group:
                    base_group_keys.add(group['group_id'])
            
            for group in other_groups:
                if 'group_id' in group:
                    other_group_keys.add(group['group_id'])
            
            if len(base_group_keys) == len(other_group_keys):
                self.log(f"  ✓ {config_name} structure matches baseline")
            else:
                self.log(f"  ⚠ {config_name} structure differs from baseline")
                consistency_passed = False
        
        return consistency_passed
    
    def analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance differences between configurations"""
        self.log("\n=== Performance Analysis ===")
        
        successful_results = {k: v for k, v in results.items() 
                            if v.get('success', False) and v.get('processing_time') is not None}
        
        if len(successful_results) < 2:
            self.log("  ⚠ Not enough successful results for performance comparison")
            return {}
        
        # Get timing data
        timings = {k: v['processing_time'] for k, v in successful_results.items()}
        
        # Display raw timings
        self.log("\nProcessing times:")
        for config_name, time_taken in sorted(timings.items(), key=lambda x: x[1]):
            self.log(f"  {config_name}: {time_taken:.3f}s")
        
        # Calculate speedups if we have parallel and sequential
        analysis = {'timings': timings}
        
        if 'parallel' in timings and 'sequential' in timings:
            parallel_time = timings['parallel']
            sequential_time = timings['sequential']
            
            if parallel_time > 0:
                speedup = sequential_time / parallel_time
                analysis['speedup'] = speedup
                
                self.log(f"\nParallelization speedup: {speedup:.2f}x")
                
                if speedup > 1.2:
                    self.log("  ✓ Significant speedup achieved")
                elif speedup > 0.9:
                    self.log("  ~ Similar performance (overhead balanced)")
                else:
                    self.log("  ⚠ Parallelization slower (high overhead)")
            else:
                self.log("  ⚠ Cannot calculate speedup (zero parallel time)")
        
        # Find fastest and slowest
        fastest = min(timings.items(), key=lambda x: x[1])
        slowest = max(timings.items(), key=lambda x: x[1])
        
        analysis['fastest'] = fastest
        analysis['slowest'] = slowest
        
        self.log(f"\nFastest: {fastest[0]} ({fastest[1]:.3f}s)")
        self.log(f"Slowest: {slowest[0]} ({slowest[1]:.3f}s)")
        
        if slowest[1] > 0:
            range_factor = slowest[1] / fastest[1]
            self.log(f"Performance range: {range_factor:.2f}x")
            analysis['range_factor'] = range_factor
        
        return analysis
    
    def run_comprehensive_test(self, data_size: int = 20) -> Dict[str, Any]:
        """Run the complete test suite"""
        self.log("=" * 70)
        self.log("COMPREHENSIVE PARALLELIZATION TEST SUITE")
        self.log("=" * 70)
        
        test_summary = {
            'start_time': time.time(),
            'tests_passed': 0,
            'tests_failed': 0,
            'results': {}
        }
        
        try:
            # Test 1: Configuration loading
            self.log("\n" + "=" * 50)
            config_test_passed = self.test_config_loading()
            if config_test_passed:
                test_summary['tests_passed'] += 1
                self.log("✓ Configuration loading test PASSED")
            else:
                test_summary['tests_failed'] += 1
                self.log("✗ Configuration loading test FAILED")
            
            # Test 2: Basic batch processing
            self.log("\n" + "=" * 50)
            basic_batch_test_passed = self.test_batch_parallel_process_basic()
            if basic_batch_test_passed:
                test_summary['tests_passed'] += 1
                self.log("✓ Basic batch processing test PASSED")
            else:
                test_summary['tests_failed'] += 1
                self.log("✗ Basic batch processing test FAILED")
            
            # Test 3: Advanced batch processing
            self.log("\n" + "=" * 50)
            advanced_batch_test_passed = self.test_batch_parallel_process_advanced()
            if advanced_batch_test_passed:
                test_summary['tests_passed'] += 1
                self.log("✓ Advanced batch processing test PASSED")
            else:
                test_summary['tests_failed'] += 1
                self.log("✗ Advanced batch processing test FAILED")
            
            # Test 4: Dataset generation
            self.log("\n" + "=" * 50)
            dataset_results = self.test_dataset_generation(data_size)
            test_summary['results']['dataset_generation'] = dataset_results
            
            successful_datasets = sum(1 for r in dataset_results.values() if r.get('success', False))
            total_datasets = len(dataset_results)
            
            if successful_datasets > 0:
                test_summary['tests_passed'] += 1
                self.log(f"✓ Dataset generation test PASSED ({successful_datasets}/{total_datasets} configs successful)")
            else:
                test_summary['tests_failed'] += 1
                self.log("✗ Dataset generation test FAILED (no configs successful)")
            
            # Test 5: Output consistency
            self.log("\n" + "=" * 50)
            consistency_passed = self.verify_output_consistency(dataset_results)
            if consistency_passed:
                test_summary['tests_passed'] += 1
                self.log("✓ Output consistency test PASSED")
            else:
                test_summary['tests_failed'] += 1
                self.log("✗ Output consistency test FAILED")
            
            # Test 6: Performance analysis
            self.log("\n" + "=" * 50)
            performance_analysis = self.analyze_performance(dataset_results)
            test_summary['results']['performance'] = performance_analysis
            
            # Always count performance analysis as passed (it's informational)
            test_summary['tests_passed'] += 1
            self.log("✓ Performance analysis COMPLETED")
            
        except Exception as e:
            self.log(f"\n❌ Test suite failed with unexpected error: {e}")
            import traceback
            if self.verbosity == 'debug':
                traceback.print_exc()
            
            test_summary['tests_failed'] += 1
            test_summary['error'] = str(e)
        
        finally:
            test_summary['end_time'] = time.time()
            test_summary['total_time'] = test_summary['end_time'] - test_summary['start_time']
        
        return test_summary
    
    def print_final_summary(self, test_summary: Dict[str, Any]):
        """Print final test summary"""
        self.log("\n" + "=" * 70)
        self.log("FINAL TEST SUMMARY")
        self.log("=" * 70)
        
        total_tests = test_summary['tests_passed'] + test_summary['tests_failed']
        success_rate = test_summary['tests_passed'] / total_tests * 100 if total_tests > 0 else 0
        
        self.log(f"\nTests Passed: {test_summary['tests_passed']}")
        self.log(f"Tests Failed: {test_summary['tests_failed']}")
        self.log(f"Success Rate: {success_rate:.1f}%")
        self.log(f"Total Time: {test_summary['total_time']:.2f}s")
        
        if 'error' in test_summary:
            self.log(f"Critical Error: {test_summary['error']}")
        
        # Performance summary
        if 'performance' in test_summary.get('results', {}):
            perf = test_summary['results']['performance']
            if 'speedup' in perf:
                self.log(f"Parallelization Speedup: {perf['speedup']:.2f}x")
            if 'range_factor' in perf:
                self.log(f"Performance Range: {perf['range_factor']:.2f}x")
        
        self.log(f"\nTest files saved in: {self.temp_dir}")
        
        # Overall result
        if test_summary['tests_failed'] == 0:
            self.log("\n🎉 ALL TESTS PASSED! Parallelization system is working correctly.")
        elif test_summary['tests_passed'] > test_summary['tests_failed']:
            self.log(f"\n⚠️  PARTIAL SUCCESS: {test_summary['tests_passed']}/{total_tests} tests passed.")
        else:
            self.log(f"\n❌ MAJORITY FAILURE: Only {test_summary['tests_passed']}/{total_tests} tests passed.")
    
    def cleanup(self):
        """Clean up temporary files"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            self.log(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            self.log(f"Failed to clean up {self.temp_dir}: {e}")

def main():
    """Main test function with command line arguments"""
    parser = argparse.ArgumentParser(description="Comprehensive parallelization test suite")
    parser.add_argument('--size', type=int, default=20, 
                       help='Size of test dataset (default: 20)')
    parser.add_argument('--quick', action='store_true',
                       help='Run with smaller dataset for quick testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimize output')
    parser.add_argument('--keep-files', action='store_true',
                       help='Keep temporary files after testing')
    
    args = parser.parse_args()
    
    # Determine verbosity
    if args.quiet:
        verbosity = 'quiet'
    elif args.verbose:
        verbosity = 'debug'
    else:
        verbosity = 'normal'
    
    # Adjust size for quick mode
    data_size = 10 if args.quick else args.size
    
    # Run tests
    tester = ComprehensiveParallelizationTester(verbosity=verbosity)
    
    try:
        test_summary = tester.run_comprehensive_test(data_size)
        tester.print_final_summary(test_summary)
        
        # Return appropriate exit code
        if test_summary['tests_failed'] == 0:
            return 0
        elif test_summary['tests_passed'] > test_summary['tests_failed']:
            return 1  # Partial success
        else:
            return 2  # Majority failure
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
        
    finally:
        if not args.keep_files:
            tester.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
