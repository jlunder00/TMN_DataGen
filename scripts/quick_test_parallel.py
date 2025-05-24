#!/usr/bin/env python3
"""
Quick test to verify basic parallelization functionality
"""

import time
from TMN_DataGen.utils.parallel_framework import batch_parallel_process, ParallelizationMixin
from omegaconf import OmegaConf

def simple_processor(x):
    """Simple processor function - must be at module level for pickling"""
    return x * 2

def test_basic_functionality():
    """Test basic parallelization components"""
    print("=== Quick Parallelization Test ===")
    
    # Test 1: ParallelizationMixin configuration
    print("\n1. Testing configuration loading...")
    
    config = {
        'parallelization': {
            'min_items_for_parallel': 25,
            'auto_chunk_size': True,
            'chunk_sizes': {
                'test_operation': 10
            }
        }
    }
    
    class TestClass(ParallelizationMixin):
        def __init__(self, config):
            self.config = OmegaConf.create(config)
            self.num_workers = 2
            super().__init__()
    
    test_obj = TestClass(config)
    
    # Check min_items_for_parallel
    min_items = test_obj._get_min_items_for_parallel()
    print(f"   min_items_for_parallel: {min_items}")
    assert min_items == 25, f"Expected 25, got {min_items}"
    
    # Check chunk size calculation
    chunk_size = test_obj._get_chunk_size('test_operation', 50, 100)
    print(f"   chunk_size: {chunk_size}")
    
    print("   ✓ Configuration loading works")
    
    # Test 2: batch_parallel_process
    print("\n2. Testing batch_parallel_process...")
    
    test_data = list(range(50))
    
    start_time = time.time()
    result_parallel = batch_parallel_process(
        test_data, simple_processor, 
        num_workers=2, min_items=20
    )
    parallel_time = time.time() - start_time
    
    start_time = time.time()
    result_sequential = [simple_processor(x) for x in test_data]
    sequential_time = time.time() - start_time
    
    print(f"   Parallel time: {parallel_time:.4f}s")
    print(f"   Sequential time: {sequential_time:.4f}s")
    
    assert result_parallel == result_sequential, "Results don't match!"
    print("   ✓ batch_parallel_process works correctly")
    
    # Test 3: min_items threshold
    print("\n3. Testing min_items threshold...")
    
    small_data = list(range(10))
    result_small = batch_parallel_process(
        small_data, simple_processor,
        num_workers=2, min_items=20  # Should trigger sequential
    )
    
    expected_small = [simple_processor(x) for x in small_data]
    assert result_small == expected_small, "Small data test failed!"
    print("   ✓ min_items threshold works correctly")
    
    print("\n=== All Quick Tests Passed! ===")

if __name__ == "__main__":
    test_basic_functionality()
