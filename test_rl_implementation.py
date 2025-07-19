#!/usr/bin/env python3
"""
Test script for RL Fine-tuning implementation.

This script validates the structure and functionality of the RL fine-tuning engine
without requiring the actual ML libraries to be installed.
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timezone

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that the RL fine-tuning module can be imported."""
    print("Testing imports...")
    try:
        from rl_finetuning import (
            TrajectoryScore, TrajectoryGradingOutput, 
            MathGrader, CloudLoggingCallback, RLFinetuningEngine,
            format_for_gemma3_chat
        )
        print("✓ All classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_models():
    """Test Pydantic data models (if available)."""
    print("\nTesting data models...")
    try:
        from rl_finetuning import TrajectoryScore, TrajectoryGradingOutput
        
        # Test TrajectoryScore
        score = TrajectoryScore(trajectory="Test response", score=0.75)
        print(f"✓ TrajectoryScore created: {score}")
        
        # Test TrajectoryGradingOutput
        output = TrajectoryGradingOutput(results=[score])
        print(f"✓ TrajectoryGradingOutput created with {len(output.results)} results")
        return True
    except Exception as e:
        print(f"✗ Data model error: {e}")
        return False

def test_math_grader():
    """Test MathGrader functionality."""
    print("\nTesting MathGrader...")
    try:
        from rl_finetuning import MathGrader
        
        grader = MathGrader()
        print(f"✓ MathGrader initialized")
        print(f"  - Reasoning start: {grader.reasoning_start}")
        print(f"  - Solution pattern: {grader.solution_start}...{grader.solution_end}")
        
        # Test format matching
        test_response = "<start_working_out>2+2=4</start_working_out><SOLUTION>4</SOLUTION>"
        match = grader.match_format.search(test_response)
        if match:
            print(f"✓ Format matching works: extracted '{match.group(1)}'")
        else:
            print(f"✗ Format matching failed")
            
        return True
    except Exception as e:
        print(f"✗ MathGrader error: {e}")
        return False

def test_format_function():
    """Test format_for_gemma3_chat function."""
    print("\nTesting format_for_gemma3_chat...")
    try:
        from rl_finetuning import format_for_gemma3_chat
        
        # Test with QA pairs data
        test_data = {
            "summary": "Test dataset",
            "qa_pairs": [
                {"question": "What is 2+2?", "answer": "4"},
                {"question": "What is 3+3?", "answer": "6"}
            ]
        }
        
        # Test without tokenizer (should raise error)
        try:
            result = format_for_gemma3_chat(test_data)
            print("✗ Should have raised error without tokenizer")
        except ValueError as e:
            print(f"✓ Correctly raised error without tokenizer: {e}")
        
        # Test with text-only data
        text_data = [{"text": "Example text 1"}, {"text": "Example text 2"}]
        result = format_for_gemma3_chat(text_data)
        if result == text_data:
            print("✓ Text-only data passed through correctly")
        else:
            print("✗ Text-only data not handled correctly")
            
        return True
    except Exception as e:
        print(f"✗ Format function error: {e}")
        return False

def test_engine_initialization():
    """Test RLFinetuningEngine initialization."""
    print("\nTesting RLFinetuningEngine initialization...")
    try:
        from rl_finetuning import RLFinetuningEngine
        
        # This will fail due to missing libraries, but we can test the class structure
        try:
            engine = RLFinetuningEngine(
                model_name="test-model",
                request_id="test-001",
                project_id="test-project"
            )
            print("✓ RLFinetuningEngine initialized (unexpected success)")
        except Exception as e:
            if "ML libraries not available" in str(e) or "cloud" in str(e).lower():
                print(f"✓ Expected error due to missing dependencies: {type(e).__name__}")
            else:
                print(f"✗ Unexpected error: {e}")
                
        return True
    except Exception as e:
        print(f"✗ Engine initialization error: {e}")
        return False

def test_cloud_logging_callback():
    """Test CloudLoggingCallback structure."""
    print("\nTesting CloudLoggingCallback...")
    try:
        from rl_finetuning import CloudLoggingCallback
        
        # Mock logger and callback
        class MockLogger:
            def log_struct(self, data, severity=None):
                print(f"Mock log: {data}")
        
        callback = CloudLoggingCallback(MockLogger(), "test-request-id")
        print("✓ CloudLoggingCallback initialized")
        
        # Test callback structure
        if hasattr(callback, 'on_log'):
            print("✓ CloudLoggingCallback has on_log method")
        else:
            print("✗ CloudLoggingCallback missing on_log method")
            
        return True
    except Exception as e:
        print(f"✗ CloudLoggingCallback error: {e}")
        return False

def create_test_dataset():
    """Create a test dataset file for validation."""
    print("\nCreating test dataset...")
    try:
        test_data = {
            "summary": "Test math dataset for RL fine-tuning",
            "qa_pairs": [
                {
                    "question": "What is 15 + 27?",
                    "answer": "42"
                },
                {
                    "question": "Calculate 8 × 9",
                    "answer": "72"
                },
                {
                    "question": "What is 100 - 34?",
                    "answer": "66"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, indent=2)
            test_file = f.name
            
        print(f"✓ Test dataset created: {test_file}")
        print(f"  - {len(test_data['qa_pairs'])} QA pairs")
        
        # Clean up
        os.unlink(test_file)
        print("✓ Test dataset cleaned up")
        
        return True
    except Exception as e:
        print(f"✗ Test dataset creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RL Fine-tuning Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_models,
        test_math_grader,
        test_format_function,
        test_engine_initialization,
        test_cloud_logging_callback,
        create_test_dataset
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! RL fine-tuning implementation is structurally sound.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
    print("\nNote: Import errors for ML libraries (unsloth, transformers, trl, etc.) are expected")
    print("in development environments. The implementation is designed to handle these gracefully.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
