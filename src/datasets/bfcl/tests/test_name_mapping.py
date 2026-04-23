#!/usr/bin/env python3
"""
Test script to verify name_mapping handling in enhanced metrics.

Example usage:
    python -m src.datasets.bfcl.tests.test_name_mapping
"""

import json
from src.datasets.bfcl.enhanced_metrics import (
    calculate_parameter_accuracy,
    calculate_ast_metrics,
    evaluate_enhanced_metrics
)


def test_name_mapping():
    """Test that name_mapping correctly maps obfuscated to real function names."""
    print("\n=== Testing Name Mapping ===")

    # Test data mimicking real BFCL results
    name_mapping = {
        "calc_binomial_probability": "function_2",
        "get_weather_data": "function_1"
    }

    # Test 1: Correct obfuscated function with parameters
    print("\n1. Testing correct obfuscated function with parameters:")

    # Model uses obfuscated name
    model_func = "function_2"
    model_args = {"n": 20, "k": 5, "p": 0.16666666666666666}

    # Ground truth uses real name
    gt_func = "calc_binomial_probability"
    gt_args = {"n": 20, "k": 5, "p": 0.16666666666666666}

    result = calculate_parameter_accuracy(model_func, model_args, gt_func, gt_args, name_mapping)
    print(f"   Parameter accuracy: {result.accuracy:.2%}")
    assert result.accuracy == 1.0, f"Expected 100% accuracy, got {result.accuracy}"
    print("   ✓ Test passed!")

    # Test 2: Wrong obfuscated function
    print("\n2. Testing wrong obfuscated function:")

    model_func = "function_1"  # Wrong obfuscated function
    result = calculate_parameter_accuracy(model_func, model_args, gt_func, gt_args, name_mapping)
    print(f"   Parameter accuracy: {result.accuracy:.2%}")
    assert result.accuracy == 0.0, f"Expected 0% accuracy, got {result.accuracy}"
    print("   ✓ Test passed!")

    # Test 3: AST metrics with name mapping
    print("\n3. Testing AST metrics with name mapping:")

    model_call = {"function": "function_2", "args": {"n": 20, "k": 5, "p": 0.16666666666666666}}
    gt_call = {"function": "calc_binomial_probability", "args": {"n": 20, "k": 5, "p": 0.16666666666666666}}

    ast_result = calculate_ast_metrics(model_call, gt_call, None, name_mapping)
    print(f"   AST score: {ast_result.score:.2f}")
    print(f"   Type correctness: {ast_result.type_correctness:.2%}")
    assert ast_result.type_correctness == 1.0, f"Expected 100% type correctness with name mapping"
    print("   ✓ Test passed!")

    # Test 4: Full evaluation with name mapping
    print("\n4. Testing full evaluation with name mapping:")

    model_result = {
        "result": json.dumps([{
            "function": "function_2",
            "args": {"n": 20, "k": 5, "p": 0.16666666666666666}
        }]),
        "name_mapping": name_mapping
    }

    ground_truth = [{
        "function": "calc_binomial_probability",
        "args": {"n": 20, "k": 5, "p": 0.16666666666666666}
    }]

    metrics = evaluate_enhanced_metrics(model_result, ground_truth, None, name_mapping)

    print(f"   Parameter accuracy: {metrics['parameter_metrics']['accuracy']:.2%}")
    print(f"   AST score: {metrics['ast_metrics']['score']:.2f}")

    assert metrics['parameter_metrics']['accuracy'] == 1.0, "Parameter accuracy should be 100%"
    assert metrics['ast_metrics']['type_correctness'] == 1.0, "Type correctness should be 100%"
    print("   ✓ All metrics correctly calculated with name mapping!")

    print("\n✅ All name mapping tests passed!")


def test_without_name_mapping():
    """Test behavior without name mapping (original functionality)."""
    print("\n=== Testing Without Name Mapping ===")

    model_func = "send_email"
    model_args = {"to": "john@example.com", "subject": "Hello", "body": "Hi"}
    gt_func = "send_email"
    gt_args = {"to": "john@example.com", "subject": "Hello", "body": "Hi"}

    result = calculate_parameter_accuracy(model_func, model_args, gt_func, gt_args, None)
    print(f"Parameter accuracy (no mapping): {result.accuracy:.2%}")
    assert result.accuracy == 1.0
    print("✓ Non-obfuscated test passed!")


if __name__ == "__main__":
    test_name_mapping()
    test_without_name_mapping()
    print("\n" + "="*50)
    print("✅ All tests passed! Name mapping is working correctly.")
    print("The enhanced metrics now properly handle obfuscated function names.")