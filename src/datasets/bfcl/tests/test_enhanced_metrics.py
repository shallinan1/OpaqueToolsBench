#!/usr/bin/env python3
"""
Test script for enhanced BFCL metrics.
Tests the parameter accuracy and AST format analysis functionality.

Example usage:
    python -m src.datasets.bfcl.tests.test_enhanced_metrics
"""

import json
from src.datasets.bfcl.enhanced_metrics import (
    calculate_parameter_accuracy,
    calculate_ast_metrics,
    evaluate_enhanced_metrics,
    aggregate_enhanced_metrics
)


def test_parameter_accuracy():
    """Test parameter accuracy calculation."""
    print("\n=== Testing Parameter Accuracy ===")

    # Test 1: All parameters correct
    result = calculate_parameter_accuracy(
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hi"},
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hi"}
    )
    print(f"Test 1 (all correct): {result.accuracy:.2%}")
    assert result.accuracy == 1.0

    # Test 2: One parameter wrong value
    result = calculate_parameter_accuracy(
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hey"},
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hi"}
    )
    print(f"Test 2 (one wrong value): {result.accuracy:.2%}")
    assert result.accuracy == 2/3

    # Test 3: Missing parameter
    result = calculate_parameter_accuracy(
        "send_email",
        {"to": "john@example.com", "subject": "Hello"},
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hi"}
    )
    print(f"Test 3 (missing param): {result.accuracy:.2%}")
    assert result.accuracy == 2/3
    assert "body" in result.missing_params

    # Test 4: Extra parameter
    result = calculate_parameter_accuracy(
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hi", "cc": "jane@example.com"},
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hi"}
    )
    print(f"Test 4 (extra param): {result.accuracy:.2%}")
    assert result.accuracy == 1.0  # All expected params are correct
    assert "cc" in result.extra_params

    # Test 5: Wrong function name
    result = calculate_parameter_accuracy(
        "send_mail",  # Wrong function
        {"to": "john@example.com", "subject": "Hello", "body": "Hi"},
        "send_email",
        {"to": "john@example.com", "subject": "Hello", "body": "Hi"}
    )
    print(f"Test 5 (wrong function): {result.accuracy:.2%}")
    assert result.accuracy == 0.0

    print("✓ All parameter accuracy tests passed")


def test_ast_metrics():
    """Test AST/format analysis."""
    print("\n=== Testing AST/Format Analysis ===")

    # Test 1: Valid format
    model_call = {"function": "send_email", "args": {"to": "john@example.com"}}
    ground_truth = {"function": "send_email", "args": {"to": "john@example.com"}}
    result = calculate_ast_metrics(model_call, ground_truth)
    print(f"Test 1 (valid format): score={result.score:.2f}")
    assert result.format_valid == True
    assert result.structure_valid == True

    # Test 2: Invalid structure (missing args key)
    model_call = {"function": "send_email", "to": "john@example.com"}
    result = calculate_ast_metrics(model_call, ground_truth)
    print(f"Test 2 (invalid structure): score={result.score:.2f}")
    assert result.format_valid == True  # Can be parsed
    assert result.structure_valid == False  # Wrong structure

    # Test 3: Type mismatch
    model_call = {"function": "calculate", "args": {"values": "1,2,3"}}  # String instead of list
    ground_truth = {"function": "calculate", "args": {"values": [1, 2, 3]}}
    result = calculate_ast_metrics(model_call, ground_truth)
    print(f"Test 3 (type mismatch): type_correctness={result.type_correctness:.2%}")
    assert result.type_correctness < 1.0
    assert len(result.type_errors) > 0

    print("✓ All AST metric tests passed")


def test_full_evaluation():
    """Test full enhanced metrics evaluation."""
    print("\n=== Testing Full Evaluation ===")

    # Simulate a model result
    model_result = {
        "result": json.dumps([{
            "function": "calculate_total",
            "args": {
                "quantities": [1, 2, 3],
                "prices": [10.5, 20.0, 15.0]  # First price is wrong
            }
        }])
    }

    ground_truth = [{
        "function": "calculate_total",
        "args": {
            "quantities": [1, 2, 3],
            "prices": [10.0, 20.0, 15.0]
        }
    }]

    # Evaluate
    metrics = evaluate_enhanced_metrics(model_result, ground_truth)

    print(f"Parameter accuracy: {metrics['parameter_metrics']['accuracy']:.2%}")
    print(f"AST format score: {metrics['ast_metrics']['score']:.2f}")

    assert metrics['parameter_metrics']['accuracy'] == 0.5  # 1 of 2 params correct
    assert metrics['ast_metrics']['format_valid'] == True

    print("✓ Full evaluation test passed")


def test_aggregation():
    """Test metrics aggregation."""
    print("\n=== Testing Aggregation ===")

    # Create sample metrics from multiple tests
    all_metrics = [
        {
            "parameter_metrics": {"accuracy": 1.0, "missing_params": [], "extra_params": [], "wrong_value_params": []},
            "ast_metrics": {"score": 1.0, "format_valid": True, "structure_valid": True, "type_errors": []}
        },
        {
            "parameter_metrics": {"accuracy": 0.5, "missing_params": ["body"], "extra_params": [], "wrong_value_params": ["subject"]},
            "ast_metrics": {"score": 0.8, "format_valid": True, "structure_valid": True, "type_errors": []}
        },
        {
            "parameter_metrics": {"accuracy": 0.0, "missing_params": ["to", "subject"], "extra_params": ["cc"], "wrong_value_params": []},
            "ast_metrics": {"score": 0.5, "format_valid": True, "structure_valid": False, "type_errors": [{"param": "body", "expected_type": "str", "got_type": "int"}]}
        }
    ]

    aggregated = aggregate_enhanced_metrics(all_metrics)

    print(f"Average parameter accuracy: {aggregated['parameter_accuracy_avg']:.2%}")
    print(f"Average AST format score: {aggregated['ast_format_score_avg']:.2%}")
    print(f"Common missing params: {aggregated['common_errors']['missing_parameters']}")

    assert aggregated['parameter_accuracy_avg'] == 0.5  # (1.0 + 0.5 + 0.0) / 3
    assert aggregated['ast_format_score_avg'] > 0

    print("✓ Aggregation test passed")


if __name__ == "__main__":
    test_parameter_accuracy()
    test_ast_metrics()
    test_full_evaluation()
    test_aggregation()
    print("\n" + "="*50)
    print("✅ All tests passed successfully!")
    print("The enhanced metrics implementation is working correctly.")