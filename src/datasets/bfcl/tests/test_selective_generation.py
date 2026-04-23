#!/usr/bin/env python3
"""
Test script to verify selective generation and evaluation works correctly.
"""

import json
from pathlib import Path
from src.datasets.bfcl.utils.function_utils import functions_are_identical

def test_functions_are_identical():
    """Test the function comparison logic."""

    # Test 1: Identical functions
    tools1 = [
        {"name": "func1", "description": "Does something"},
        {"name": "func2", "description": "Does something else"}
    ]
    tools2 = [
        {"name": "func1", "description": "Does something"},
        {"name": "func2", "description": "Does something else"}
    ]
    assert functions_are_identical(tools1, tools2) == True, "Identical functions should match"

    # Test 2: Different descriptions
    tools3 = [
        {"name": "func1", "description": "Does something different"},
        {"name": "func2", "description": "Does something else"}
    ]
    assert functions_are_identical(tools1, tools3) == False, "Different descriptions should not match"

    # Test 3: Different names
    tools4 = [
        {"name": "func3", "description": "Does something"},
        {"name": "func2", "description": "Does something else"}
    ]
    assert functions_are_identical(tools1, tools4) == False, "Different names should not match"

    # Test 4: Different order but same functions
    tools5 = [
        {"name": "func2", "description": "Does something else"},
        {"name": "func1", "description": "Does something"}
    ]
    assert functions_are_identical(tools1, tools5) == True, "Same functions in different order should match"

    # Test 5: Different number of functions
    tools6 = [
        {"name": "func1", "description": "Does something"}
    ]
    assert functions_are_identical(tools1, tools6) == False, "Different number of functions should not match"

    # Test 6: Empty lists
    assert functions_are_identical([], []) == True, "Empty lists should match"

    # Test 7: One empty, one not
    assert functions_are_identical([], tools1) == False, "Empty vs non-empty should not match"

    print("✅ All function comparison tests passed!")


def test_config_comparison():
    """Test comparing configs from actual BFCL format."""

    config1 = {
        "tests": [
            {
                "tools": [
                    {"name": "get_weather", "description": "Get weather for a location"},
                    {"name": "get_time", "description": "Get current time"}
                ]
            },
            {
                "tools": [
                    {"name": "calculate_tax", "description": "Calculate tax amount"}
                ]
            }
        ]
    }

    # Same config
    config2 = json.loads(json.dumps(config1))  # Deep copy

    # Check all tests are identical
    for i in range(len(config1["tests"])):
        assert functions_are_identical(
            config1["tests"][i]["tools"],
            config2["tests"][i]["tools"]
        ) == True, f"Test {i} should be identical"

    # Modified config - change one description
    config3 = json.loads(json.dumps(config1))
    config3["tests"][0]["tools"][0]["description"] = "Get weather information for a city"

    # First test should be different, second should be same
    assert functions_are_identical(
        config1["tests"][0]["tools"],
        config3["tests"][0]["tools"]
    ) == False, "Modified test should not match"

    assert functions_are_identical(
        config1["tests"][1]["tools"],
        config3["tests"][1]["tools"]
    ) == True, "Unmodified test should still match"

    print("✅ Config comparison tests passed!")


def main():
    """Run all tests."""
    print("Testing selective generation functionality...")
    print()

    test_functions_are_identical()
    print()

    test_config_comparison()
    print()

    print("🎉 All tests passed successfully!")
    print()
    print("The implementation should correctly:")
    print("1. Compare function descriptions between configs")
    print("2. Reuse results for unchanged tests")
    print("3. Only generate/evaluate changed tests")
    print()
    print("To test with real data, run an iterative improvement and check the logs for:")
    print("- 'Reusing result (functions unchanged)' messages in run.py")
    print("- 'Reusing evaluation (functions unchanged)' messages in evaluate.py")


if __name__ == "__main__":
    main()