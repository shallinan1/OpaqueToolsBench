"""
Generate tool configuration files for Chess tests.

This generates configs from chess tool sets and test data, similar to how BFCL generates configs.

Example usage:
    # Generate configs for all tool sets with first 100 rows
    python -m src.datasets.chess.generate_configs \
        --tool-sets all \
        --num-tests 50 \
        --output-dir src/datasets/chess/individual_tools

    # Generate config for specific tool set
    python -m src.datasets.chess.generate_configs \
        --tool-sets random_vs_best01_accurate \
        --num-tests 50 \
        --output-dir src/datasets/chess/individual_tools
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_tool_set(tool_set_name: str) -> Dict:
    """Load a tool set configuration from the shared_tools directory."""
    tool_sets_dir = Path("src/datasets/chess/shared_tools")

    # Handle file name with or without .json extension
    if not tool_set_name.endswith('.json'):
        tool_set_name = f"{tool_set_name}.json"

    tool_set_path = tool_sets_dir / tool_set_name

    if not tool_set_path.exists():
        raise ValueError(f"Tool set not found: {tool_set_path}")

    with open(tool_set_path, 'r') as f:
        return json.load(f)


def load_test_data(num_tests: int) -> List[Dict]:
    """Load the first num_tests rows from the train.jsonl file."""
    data_path = Path("src/datasets/chess/data/train.jsonl")

    if not data_path.exists():
        raise ValueError(f"Data file not found: {data_path}")

    test_data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_tests:
                break
            test_data.append(json.loads(line))

    return test_data


def generate_test_config(test_item: Dict, test_index: int, tools: List[Dict]) -> Dict:
    """Generate a configuration for a single test.

    Returns a config similar to BFCL tool configs.
    """
    # Create config in BFCL style
    config = {
        "test_id": test_item.get("test_id", f"chess_{test_index:05d}"),
        "question": f"Find the best move for the position: {test_item['fen']}",
        "tools": tools,
        "ground_truth": [],  # Chess doesn't have ground truth function calls
        "metadata": {
            "category": "chess",
            "fen": test_item["fen"],
            "phase": test_item.get("phase", ""),
            "evaluation": test_item.get("evaluation", ""),
            "cp": test_item.get("cp"),
            "mate": test_item.get("mate"),
            "depth": test_item.get("depth")
        }
    }

    return config


def generate_tool_set_configs(
    tool_set_name: str,
    num_tests: int,
    output_dir: Path
) -> Dict:
    """Generate configs for a specific tool set."""

    # Load tool set
    tool_set = load_tool_set(tool_set_name)

    # Load test data
    test_data = load_test_data(num_tests)

    # Extract tools from tool set
    tools = tool_set.get("tools", [])

    # Generate config for each test
    configs = []
    for i, test_item in enumerate(test_data):
        config = generate_test_config(test_item, i, tools)
        configs.append(config)

    # Create summary config (like BFCL)
    # Remove .json extension if present for config name
    clean_tool_set_name = tool_set_name.replace('.json', '')

    summary = {
        "config_name": f"{clean_tool_set_name}_{num_tests}tests",
        "config_description": f"Tool configurations for chess tests using {clean_tool_set_name} tool set",
        "test_category": "chess",
        "tool_set": clean_tool_set_name,
        "num_tests": len(configs),
        "tools": tools,
        "tests": configs
    }

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    config_filename = f"{clean_tool_set_name}_{num_tests}tests_config.json"
    config_file = output_dir / config_filename

    with open(config_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Generated config for {clean_tool_set_name}: {config_file}")
    print(f"  - {len(configs)} test configurations")
    print(f"  - {len(tools)} tools available")

    return summary


def get_all_tool_sets() -> List[str]:
    """Get all available tool set names."""
    tool_sets_dir = Path("src/datasets/chess/shared_tools")

    if not tool_sets_dir.exists():
        return []

    # Get all .json files in the directory
    tool_sets = []
    for file_path in tool_sets_dir.glob("*.json"):
        # Remove .json extension for cleaner names
        tool_sets.append(file_path.stem)

    return sorted(tool_sets)


def main():
    parser = argparse.ArgumentParser(description="Generate Chess tool configurations")
    parser.add_argument("--tool-sets", type=str, nargs="+", default=["all"], help="Tool sets to generate configs for (use 'all' for all available sets)")
    parser.add_argument("--num-tests", type=int, default=50, help="Number of test cases to include (default: 50)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for configs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Determine which tool sets to process
    if "all" in args.tool_sets:
        tool_sets = get_all_tool_sets()
        if not tool_sets:
            print("No tool sets found in src/datasets/chess/shared_tools/")
            return
        print(f"Found {len(tool_sets)} tool sets: {', '.join(tool_sets)}")
    else:
        tool_sets = args.tool_sets

    # Generate configs for each tool set
    for tool_set in tool_sets:
        try:
            generate_tool_set_configs(
                tool_set,
                args.num_tests,
                output_dir
            )
        except Exception as e:
            print(f"Error generating config for {tool_set}: {e}")
            continue

    print(f"\nAll configs saved to: {output_dir}")

if __name__ == "__main__":
    main()