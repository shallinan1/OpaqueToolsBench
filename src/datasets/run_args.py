"""
Shared argument parser for dataset evaluation scripts.
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)

# Model and generation parameters
parser.add_argument("--model", default="gpt-4o-2024-08-06",
                   help="OpenAI chat model used for reasoning and tool use)")
parser.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for the model (default: 1.0)")
parser.add_argument("--top-p", type=float, default=1.0, 
                   help="Top P for the model (default: 1.0)")
parser.add_argument("--tool-choice", default="required",
                   choices=["auto", "none", "required"],
                   help="How the model decides to use tools: 'auto' (model decides), 'none' (no tools), 'required' (must use tools) (default: required)")

# Evaluation configuration
parser.add_argument("--num-queries", type=int, default=None,
                   help="Number of queries to process (default: all available queries)")

# Config source
parser.add_argument("--config-source", type=Path, default=None,
                   help="Path to config JSON file (e.g., tool_configs/.../base.json or runs/.../improved_configs/gpt4o_t07_basic.json or src/datasets/bfcl/tool_configs/base/simple_base_config.json)")

# Output settings
parser.add_argument("--output-dir", type=Path, default=None,
                   help="Directory to save results (each dataset will use its own default)")