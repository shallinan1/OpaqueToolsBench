"""
BFCL-specific argument extensions for the shared dataset runner.
"""

import argparse
from pathlib import Path

def add_bfcl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add BFCL-specific arguments to a parser.
    
    Args:
        parser: Base argument parser (typically from run_args.py)
        
    Returns:
        Parser with BFCL-specific arguments added
    """
    
    # BFCL-specific model configuration (override defaults from run_args)
    parser.set_defaults(
        temperature=0.001,
        top_p=1.0,
        model="gpt-5-mini"
    )
    
    # Add max-tokens which is BFCL specific
    parser.add_argument("--max-tokens", type=int, default=1200, help="Maximum tokens for response (default: 1200)")
    parser.add_argument("--prompt-key", type=str, default="must_call_tool", help="Key of the prompt to use (default: must_call_tool)")

    # API provider selection
    parser.add_argument("--together", action="store_true", help="Use Together AI API instead of OpenAI API")
    
    # Execution configuration
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout in seconds (default: 60)")
    parser.add_argument("--num-threads", type=int, default=1, help="Number of parallel threads (default: 1)")
    
    # Debug/verbose output
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")

    # Cross-model testing
    parser.add_argument("--cross-model-source", type=Path, default=None,
                       help="Path to a config from another model's iterate-improve run (e.g., .../improvements/.../v10/config.json). "
                            "Runs the target --model on this config and saves results under a cross_model/ subdirectory.")

    # Reasoning effort for reasoning models
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                       choices=["none", "minimal", "low", "medium", "high"],
                       help="Constrains effort on reasoning for reasoning models like gpt-5/gpt-5.1. GPT-5 default is 'medium', GPT-5.1 default is 'none' (default: medium)")

    return parser

def create_bfcl_parser() -> argparse.ArgumentParser:
    from src.datasets.run_args import parser as shared_parser
    parser = argparse.ArgumentParser(
        parents=[shared_parser],
        description="Run BFCL evaluation"
    )
    parser = add_bfcl_args(parser)    
    return parser