"""
Chess-specific argument extensions for the shared dataset runner.
"""

import argparse
from src.datasets.chess.prompts import CHESS_AGENT_PROMPTS

def add_chess_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add chess-specific arguments to a parser.

    Args:
        parser: Base argument parser (typically from run_args.py)

    Returns:
        Parser with chess-specific arguments added
    """

    # Get available chess tools from __all__
    from src.datasets.chess.chess_tools import __all__ as available_tools

    # Player choices are "agent" (uses config tools), "probabilistic" (random tool selection), baseline players, or specific tool names
    player_choices = ["agent", "probabilistic", "best", "worst"] + available_tools

    # Chess-specific defaults (override defaults from run_args)
    parser.set_defaults(model="gpt-5", temperature=1.0, output_dir="runs/chess/tool_observer")

    # Tool configuration mode (mutually exclusive)
    tool_group = parser.add_mutually_exclusive_group(required=True)
    tool_group.add_argument("--individual-tools", type=str, dest="tool_config",
                           help="Path to individual tools config JSON with per-problem tools (e.g., individual_tools/random_vs_best01_accurate_50tests_config.json)")
    tool_group.add_argument("--shared-tools", type=str, dest="tool_set",
                           help="Path to shared tools JSON for all problems (e.g., shared_tools/random_vs_best01_accurate.json)")

    # Chess-specific arguments
    parser.add_argument("--prompt-key", type=str, default="optimized_single",
                       choices=list(CHESS_AGENT_PROMPTS.keys()),
                       help="Key of the prompt to use (default: optimized_single, paper canonical)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                       help="Maximum tokens for response (default: 8192)")
    parser.add_argument("--provider", type=str, default="openai",
                       choices=["openai", "anthropic"],
                       help="LLM provider to use (default: openai)")
    parser.add_argument("--max-retries", type=int, default=3,
                       help="Maximum number of retries for illegal moves (default: 3)")
    parser.add_argument("--timeout", type=int, default=60,
                       help="Request timeout in seconds (default: 60)")

    # Game configuration for chess trajectories
    parser.add_argument("--white-type", type=str, default="agent",
                       choices=player_choices,
                       help="Type of player for white: 'agent' uses tools from config, 'probabilistic' uses weighted random tool selection, or specify a tool function name to use exclusively")
    parser.add_argument("--black-type", type=str, default="agent",
                       choices=player_choices,
                       help="Type of player for black: 'agent' uses tools from config, 'probabilistic' uses weighted random tool selection, or specify a tool function name to use exclusively")
    parser.add_argument("--max-moves", type=int, default=120,
                       help="Maximum number of moves before declaring a draw (default: 120)")
    parser.add_argument("--white-tool-probabilities", type=str,
                       help="For probabilistic white player: comma-separated list of tool:probability pairs (e.g., 'elo_1200:0.7,elo_1800:0.3')")
    parser.add_argument("--black-tool-probabilities", type=str,
                       help="For probabilistic black player: comma-separated list of tool:probability pairs (e.g., 'elo_1200:0.7,elo_1800:0.3')")

    # Starting position configuration
    parser.add_argument("--start-fen", type=str, nargs='+',
                       default=None,
                       help="FEN string(s) for starting positions. Can specify multiple FENs (default: use split if specified, otherwise standard position)")

    # Data split configuration
    parser.add_argument("--split", type=str, choices=["train", "test"],
                       help="Load FEN positions from train or test split. Overrides --start-fen")

    parser.add_argument("--num-trajectories", type=int, default=1,
                       help="Number of game trajectories to run per FEN position (default: 1)")
    parser.add_argument("--mirror", action="store_true",
                       help="Also run each position with white and black player types swapped (doubles the number of games)")
    parser.add_argument("--include-history", action="store_true",
                       help="Include move history and tool call history in conversation context")

    # Reasoning effort for reasoning models
    parser.add_argument("--reasoning-effort", type=str, default="medium",
                       choices=["minimal", "low", "medium", "high"],
                       help="Constrains effort on reasoning for reasoning models. Reducing reasoning effort can result in faster responses and fewer tokens used (default: medium)")

    # Parallel execution configuration
    parser.add_argument("--parallel-workers", type=int, default=12,
                       help="Number of parallel tool executors and session pool size (default: 12)")

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility (default: 0)")

    # Debug/verbose output
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser

def create_chess_parser() -> argparse.ArgumentParser:
    from src.datasets.run_args import parser as shared_parser
    parser = argparse.ArgumentParser(
        parents=[shared_parser],
        description="Collect chess trajectories"
    )
    parser = add_chess_args(parser)
    return parser
