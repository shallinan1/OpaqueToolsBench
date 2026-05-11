"""
Probabilistic tool selection utilities for chess agents.

Supports weighted random tool selection where agents pick tools based on
specified probabilities.
"""

import random
from typing import Dict, Optional


def parse_tool_probabilities(prob_string: Optional[str]) -> Dict[str, float]:
    """Parse tool probability string and normalize probabilities.

    Args:
        prob_string: Comma-separated tool:probability pairs
                    e.g., "elo_1200:0.7,elo_1800:0.3"
                    or "elo_1200:1,elo_1800:1,elo_2400:1"

    Returns:
        Dictionary mapping tool names to normalized probabilities
    """
    if not prob_string:
        return {}

    tool_probs = {}
    total_weight = 0.0

    # Parse the string
    for pair in prob_string.split(','):
        pair = pair.strip()
        if ':' not in pair:
            raise ValueError(f"Invalid probability pair: '{pair}'. Expected format: 'tool_name:probability'")

        tool_name, prob_str = pair.split(':', 1)
        tool_name = tool_name.strip()
        prob_str = prob_str.strip()

        try:
            prob = float(prob_str)
        except ValueError:
            raise ValueError(f"Invalid probability value: '{prob_str}' for tool '{tool_name}'")

        if prob < 0:
            raise ValueError(f"Probability must be non-negative, got {prob} for tool '{tool_name}'")

        tool_probs[tool_name] = prob
        total_weight += prob

    # Normalize probabilities
    if total_weight == 0:
        raise ValueError("Total probability weight cannot be zero")

    normalized_probs = {
        tool: prob / total_weight
        for tool, prob in tool_probs.items()
    }

    return normalized_probs


def select_tool_probabilistic(tool_probs: Dict[str, float]) -> str:
    """Select a tool based on probability distribution.

    Args:
        tool_probs: Dictionary mapping tool names to probabilities (must sum to 1.0)

    Returns:
        Selected tool name
    """
    if not tool_probs:
        raise ValueError("No tool probabilities provided")

    # Use random.choices for weighted selection
    tools = list(tool_probs.keys())
    weights = list(tool_probs.values())

    return random.choices(tools, weights=weights, k=1)[0]


def format_probabilities_for_path(tool_probs: Dict[str, float]) -> str:
    """Format tool probabilities as a compact string for directory names.

    Args:
        tool_probs: Dictionary mapping tool names to probabilities

    Returns:
        Compact string representation (e.g., "e1200_70_e1800_30" for 70% elo_1200, 30% elo_1800)
    """
    if not tool_probs:
        return "uniform"

    # Sort by tool name for consistency
    sorted_tools = sorted(tool_probs.items())

    parts = []
    for tool, prob in sorted_tools:
        # Shorten tool name for path
        # e.g., "best_move_depth_8" -> "d8"
        if tool.startswith("best_move_depth_"):
            short = "d" + tool.replace("best_move_depth_", "")
        elif tool.startswith("elo_"):
            short = "e" + tool.replace("elo_", "")
        elif tool.endswith("_specialist"):
            # e.g., "opening_specialist" -> "opening"
            short = tool.replace("_specialist", "")
        else:
            # Use first few chars
            short = tool[:8]

        # Format probability as integer percentage
        prob_pct = int(prob * 100)
        parts.append(f"{short}{prob_pct}")

    return "_".join(parts)
