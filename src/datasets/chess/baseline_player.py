"""
Baseline player policies for deterministic tool selection.

This module implements "best" and "worst" baseline players that select tools
based on game phase, depth, or Elo ratings.
"""

import re
import random
import logging
from typing import List, Dict, Optional, Tuple, Set
import chess

from src.datasets.chess.process_data import get_game_phase_fast

logger = logging.getLogger(__name__)


class ToolClassification:
    """Classification of available tools by type."""

    def __init__(self):
        self.specialists: Dict[str, str] = {}  # phase -> tool_name
        self.depth_tools: Dict[int, str] = {}  # depth -> tool_name
        self.elo_tools: Dict[int, str] = {}  # elo_rating -> tool_name
        self.worst_tools: List[str] = []  # worst_move tools
        self.random_tools: List[str] = []  # random_move tools
        self.other_tools: List[str] = []  # unclassified tools


def resolve_tool_name(tool_name: str, function_mapping: Optional[Dict] = None) -> str:
    """Resolve tool name using function_mapping if available.

    Args:
        tool_name: Tool name (possibly obfuscated like "function_1")
        function_mapping: Optional mapping from obfuscated to real names

    Returns:
        Real tool name
    """
    if function_mapping and tool_name in function_mapping:
        return function_mapping[tool_name]
    return tool_name


def classify_tools(tools: List[Dict], function_mapping: Optional[Dict] = None) -> ToolClassification:
    """Classify tools by type (specialist, depth, Elo, etc.).

    Args:
        tools: List of tool definitions (OpenAI format)
        function_mapping: Optional mapping from obfuscated to real names

    Returns:
        ToolClassification with categorized tools
    """
    classification = ToolClassification()

    for tool in tools:
        tool_name = tool.get("name") or tool.get("function", {}).get("name")
        if not tool_name:
            continue

        # Resolve real name if obfuscated
        real_name = resolve_tool_name(tool_name, function_mapping)

        # Classify by pattern matching on real name
        if "opening_specialist" in real_name.lower():
            classification.specialists["opening"] = tool_name
        elif "middlegame_specialist" in real_name.lower():
            classification.specialists["middlegame"] = tool_name
        elif "endgame_specialist" in real_name.lower() and "late" not in real_name.lower():
            classification.specialists["endgame"] = tool_name
        elif "late_endgame_specialist" in real_name.lower():
            classification.specialists["late_endgame"] = tool_name

        # Match depth tools (e.g., "best_move_depth_16")
        depth_match = re.search(r'depth[_\s]*(\d+)', real_name.lower())
        if depth_match:
            depth = int(depth_match.group(1))
            classification.depth_tools[depth] = tool_name

        # Match Elo tools (e.g., "elo_2320")
        elo_match = re.search(r'elo[_\s]*(\d+)', real_name.lower())
        if elo_match:
            elo = int(elo_match.group(1))
            classification.elo_tools[elo] = tool_name

        # Match explicit worst/random tools
        if "worst_move" in real_name.lower():
            classification.worst_tools.append(tool_name)
        elif "random_move" in real_name.lower():
            classification.random_tools.append(tool_name)

    return classification


def select_best_tool(board: chess.Board, tools: List[Dict],
                     function_mapping: Optional[Dict] = None) -> str:
    """Select the best tool for the current position.

    Args:
        board: Current chess board state
        tools: Available tools
        function_mapping: Optional mapping from obfuscated to real names

    Returns:
        Tool name to use (possibly obfuscated)

    Raises:
        ValueError: If no appropriate best tool can be determined
    """
    classification = classify_tools(tools, function_mapping)

    # Strategy 1: Specialist tools - match by phase
    if classification.specialists:
        phase = get_game_phase_fast(board.fen())
        if phase in classification.specialists:
            best_tool = classification.specialists[phase]
            logger.debug(f"Best tool: {best_tool} (phase={phase})")
            return best_tool
        else:
            # Phase detected but no matching specialist - this shouldn't happen
            # if the config is well-formed, but we'll try other strategies
            logger.warning(f"Phase {phase} detected but no matching specialist in {classification.specialists.keys()}")

    # Strategy 2: Depth tools - use highest depth
    if classification.depth_tools:
        max_depth = max(classification.depth_tools.keys())
        best_tool = classification.depth_tools[max_depth]
        logger.debug(f"Best tool: {best_tool} (depth={max_depth})")
        return best_tool

    # Strategy 3: Elo tools - use highest Elo
    if classification.elo_tools:
        max_elo = max(classification.elo_tools.keys())
        best_tool = classification.elo_tools[max_elo]
        logger.debug(f"Best tool: {best_tool} (elo={max_elo})")
        return best_tool

    # No recognizable tool pattern
    raise ValueError(
        f"Cannot determine 'best' tool from available tools. "
        f"Config must contain specialists, depth tools, or Elo tools. "
        f"Found: {[t.get('name') or t.get('function', {}).get('name') for t in tools]}"
    )


def select_worst_tool(board: chess.Board, tools: List[Dict],
                      function_mapping: Optional[Dict] = None) -> str:
    """Select the worst tool for the current position.

    For specialists: Randomly selects from specialists that DON'T match current phase.
    For depth tools: Selects lowest depth.
    For Elo tools: Selects lowest Elo.
    Prefers explicit worst_move if available.

    Args:
        board: Current chess board state
        tools: Available tools
        function_mapping: Optional mapping from obfuscated to real names

    Returns:
        Tool name to use (possibly obfuscated)

    Raises:
        ValueError: If no appropriate worst tool can be determined
    """
    classification = classify_tools(tools, function_mapping)

    # Strategy 0: Use explicit worst_move if available
    if classification.worst_tools:
        # If there's only one, use it; otherwise randomly select
        worst_tool = random.choice(classification.worst_tools)
        logger.debug(f"Worst tool: {worst_tool} (explicit worst_move)")
        return worst_tool

    # Strategy 1: Specialist tools - select from NON-matching phases
    if classification.specialists:
        phase = get_game_phase_fast(board.fen())
        # Get all specialists that DON'T match current phase
        wrong_specialists = [
            tool_name for phase_name, tool_name in classification.specialists.items()
            if phase_name != phase
        ]

        if wrong_specialists:
            worst_tool = random.choice(wrong_specialists)
            logger.debug(f"Worst tool: {worst_tool} (wrong phase, current={phase})")
            return worst_tool
        else:
            # Only one specialist and it matches - fall back to random if available
            if classification.random_tools:
                worst_tool = random.choice(classification.random_tools)
                logger.debug(f"Worst tool: {worst_tool} (fallback to random)")
                return worst_tool

    # Strategy 2: Depth tools - use lowest depth
    if classification.depth_tools:
        min_depth = min(classification.depth_tools.keys())
        worst_tool = classification.depth_tools[min_depth]
        logger.debug(f"Worst tool: {worst_tool} (depth={min_depth})")
        return worst_tool

    # Strategy 3: Elo tools - use lowest Elo
    if classification.elo_tools:
        min_elo = min(classification.elo_tools.keys())
        worst_tool = classification.elo_tools[min_elo]
        logger.debug(f"Worst tool: {worst_tool} (elo={min_elo})")
        return worst_tool

    # Strategy 4: Random move as last resort
    if classification.random_tools:
        worst_tool = random.choice(classification.random_tools)
        logger.debug(f"Worst tool: {worst_tool} (random as worst)")
        return worst_tool

    # No recognizable tool pattern
    raise ValueError(
        f"Cannot determine 'worst' tool from available tools. "
        f"Config must contain specialists, depth tools, Elo tools, worst_move, or random_move. "
        f"Found: {[t.get('name') or t.get('function', {}).get('name') for t in tools]}"
    )
