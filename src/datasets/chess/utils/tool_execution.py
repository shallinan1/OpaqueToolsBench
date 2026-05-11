"""
Tool execution utilities for chess trajectory collection.

This module provides functions for executing chess tools both individually
and in parallel batches.
"""

import logging
from typing import List, Tuple, Dict, Optional
import chess

from src.datasets.chess import chess_tools
from src.datasets.chess.utils.session_management import SessionPool, GameState

logger = logging.getLogger(__name__)


def execute_tool_directly(tool_name: str, game_board: chess.Board, session_pool: SessionPool,
                         function_mapping: Optional[Dict[str, str]] = None) -> str:
    """Execute a chess tool directly using session pool.

    Args:
        tool_name: Name of the tool to execute
        game_board: The game's board with full history
        session_pool: Pool of sessions for tool execution
        function_mapping: Optional mapping for obfuscated names

    Returns:
        The move in algebraic notation or error message
    """
    # Resolve obfuscated name if mapping provided
    actual_tool_name = function_mapping.get(tool_name, tool_name) if function_mapping else tool_name

    # Get the function from chess_tools
    if not hasattr(chess_tools, actual_tool_name):
        return f"Error: Tool {tool_name} not found"

    tool_func = getattr(chess_tools, actual_tool_name)

    # Call the tool via session pool (preserves board history)
    try:
        return session_pool.execute_tool_for_game(tool_func, game_board)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def execute_tools_parallel(tool_calls: List[Tuple], session_pool: SessionPool, max_workers: int = 8) -> Dict[int, str]:
    """Execute multiple tool calls in parallel using session pool's persistent threads.

    NOTE: max_workers parameter is ignored - pool size is determined at SessionPool creation.
    Keeping parameter for backward compatibility.

    Args:
        tool_calls: List of tuples (game_id, tool_name, game_board, function_mapping)
        session_pool: Pool of sessions for tool execution
        max_workers: Number of parallel workers (ignored, kept for compatibility)

    Returns:
        Dictionary mapping game_id to move result
    """
    if not tool_calls:
        return {}

    # Delegate to session pool's persistent executor
    return session_pool.execute_parallel(
        execute_tool_directly,
        tool_calls,
        progress_desc="   Executing non-agent moves"
    )


def check_and_mark_game_over(game: GameState, max_moves: Optional[int] = None) -> None:
    """Check if game is over and mark result/termination.

    Args:
        game: The game state to check
        max_moves: Optional maximum moves threshold
    """
    if game.board.is_game_over():
        game.is_complete = True
        if game.board.is_checkmate():
            game.result = "0-1" if game.board.turn == chess.WHITE else "1-0"
            game.termination = "checkmate"
        elif game.board.is_stalemate():
            game.result = "1/2-1/2"
            game.termination = "stalemate"
        elif game.board.is_insufficient_material():
            game.result = "1/2-1/2"
            game.termination = "insufficient_material"
        else:
            game.result = "1/2-1/2"
            game.termination = "draw"
    elif max_moves and game.move_count >= max_moves:
        game.is_complete = True
        game.result = "1/2-1/2"
        game.termination = "max_moves"
