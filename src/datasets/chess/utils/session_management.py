"""
Session and game state management for chess trajectory collection.

This module provides:
- SessionPool: Thread-local session management for parallel tool execution
- GameState: Tracking individual game state and move history
"""

import threading
import logging
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import chess
from tqdm import tqdm

from src.datasets.chess.chess_game_session_v2 import (
    ChessGameSession,
    set_current_session,
    clear_current_session,
)

logger = logging.getLogger(__name__)


class SessionPool:
    """Manages a pool of sessions for parallel tool execution.

    Uses thread-local storage to assign one session per worker thread.
    This ensures we have at most max_workers sessions/engines, regardless
    of the number of games.
    """

    def __init__(self, pool_size: int):
        self.pool_size = pool_size
        self._local = threading.local()
        self._all_sessions = []  # Track all created sessions for cleanup
        self._sessions_lock = threading.Lock()

        # Create persistent thread pool - threads will be reused across all moves
        self._executor = ThreadPoolExecutor(
            max_workers=pool_size,
            thread_name_prefix="chess_pool"
        )
        logger.info(f"SessionPool created with {pool_size} persistent worker threads")

    def _get_or_create_session(self) -> ChessGameSession:
        """Get session for current thread, creating if needed."""
        if not hasattr(self._local, 'session'):
            session_id = f"pool_session_{threading.get_ident()}"
            self._local.session = ChessGameSession(session_id=session_id)
            logger.debug(f"Created new session {session_id} for thread {threading.current_thread().name}")

            # Track this session for cleanup
            with self._sessions_lock:
                self._all_sessions.append(self._local.session)

        return self._local.session

    def execute_tool_for_game(self, tool_func, game_board: chess.Board) -> str:
        """Execute a tool function for a game's board state.

        Args:
            tool_func: Tool function to call (takes no parameters)
            game_board: The game's board (with full history)

        Returns:
            The tool's result (move in SAN notation)
        """
        # Get this thread's session
        session = self._get_or_create_session()

        # Copy the game's board into the session (preserves move_stack!)
        session.board = game_board.copy()

        # Set as current session for this thread
        set_current_session(session)

        try:
            # Call the tool (it will use get_current_session())
            result = tool_func()
            return result
        finally:
            # Clear current session (but don't destroy the session)
            clear_current_session()

    def execute_parallel(self, tool_func, tool_calls: List[Tuple],
                        progress_desc: str = "Executing tools") -> Dict[int, str]:
        """Execute tool calls in parallel using persistent thread pool.

        Args:
            tool_func: The function to execute (execute_tool_directly)
            tool_calls: List of tuples (game_id, tool_name, game_board, func_map)
            progress_desc: Description for progress bar

        Returns:
            Dictionary mapping game_id to result
        """
        results = {}

        if not tool_calls:
            return results

        # Submit all tasks to persistent executor
        future_to_game = {}
        for game_id, tool_name, game_board, func_map in tool_calls:
            future = self._executor.submit(
                tool_func,
                tool_name,
                game_board,
                self,  # Pass self (SessionPool)
                func_map
            )
            future_to_game[future] = (game_id, tool_name)

        # Collect results with progress bar
        with tqdm(total=len(future_to_game), desc=progress_desc, leave=False, unit="game") as pbar:
            for future in as_completed(future_to_game):
                game_id, tool_name = future_to_game[future]
                try:
                    results[game_id] = future.result()
                    logger.debug(f"Game {game_id}: Tool {tool_name} completed")
                except Exception as e:
                    results[game_id] = f"Error executing {tool_name}: {str(e)}"
                    logger.error(f"Game {game_id}: Tool {tool_name} failed: {e}")
                pbar.update(1)

        return results

    def cleanup(self):
        """Clean up thread pool and all sessions."""
        # Shutdown executor first to ensure no new work is submitted
        if hasattr(self, '_executor') and self._executor:
            logger.debug("Shutting down thread pool executor...")
            self._executor.shutdown(wait=True)
            logger.debug("Thread pool executor shut down")

        # Then clean up all sessions
        with self._sessions_lock:
            for session in self._all_sessions:
                try:
                    session.cleanup()
                    logger.debug(f"Cleaned up session {session.session_id}")
                except Exception as e:
                    logger.error(f"Error cleaning up session {session.session_id}: {e}")
            self._all_sessions.clear()

        # Clean up current thread's session reference if it exists
        if hasattr(self._local, 'session'):
            delattr(self._local, 'session')


class GameState:
    """Tracks state for a single chess game."""

    def __init__(self, game_id: int, start_fen: str, white_type: str = "agent", black_type: str = "agent"):
        self.game_id = game_id
        self.board = chess.Board(start_fen)
        self.start_fen = start_fen
        self.moves = []  # List of move dictionaries for trajectory
        self.tool_calls = []  # List of tool call records
        self.is_complete = False
        self.result = None  # "1-0", "0-1", "1/2-1/2"
        self.termination = None  # Reason for game end
        self.move_count = 0

        # Track player types for this specific game
        self.white_type = white_type
        self.black_type = black_type

        # For tracking conversation state
        self.full_responses = []  # Store complete API responses for analysis

        # Token tracking for this game
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
