"""
Stateful chess game session management.
Test harness manages game state, tools are defined in chess_tools.py.
"""

import chess
import chess.engine
import os
import random
from typing import Optional
from uuid import uuid4
import threading
from dotenv import load_dotenv

load_dotenv()


class ChessGameSession:
    """Manages a chess game with full state tracking."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid4())
        self.board = chess.Board()
        self._engine = None  # Lazy load engine per session
    
    def _get_engine(self):
        """Get or create engine for this session."""
        if self._engine is None:
            stockfish_path = os.getenv("FAIRY_STOCKFISH_PATH")
            if not stockfish_path:
                raise ValueError("FAIRY_STOCKFISH_PATH environment variable not set")
            self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            self._engine.configure({"Threads": 1})
        return self._engine
    
    def cleanup(self):
        """Clean up engine resources."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def set_position(self, fen: str):
        """Set position from FEN."""
        self.board = chess.Board(fen)

    def make_move(self, move_san: str) -> bool:
        """Apply a move to the game."""
        try:
            move = self.board.parse_san(move_san)
            self.board.push(move)
            return True
        except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            return False

    def get_best_move(self, depth: int) -> str:
        """Get best move at specified depth using full game history."""
        engine = self._get_engine()
        # Clear ALL engine state to ensure deterministic results independent of call order
        engine.protocol.send_line("ucinewgame")
        result = engine.play(self.board, chess.engine.Limit(depth=depth))
        return self.board.san(result.move)

    def get_elo_move(self, elo: int, time: float = 0.5) -> str:
        """Get a move limited to specific Elo rating using UCI_LimitStrength.

        Note: UCI_Elo is calibrated at 120s+1s time control (CCRL 40/4).
        Time limit affects actual strength - more time = stronger play than target Elo.
        Default of 0.5s per move balances speed and reasonable Elo approximation.
        """
        engine = self._get_engine()
        # Clear ALL engine state to ensure deterministic results independent of call order
        engine.protocol.send_line("ucinewgame")
        
        # Configure engine to play at specific Elo rating
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        result = engine.play(self.board, chess.engine.Limit(time=time))
        
        # Reset to unlimited strength for other uses
        engine.configure({"UCI_LimitStrength": False})
        return self.board.san(result.move)
    
    def get_worst_move(self, depth: int = 12) -> str:
        """Get worst move."""
        engine = self._get_engine()
        # Clear ALL engine state to ensure deterministic results independent of call order
        engine.protocol.send_line("ucinewgame")
        worst_move = None
        worst_score = float('inf') if self.board.turn == chess.WHITE else float('-inf')

        # Collect all moves and scores
        move_scores = []
        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)

            info = engine.analyse(test_board, chess.engine.Limit(depth=depth))

            # Use python-chess's built-in conversion with a standard mate score
            # 32000 is a common convention (close to the maximum value for a signed 16-bit integer)
            # .white() returns the Score object (Cp or Mate) from white's perspective
            score = info["score"].white().score(mate_score=32000)

            move_scores.append((move, score, self.board.san(move)))

            if score is not None:
                if self.board.turn == chess.WHITE and score < worst_score:
                    worst_score = score
                    worst_move = move
                elif self.board.turn == chess.BLACK and score > worst_score:
                    worst_score = score
                    worst_move = move


        if worst_move:
            return self.board.san(worst_move)
        # Fallback to random
        move = random.choice(list(self.board.legal_moves))
        return self.board.san(move)
    
    def get_random_move(self) -> str:
        """Get a random legal move."""
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return "Error: No legal moves available"
        move = random.choice(legal_moves)
        return self.board.san(move)

    def get_fen(self) -> str:
        return self.board.fen()

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def get_result(self) -> Optional[str]:
        if not self.is_game_over():
            return None
        outcome = self.board.outcome()
        if outcome.winner is None:
            return f"Draw by {outcome.termination.name}"
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        return f"{winner} wins by {outcome.termination.name}"

    def reset(self):
        """Reset to starting position."""
        self.board = chess.Board()


# Thread-local storage
_thread_local = threading.local()

def set_current_session(session: ChessGameSession):
    """Set the session for this thread."""
    _thread_local.session = session

def get_current_session() -> ChessGameSession:
    """Get thread-local session."""
    if not hasattr(_thread_local, 'session') or _thread_local.session is None:
        raise RuntimeError("No session set for this thread. Test harness must call set_current_session() first.")
    return _thread_local.session

def clear_current_session(cleanup: bool = False):
    """Clear the current session reference for this thread.

    Args:
        cleanup: If True, also cleanup the session's engine before clearing.
                 If False (default), just clear the reference without destroying the session.

    Usage:
        - SessionPool calls this with cleanup=False to clear reference while keeping engine alive
        - Direct session users should call with cleanup=True to properly destroy the engine
    """
    if hasattr(_thread_local, 'session'):
        session = _thread_local.session
        if cleanup and session:
            session.cleanup()  # Clean up engine
        _thread_local.session = None

def cleanup_engines():
    """
    Compatibility function - no-op in v2.
    Engines are now owned by sessions and cleaned up via clear_current_session().
    """
    pass


__all__ = [
    "ChessGameSession",
    "set_current_session",
    "get_current_session",
    "clear_current_session",
    "cleanup_engines",
]