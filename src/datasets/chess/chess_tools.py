"""
Stateful chess game tools that work with chess_game_session_v2.
These are parameter-free tools that agents can call to get move suggestions.
"""

from src.datasets.chess.chess_game_session_v2 import get_current_session
from src.datasets.chess.process_data import get_game_phase_fast


# ===== AGENT-CALLABLE TOOLS =====

def random_move() -> str:
    """Get a random legal move."""
    return get_current_session().get_random_move()

def worst_move() -> str:
    """Get the worst possible move."""
    return get_current_session().get_worst_move()

def opening_specialist() -> str:
    """Get a move that plays well in opening, poorly in other phases."""
    session = get_current_session()
    game_phase = get_game_phase_fast(session.board.fen())
    if game_phase == 'opening':
        return session.get_elo_move(2400)
    else:
        return session.get_random_move()

def middlegame_specialist() -> str:
    """Get a move that plays well in middlegame, poorly in other phases."""
    session = get_current_session()
    game_phase = get_game_phase_fast(session.board.fen())
    if game_phase == 'middlegame':
        return session.get_elo_move(2400)
    else:
        return session.get_random_move()

def endgame_specialist() -> str:
    """Get a move that plays well in endgame, poorly in other phases."""
    session = get_current_session()
    game_phase = get_game_phase_fast(session.board.fen())
    if game_phase == 'endgame':
        return session.get_elo_move(2400)
    else:
        return session.get_random_move()

def late_endgame_specialist() -> str:
    """Get a move that plays well in late endgame, poorly in other phases."""
    session = get_current_session()
    game_phase = get_game_phase_fast(session.board.fen())
    if game_phase == 'late_endgame':
        return session.get_elo_move(2400)
    else:
        return session.get_random_move()


# Generate all best_move_depth_N functions
def _create_best_move_func(depth: int):
    def best_move() -> str:
        return get_current_session().get_best_move(depth)
    best_move.__name__ = f"best_move_depth_{depth}"
    best_move.__doc__ = f"Get the best move at depth {depth}."
    return best_move

# Create best_move_depth_N functions for depths 1-24
for d in range(1, 25):
    globals()[f"best_move_depth_{d}"] = _create_best_move_func(d)


# Generate all elo_N functions
def _create_elo_func(elo: int):
    def elo_move() -> str:
        return get_current_session().get_elo_move(elo)
    elo_move.__name__ = f"elo_{elo}"
    elo_move.__doc__ = f"Get a move at Elo rating {elo}."
    return elo_move

# Create elo_N functions for Elo ratings from 500 to 2800 in increments of 100
# Valid range for Fairy-Stockfish UCI_Elo option: 500 to 2850
for elo in range(500, 2801, 100):
    globals()[f"elo_{elo}"] = _create_elo_func(elo)


# Export all tools
__all__ = [
    "random_move",
    "worst_move",
    "opening_specialist",
    "middlegame_specialist",
    "endgame_specialist",
    "late_endgame_specialist",
] + [f"best_move_depth_{i}" for i in range(1, 25)] + [f"elo_{elo}" for elo in range(500, 2801, 100)]