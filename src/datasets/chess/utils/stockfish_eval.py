import chess.engine
from dotenv import load_dotenv
import os
import multiprocessing
load_dotenv()
MATE_SCORE = 32000  # Score used for mate positions



def get_board_value(board, time=None, depth=None):
    """
    Evaluate a chess position using Stockfish.
    
    Args:
        board (chess.Board): The chess position to evaluate
        time (float, optional): Time limit in seconds
        depth (int, optional): Search depth limit
    
    Returns:
        float: 
            - Centipawn score from White's perspective (e.g., +34)
            - Large number for mate (e.g., 32000 - N for white mates in N, -(32000 - N) for black)
    """
    if time is not None and depth is not None:
        raise ValueError("Cannot specify both time and depth limits")
    if time is not None:
        limit = chess.engine.Limit(time=time)
    elif depth is not None:
        limit = chess.engine.Limit(depth=depth)
    else:
        limit = chess.engine.Limit(depth=16)  # Default to depth 16 if neither specified
    
    with chess.engine.SimpleEngine.popen_uci(os.getenv("FAIRY_STOCKFISH_PATH")) as engine:
        engine.configure({"Threads": 1})
        result = engine.analyse(board, limit)
        score = result["score"]

        if score.is_mate():
            mate_in = score.white().mate()
            # Handle immediate checkmate
            if mate_in == 0:
                # If it's Black's turn and they're in checkmate, White just won
                # If it's White's turn and they're in checkmate, Black just won
                return MATE_SCORE if board.turn == chess.BLACK else -MATE_SCORE
            # Handle mate in N
            return (MATE_SCORE - abs(mate_in)) if mate_in > 0 else -(MATE_SCORE - abs(mate_in))
        else:
            return score.white().score()