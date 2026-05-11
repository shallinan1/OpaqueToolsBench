"""

"""

import chess

def get_game_phase_lichess(board: chess.Board) -> str:
    """
    Exact Lichess game phase detection.
    Source: Lichess Divider.scala
    
    Returns: 'opening', 'middlegame', or 'endgame'
    """
    majors_minors = majors_and_minors(board)
    
    # Middlegame starts when ANY of these conditions are met
    is_middlegame = (
        majors_minors <= 10 or 
        backrank_sparse(board) or 
        mixedness(board) > 150
    )
    
    if not is_middlegame:
        return 'opening'
    elif majors_minors <= 6:
        return 'endgame'
    else:
        return 'middlegame'


def majors_and_minors(board: chess.Board) -> int:
    """Count all pieces except kings and pawns."""
    count = 0
    for piece in board.piece_map().values():
        if piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            count += 1
    return count


def backrank_sparse(board: chess.Board) -> bool:
    """Sparse back-rank indicates pieces have been developed."""
    white_backrank = 0
    black_backrank = 0
    
    # First rank (a1-h1) for white
    for file in range(8):
        square = chess.square(file, 0)  # rank 0 = first rank
        piece = board.piece_at(square)
        if piece and piece.color == chess.WHITE:
            white_backrank += 1
    
    # Last rank (a8-h8) for black  
    for file in range(8):
        square = chess.square(file, 7)  # rank 7 = eighth rank
        piece = board.piece_at(square)
        if piece and piece.color == chess.BLACK:
            black_backrank += 1
    
    return white_backrank < 4 or black_backrank < 4


def mixedness(board: chess.Board) -> int:
    """
    Calculate mixedness score - how intermingled pieces are.
    Exact translation of Lichess implementation.
    """
    def score(y: int, white: int, black: int) -> int:
        # Direct translation of Scala pattern matching
        if white == 0 and black == 0:
            return 0
        elif white == 1 and black == 0:
            return 1 + (8 - y)
        elif white == 2 and black == 0:
            return (2 + (y - 2)) if y > 2 else 0
        elif white == 3 and black == 0:
            return (3 + (y - 1)) if y > 1 else 0
        elif white == 4 and black == 0:
            return (3 + (y - 1)) if y > 1 else 0  # group of 4 on homerow = 0
        elif white == 0 and black == 1:
            return 1 + y
        elif white == 1 and black == 1:
            return 5 + abs(4 - y)
        elif white == 2 and black == 1:
            return 4 + (y - 1)
        elif white == 3 and black == 1:
            return 5 + (y - 1)
        elif white == 0 and black == 2:
            return (2 + (6 - y)) if y < 6 else 0
        elif white == 1 and black == 2:
            return 4 + (7 - y)
        elif white == 2 and black == 2:
            return 7
        elif white == 0 and black == 3:
            return (3 + (7 - y)) if y < 7 else 0
        elif white == 1 and black == 3:
            return 5 + (7 - y)
        elif white == 0 and black == 4:
            return (3 + (7 - y)) if y < 7 else 0
        else:
            return 0
    
    total = 0
    
    # Check each 2x2 region on the board
    # Scala: for y <- 0 to 6, x <- 0 to 6
    for rank_start in range(7):  # 0-6 ranks
        for file_start in range(7):  # 0-6 files
            white_count = 0
            black_count = 0
            
            # Count pieces in 2x2 square
            for file_offset in range(2):
                for rank_offset in range(2):
                    square = chess.square(
                        file_start + file_offset,
                        rank_start + rank_offset
                    )
                    piece = board.piece_at(square)
                    if piece:
                        if piece.color == chess.WHITE:
                            white_count += 1
                        else:
                            black_count += 1
            
            # y in score function is 1-indexed (rank_start + 1)
            # Scala: yield (smallSquare << (x + 8 * y), y + 1)
            total += score(rank_start + 1, white_count, black_count)
    
    return total