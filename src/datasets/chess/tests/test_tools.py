"""
Test ELO-rated chess tools and record win rates.

This runs a round-robin tournament between:
- Elo 500, 1000, 1500, 2000, 2500
- Random move
- Worst move

Results are displayed as a lower triangular win rate table.

Usage:
    # Run full tournament with 2 games per matchup
    python3 -m src.datasets.chess.tests.test_tools --samples 2

    # Run simple test (Elo 500 vs Elo 2500)
    python3 -m src.datasets.chess.tests.test_tools --simple
"""
import chess
import chess.pgn
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import argparse
from tqdm import tqdm
from io import StringIO

from src.datasets.chess.chess_game_session_v2 import (
    ChessGameSession,
    set_current_session,
    clear_current_session,
    cleanup_engines
)
from src.datasets.chess.chess_tools import (
    elo_500,
    elo_600,
    elo_700,
    elo_800,
    elo_900,
    elo_1000,
    elo_1100,
    elo_1200,
    elo_1300,
    elo_1400,
    elo_1500,
    elo_1600,
    elo_1700,
    elo_1800,
    elo_1900,
    elo_2000,
    elo_2100,
    elo_2200,
    elo_2300,
    elo_2400,
    elo_2500,
    elo_2600,
    elo_2700,
    elo_2800,
    random_move,
    worst_move,
)

def play_tool_game(white_tool, black_tool, white_name, black_name, max_moves=200, verbose=True):
    """Play a game between two tools and return the result."""
    # Create session for this game
    session = ChessGameSession()
    set_current_session(session)

    move_count = 0

    if verbose:
        print(f"\n{white_name} (White) vs {black_name} (Black)")
        print("-" * 40)

    try:
        while not session.is_game_over() and move_count < max_moves:
            # Get current player
            current_tool = white_tool if session.board.turn == chess.WHITE else black_tool
            current_name = white_name if session.board.turn == chess.WHITE else black_name

            # Get move (tool uses current session automatically)
            move_san = current_tool()

            try:
                # Apply move to session
                if session.make_move(move_san):
                    move_count += 1

                    if verbose and move_count % 50 == 0:
                        print(f"Move {move_count}...")
                else:
                    if verbose:
                        print(f"Error: {current_name} made illegal move {move_san}")
                    return f"{white_name if current_name == black_name else black_name} wins by illegal move", move_count

            except Exception as e:
                if verbose:
                    print(f"Error: {current_name} made illegal move {move_san}: {e}")
                return f"{white_name if current_name == black_name else black_name} wins by illegal move", move_count

        # Determine result
        if session.is_game_over():
            result = session.get_result()
        else:
            result = f"Draw by max moves ({max_moves})"

        if verbose:
            print(f"Result: {result}")
            print(f"Final position: {session.get_fen()}")
            print(f"Total moves: {move_count}")

        return result, move_count

    finally:
        # Cleanup session
        clear_current_session(cleanup=True)

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test chess ELO tools")
    parser.add_argument('--simple', action='store_true', help='Run simple test mode (Elo 2500 vs Elo 500, Random, Worst)')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--samples', type=int, default=2, help='Number of samples per matchup (default: 2)')
    args = parser.parse_args()

    # ===== SIMPLE TEST MODE =====
    if args.simple:
        print("=" * 60)
        print("SIMPLE TEST MODE: Elo 2500 vs [Elo 500, Random, Worst]")
        print("=" * 60)

        # Game 1: Elo 2500 (White) vs Elo 500 (Black)
        print("\n" + "=" * 60)
        print("GAME 1: Elo 2500 (White) vs Elo 500 (Black)")
        print("=" * 60)

        # Create session and PGN game
        session = ChessGameSession()
        set_current_session(session)

        game = chess.pgn.Game()
        game.headers["White"] = "Elo 2500"
        game.headers["Black"] = "Elo 500"
        node = game

        move_count = 0

        try:
            while not session.is_game_over() and move_count < 200:
                # Get move
                if session.board.turn == chess.WHITE:
                    move_san = elo_2500()
                else:
                    move_san = elo_500()

                # Apply move
                move = session.board.parse_san(move_san)
                if session.make_move(move_san):
                    move_count += 1
                    node = node.add_variation(move)
                else:
                    print(f"ERROR: Illegal move {move_san}")
                    break

            # Game over - set result
            if session.is_game_over():
                result = session.get_result()
                print(f"Result: {result}")
                if "White wins" in result:
                    game.headers["Result"] = "1-0"
                elif "Black wins" in result:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
            else:
                print("Result: Draw by max moves")
                game.headers["Result"] = "1/2-1/2"

            print(f"Total moves: {move_count}")

            print(f"\nPGN:")
            print(game)

        finally:
            clear_current_session(cleanup=True)

        # Game 2: Elo 500 (White) vs Elo 2500 (Black)
        print("\n" + "=" * 60)
        print("GAME 2: Elo 500 (White) vs Elo 2500 (Black)")
        print("=" * 60)

        # Create new session and PGN game
        session = ChessGameSession()
        set_current_session(session)

        game = chess.pgn.Game()
        game.headers["White"] = "Elo 500"
        game.headers["Black"] = "Elo 2500"
        node = game

        move_count = 0

        try:
            while not session.is_game_over() and move_count < 200:
                # Get move
                if session.board.turn == chess.WHITE:
                    move_san = elo_500()
                else:
                    move_san = elo_2500()

                # Apply move
                move = session.board.parse_san(move_san)
                if session.make_move(move_san):
                    move_count += 1
                    node = node.add_variation(move)
                else:
                    print(f"ERROR: Illegal move {move_san}")
                    break

            # Game over - set result
            if session.is_game_over():
                result = session.get_result()
                print(f"Result: {result}")
                if "White wins" in result:
                    game.headers["Result"] = "1-0"
                elif "Black wins" in result:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
            else:
                print("Result: Draw by max moves")
                game.headers["Result"] = "1/2-1/2"

            print(f"Total moves: {move_count}")

            print(f"\nPGN:")
            print(game)

        finally:
            clear_current_session(cleanup=True)

        # Game 3: Elo 2500 (White) vs Random (Black)
        print("\n" + "=" * 60)
        print("GAME 3: Elo 2500 (White) vs Random (Black)")
        print("=" * 60)

        session = ChessGameSession()
        set_current_session(session)

        game = chess.pgn.Game()
        game.headers["White"] = "Elo 2500"
        game.headers["Black"] = "Random"
        node = game

        move_count = 0

        try:
            while not session.is_game_over() and move_count < 200:
                if session.board.turn == chess.WHITE:
                    move_san = elo_2500()
                else:
                    move_san = random_move()

                move = session.board.parse_san(move_san)
                if session.make_move(move_san):
                    move_count += 1
                    node = node.add_variation(move)
                else:
                    print(f"ERROR: Illegal move {move_san}")
                    break

            if session.is_game_over():
                result = session.get_result()
                print(f"Result: {result}")
                if "White wins" in result:
                    game.headers["Result"] = "1-0"
                elif "Black wins" in result:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
            else:
                print("Result: Draw by max moves")
                game.headers["Result"] = "1/2-1/2"

            print(f"Total moves: {move_count}")
            print(f"\nPGN:")
            print(game)

        finally:
            clear_current_session(cleanup=True)

        # Game 4: Elo 2500 (White) vs Worst (Black)
        print("\n" + "=" * 60)
        print("GAME 4: Elo 2500 (White) vs Worst (Black)")
        print("=" * 60)

        session = ChessGameSession()
        set_current_session(session)

        game = chess.pgn.Game()
        game.headers["White"] = "Elo 2500"
        game.headers["Black"] = "Worst"
        node = game

        move_count = 0

        try:
            while not session.is_game_over() and move_count < 200:
                if session.board.turn == chess.WHITE:
                    move_san = elo_2500()
                else:
                    move_san = worst_move()

                move = session.board.parse_san(move_san)
                if session.make_move(move_san):
                    move_count += 1
                    node = node.add_variation(move)
                else:
                    print(f"ERROR: Illegal move {move_san}")
                    break

            if session.is_game_over():
                result = session.get_result()
                print(f"Result: {result}")
                if "White wins" in result:
                    game.headers["Result"] = "1-0"
                elif "Black wins" in result:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
            else:
                print("Result: Draw by max moves")
                game.headers["Result"] = "1/2-1/2"

            print(f"Total moves: {move_count}")
            print(f"\nPGN:")
            print(game)

        finally:
            clear_current_session(cleanup=True)

        # Game 5: Random (White) vs Elo 2500 (Black)
        print("\n" + "=" * 60)
        print("GAME 5: Random (White) vs Elo 2500 (Black)")
        print("=" * 60)

        session = ChessGameSession()
        set_current_session(session)

        game = chess.pgn.Game()
        game.headers["White"] = "Random"
        game.headers["Black"] = "Elo 2500"
        node = game

        move_count = 0

        try:
            while not session.is_game_over() and move_count < 200:
                if session.board.turn == chess.WHITE:
                    move_san = random_move()
                else:
                    move_san = elo_2500()

                move = session.board.parse_san(move_san)
                if session.make_move(move_san):
                    move_count += 1
                    node = node.add_variation(move)
                else:
                    print(f"ERROR: Illegal move {move_san}")
                    break

            if session.is_game_over():
                result = session.get_result()
                print(f"Result: {result}")
                if "White wins" in result:
                    game.headers["Result"] = "1-0"
                elif "Black wins" in result:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
            else:
                print("Result: Draw by max moves")
                game.headers["Result"] = "1/2-1/2"

            print(f"Total moves: {move_count}")
            print(f"\nPGN:")
            print(game)

        finally:
            clear_current_session(cleanup=True)

        # Game 6: Worst (White) vs Elo 2500 (Black)
        print("\n" + "=" * 60)
        print("GAME 6: Worst (White) vs Elo 2500 (Black)")
        print("=" * 60)

        session = ChessGameSession()
        set_current_session(session)

        game = chess.pgn.Game()
        game.headers["White"] = "Worst"
        game.headers["Black"] = "Elo 2500"
        node = game

        move_count = 0

        try:
            while not session.is_game_over() and move_count < 200:
                if session.board.turn == chess.WHITE:
                    move_san = worst_move()
                else:
                    move_san = elo_2500()

                move = session.board.parse_san(move_san)
                if session.make_move(move_san):
                    move_count += 1
                    node = node.add_variation(move)
                else:
                    print(f"ERROR: Illegal move {move_san}")
                    break

            if session.is_game_over():
                result = session.get_result()
                print(f"Result: {result}")
                if "White wins" in result:
                    game.headers["Result"] = "1-0"
                elif "Black wins" in result:
                    game.headers["Result"] = "0-1"
                else:
                    game.headers["Result"] = "1/2-1/2"
            else:
                print("Result: Draw by max moves")
                game.headers["Result"] = "1/2-1/2"

            print(f"Total moves: {move_count}")
            print(f"\nPGN:")
            print(game)

        finally:
            clear_current_session(cleanup=True)

        cleanup_engines()
        exit(0)
    # ===== END SIMPLE TEST MODE =====

    # Determine number of workers based on CPU count or command-line arg
    cpu_count = os.cpu_count() or 4
    max_workers = args.workers if args.workers else max(1, cpu_count - 1)


    print("Running chess tools tests...")
    print("=" * 60)
    print(f"System has {cpu_count} CPUs, using {max_workers} parallel workers")

    # Test: Round-robin tournament between selected tools
    print("\nRound-robin tournament between selected tools")
    print("=" * 60)

    # Select specific tools for the tournament
    tools = [
        ("Elo 5", elo_500),
        ("Elo 10", elo_1000),
        ("Elo 15", elo_1500),
        ("Elo 20", elo_2000),
        ("Elo 25", elo_2500),
        ("Random", random_move),
        ("Worst", worst_move),
    ]

    results_summary = []
    sample_size = args.samples  # Number of times to run each matchup

    # Prepare all game pairings (every tool vs every other tool)
    games_to_play = []
    for i, (name1, tool1) in enumerate(tools):
        for j, (name2, tool2) in enumerate(tools):
            if i != j:  # Don't play against self
                # Run each matchup sample_size times
                for sample_num in range(sample_size):
                    # Tool1 as white vs Tool2 as black
                    games_to_play.append((tool1, tool2, f"{name1} (S{sample_num+1})", f"{name2} (S{sample_num+1})"))

    print(f"Running {len(games_to_play)} games ({sample_size} samples per matchup) in parallel with up to {max_workers} workers...")
    print(f"Total matchups: {len(tools)} x {len(tools) - 1} = {len(tools) * (len(tools) - 1)}")
    start_time = time.time()

    # Run games in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all games
        future_to_game = {
            executor.submit(play_tool_game, white_tool, black_tool, white_name, black_name, 200, False):  # verbose=False
            (white_name, black_name)
            for white_tool, black_tool, white_name, black_name in games_to_play
        }

        # Collect results as they complete with progress bar
        with tqdm(total=len(games_to_play), desc="Round-robin games", unit="game") as pbar:
            for future in as_completed(future_to_game):
                white_name, black_name = future_to_game[future]
                try:
                    result, moves = future.result()
                    results_summary.append({
                        'white': white_name,
                        'black': black_name,
                        'result': result,
                        'moves': moves
                    })
                except Exception as exc:
                    tqdm.write(f"  ERROR: Game {white_name} vs {black_name} - {exc}")
                    results_summary.append({
                        'white': white_name,
                        'black': black_name,
                        'result': "Error",
                        'moves': 0
                    })
                pbar.update(1)

    elapsed = time.time() - start_time
    print(f"All games completed in {elapsed:.1f} seconds")

    # Calculate game length statistics
    game_lengths = [g['moves'] for g in results_summary if g['result'] != "Error"]
    avg_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    min_length = min(game_lengths) if game_lengths else 0
    max_length = max(game_lengths) if game_lengths else 0
    print(f"Game length stats: avg={avg_length:.1f} moves, min={min_length}, max={max_length}")

    # Build win rate matrix
    print("\n" + "=" * 60)
    print("WIN RATE TABLE (Row player's win rate vs Column player)")
    print("=" * 60)

    # Calculate win rates for each matchup
    win_rates = {}
    for i, (name1, _) in enumerate(tools):
        win_rates[name1] = {}
        for j, (name2, _) in enumerate(tools):
            if i == j:
                win_rates[name1][name2] = None  # Can't play against self
                continue

            # Find all games where name1 played against name2
            # Extract base names (without sample numbers)
            games_name1_white = [g for g in results_summary if g['white'].startswith(name1 + " ") and g['black'].startswith(name2 + " ")]
            games_name1_black = [g for g in results_summary if g['white'].startswith(name2 + " ") and g['black'].startswith(name1 + " ")]

            # Count wins for name1
            wins_as_white = sum(1 for g in games_name1_white if "White wins" in g['result'])
            wins_as_black = sum(1 for g in games_name1_black if "Black wins" in g['result'])

            # Count draws
            draws_as_white = sum(1 for g in games_name1_white if "Draw" in g['result'])
            draws_as_black = sum(1 for g in games_name1_black if "Draw" in g['result'])

            total_games = len(games_name1_white) + len(games_name1_black)
            total_wins = wins_as_white + wins_as_black
            total_draws = draws_as_white + draws_as_black

            # Calculate win rate (wins + 0.5*draws) / total
            if total_games > 0:
                win_rate = (total_wins + 0.5 * total_draws) / total_games * 100
                win_rates[name1][name2] = win_rate
            else:
                win_rates[name1][name2] = 0.0

    # Print lower triangular table
    tool_names = [name for name, _ in tools]

    # Print header
    print("\n" + " " * 12, end="")
    for name in tool_names:
        print(f"{name:>10}", end="")
    print()
    print("-" * (12 + 10 * len(tool_names)))

    # Print rows (lower triangular)
    for i, row_name in enumerate(tool_names):
        print(f"{row_name:>12}", end="")
        for j, col_name in enumerate(tool_names):
            if i > j:  # Lower triangular
                rate = win_rates[row_name][col_name]
                print(f"{rate:>9.1f}%", end="")
            else:
                print(f"{'':>10}", end="")
        print()

    print("\nNote: Win rate = (Wins + 0.5 * Draws) / Total Games * 100%")
    print("      Each cell shows row player's win rate against column player")

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("DETAILED MATCHUP STATISTICS")
    print("=" * 60)

    for i, (name1, _) in enumerate(tools):
        for j, (name2, _) in enumerate(tools):
            if i <= j:  # Only show lower triangular in detail
                continue

            # Find all games where name1 played against name2
            games_name1_white = [g for g in results_summary if g['white'].startswith(name1 + " ") and g['black'].startswith(name2 + " ")]
            games_name1_black = [g for g in results_summary if g['white'].startswith(name2 + " ") and g['black'].startswith(name1 + " ")]

            # Count results
            wins_as_white = sum(1 for g in games_name1_white if "White wins" in g['result'])
            draws_as_white = sum(1 for g in games_name1_white if "Draw" in g['result'])
            losses_as_white = sum(1 for g in games_name1_white if "Black wins" in g['result'])

            wins_as_black = sum(1 for g in games_name1_black if "Black wins" in g['result'])
            draws_as_black = sum(1 for g in games_name1_black if "Draw" in g['result'])
            losses_as_black = sum(1 for g in games_name1_black if "White wins" in g['result'])

            total_wins = wins_as_white + wins_as_black
            total_draws = draws_as_white + draws_as_black
            total_losses = losses_as_white + losses_as_black
            total_games = total_wins + total_draws + total_losses

            win_rate = win_rates[name1][name2]

            # Calculate average game length
            all_games = games_name1_white + games_name1_black
            avg_moves = sum(g['moves'] for g in all_games) / len(all_games) if all_games else 0

            print(f"\n{name1} vs {name2}:")
            print(f"  Win rate: {win_rate:.1f}%")
            print(f"  Record: {total_wins}W-{total_draws}D-{total_losses}L (out of {total_games} games)")
            print(f"  As white: {wins_as_white}W-{draws_as_white}D-{losses_as_white}L")
            print(f"  As black: {wins_as_black}W-{draws_as_black}D-{losses_as_black}L")
            print(f"  Avg game length: {avg_moves:.1f} moves")

    # Cleanup
    cleanup_engines()
    print("\n" + "=" * 60)
    print("Tests complete!")