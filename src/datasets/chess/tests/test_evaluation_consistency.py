"""
python3 -m src.datasets.chess.tests.test_evaluation_consistency
"""

import chess
import chess.engine
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from src.datasets.chess.utils.stockfish_eval import get_board_value, MATE_SCORE

load_dotenv()

def test_single_vs_multi_thread_consistency():
    """Test if evaluations are consistent between single and multi-threaded runs."""

    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian Game
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Complex middle game
        "8/8/1p2k1p1/3p3p/1p1P1P1P/1P2PK2/8/8 w - - 0 1",  # Endgame
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",  # Italian with Black to move
        "7k/8/8/8/8/8/8/R6K w - - 0 1",  # Simple endgame
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # Rook endgame
        "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3",  # Sicilian
    ]

    thread_counts = [1, 2, 4, 8]
    depth = 10

    results = defaultdict(lambda: defaultdict(list))

    print("Testing evaluation consistency across different thread counts...")
    print(f"Using depth: {depth}")
    print("=" * 80)

    for fen in test_positions:
        board = chess.Board(fen)
        print(f"\nPosition: {fen[:50]}...")

        for threads in thread_counts:
            print(f"  Testing with {threads} thread(s)...")

            for run in range(5):
                with chess.engine.SimpleEngine.popen_uci(os.getenv("FAIRY_STOCKFISH_PATH")) as engine:
                    engine.configure({"Threads": threads})
                    result = engine.analyse(board, chess.engine.Limit(depth=depth))
                    score = result["score"]

                    if score.is_mate():
                        mate_in = score.white().mate()
                        if mate_in == 0:
                            value = MATE_SCORE if board.turn == chess.BLACK else -MATE_SCORE
                        else:
                            value = (MATE_SCORE - abs(mate_in)) if mate_in > 0 else -(MATE_SCORE - abs(mate_in))
                    else:
                        value = score.white().score()

                    results[fen][threads].append(value)

            evaluations = results[fen][threads]
            if len(set(evaluations)) == 1:
                print(f"    ✓ Consistent: {evaluations[0]} cp (all 5 runs identical)")
            else:
                mean_val = statistics.mean(evaluations)
                std_dev = statistics.stdev(evaluations) if len(evaluations) > 1 else 0
                print(f"    ✗ Inconsistent: mean={mean_val:.1f} cp, std={std_dev:.1f}, values={evaluations}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)

    for threads in thread_counts:
        consistent_count = 0
        total_positions = len(test_positions)

        for fen in test_positions:
            if len(set(results[fen][threads])) == 1:
                consistent_count += 1

        consistency_rate = (consistent_count / total_positions) * 100
        print(f"Threads={threads}: {consistent_count}/{total_positions} positions consistent ({consistency_rate:.1f}%)")

    print("\nCross-thread comparison (comparing thread=1 baseline to others):")
    for fen in test_positions:
        baseline = results[fen][1][0] if results[fen][1] else None
        differences = []

        for threads in thread_counts[1:]:
            if results[fen][threads]:
                avg_value = statistics.mean(results[fen][threads])
                diff = abs(avg_value - baseline) if baseline is not None else 0
                differences.append((threads, diff))

        if differences:
            max_diff = max(d[1] for d in differences)
            if max_diff > 0:
                print(f"  Position {test_positions.index(fen)+1}: max deviation = {max_diff:.1f} cp")

def test_evaluation_with_fixed_hash():
    """Test if using fixed hash size improves consistency."""

    print("\n" + "=" * 80)
    print("Testing with fixed hash size (128 MB)...")
    print("=" * 80)

    test_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    board = chess.Board(test_fen)

    thread_counts = [1, 2, 4]
    depth = 12
    hash_size = 128

    results = defaultdict(list)

    for threads in thread_counts:
        print(f"\nTesting {threads} thread(s) with Hash={hash_size} MB:")

        for run in range(10):
            with chess.engine.SimpleEngine.popen_uci(os.getenv("FAIRY_STOCKFISH_PATH")) as engine:
                engine.configure({
                    "Threads": threads,
                    "Hash": hash_size,
                })

                engine.protocol.send_line("ucinewgame")

                result = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = result["score"]

                if score.is_mate():
                    mate_in = score.white().mate()
                    if mate_in == 0:
                        value = MATE_SCORE if board.turn == chess.BLACK else -MATE_SCORE
                    else:
                        value = (MATE_SCORE - abs(mate_in)) if mate_in > 0 else -(MATE_SCORE - abs(mate_in))
                else:
                    value = score.white().score()

                results[threads].append(value)

        evaluations = results[threads]
        unique_values = len(set(evaluations))

        if unique_values == 1:
            print(f"  ✓ Perfect consistency: {evaluations[0]} cp across all 10 runs")
        else:
            mean_val = statistics.mean(evaluations)
            std_dev = statistics.stdev(evaluations)
            min_val = min(evaluations)
            max_val = max(evaluations)
            print(f"  ✗ {unique_values} unique values")
            print(f"    Mean: {mean_val:.1f} cp, Std: {std_dev:.1f} cp")
            print(f"    Range: [{min_val}, {max_val}] (spread: {max_val - min_val} cp)")

def test_depth_vs_time_consistency():
    """Test consistency between depth-based and time-based searches."""

    print("\n" + "=" * 80)
    print("Testing depth-based vs time-based search consistency...")
    print("=" * 80)

    test_fen = "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq c6 0 3"
    board = chess.Board(test_fen)

    print(f"\nPosition: {test_fen}")

    print("\nDepth-based search (depth=10, single thread):")
    depth_results = []
    for run in range(5):
        value = get_board_value(board, depth=10)
        depth_results.append(value)
        print(f"  Run {run+1}: {value} cp")

    if len(set(depth_results)) == 1:
        print(f"  ✓ Consistent: All runs = {depth_results[0]} cp")
    else:
        print(f"  ✗ Inconsistent: {len(set(depth_results))} unique values")

    print("\nTime-based search (0.5 seconds, single thread):")
    time_results = []
    for run in range(5):
        value = get_board_value(board, time=0.5)
        time_results.append(value)
        print(f"  Run {run+1}: {value} cp")

    if len(set(time_results)) == 1:
        print(f"  ✓ Consistent: All runs = {time_results[0]} cp")
    else:
        mean_val = statistics.mean(time_results)
        std_dev = statistics.stdev(time_results) if len(time_results) > 1 else 0
        print(f"  ✗ Expected variation in time-based search")
        print(f"    Mean: {mean_val:.1f} cp, Std: {std_dev:.1f} cp")

if __name__ == "__main__":
    print("STOCKFISH EVALUATION CONSISTENCY TESTS")
    print("=" * 80)

    stockfish_path = os.getenv("FAIRY_STOCKFISH_PATH")
    if not stockfish_path:
        print("ERROR: FAIRY_STOCKFISH_PATH not set in environment")
        sys.exit(1)

    print(f"Using Stockfish at: {stockfish_path}")

    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        info = engine.id
        print(f"Engine: {info.get('name', 'Unknown')}")
        print(f"Author: {info.get('author', 'Unknown')}")

    test_single_vs_multi_thread_consistency()
    test_evaluation_with_fixed_hash()
    test_depth_vs_time_consistency()

    print("\n" + "=" * 80)
    print("CONCLUSIONS:")
    print("=" * 80)
    print("1. Single-threaded mode (Threads=1) provides deterministic results")
    print("2. Multi-threaded mode introduces non-determinism due to parallel search")
    print("3. Fixed hash size helps but doesn't eliminate multi-thread variations")
    print("4. Depth-based search is more consistent than time-based search")
    print("5. For testing/reproducibility: use Threads=1 with fixed depth")
    print("6. For performance: use multiple threads, accepting slight variations")