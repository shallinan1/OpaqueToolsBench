"""
Test consistency of chess tool outputs regardless of call order.
This tests for potential caching issues where prior engine calls affect subsequent results.

python3 -m src.datasets.chess.tests.test_consistency
"""

import chess
from src.datasets.chess.chess_game_session_v2 import (
    ChessGameSession,
    set_current_session,
    clear_current_session,
    cleanup_engines
)
from src.datasets.chess.chess_tools import (
    elo_1200,
    elo_1800,
    elo_2400,
)


def test_elo_consistency_simple():
    """Test that the same position returns the same move regardless of prior calls."""
    print("\n" + "="*60)
    print("TEST 1: Simple Consistency Check")
    print("="*60)

    # Use a mid-game position where different Elo levels might suggest different moves
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian Game", "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Complex middlegame", "r1bq1rk1/pp2npbp/2np2p1/2pN1pB1/2B1P2Q/3P1N2/PPP3PP/R4RK1 w - - 0 1"),
    ]

    all_passed = True

    for pos_name, fen in test_positions:
        print(f"\nTesting: {pos_name}")
        print(f"FEN: {fen}")

        # Setup session for this position
        session = ChessGameSession()
        session.set_position(fen)
        set_current_session(session)

        # Test multiple Elo strengths for consistency
        elos_to_test = [
            (elo_1200, "Elo 1200"),
            (elo_1800, "Elo 1800"),
            (elo_2400, "Elo 2400"),
        ]

        # Test each Elo level
        for test_func, test_name in elos_to_test:
            print(f"\n  {test_name} consistency:")

            # Call 1: Elo level alone
            move1 = test_func()
            print(f"    Call alone: {move1}")

            # Call 2: After a lower Elo move
            _ = elo_1200()
            move2 = test_func()
            print(f"    After Elo 1200: {move2}")

            # Call 3: After a higher Elo move
            _ = elo_2400()
            move3 = test_func()
            print(f"    After Elo 2400: {move3}")

            # Call 4: After sequence of different Elo levels
            _ = elo_1200()
            _ = elo_2400()
            _ = elo_1800()
            move4 = test_func()
            print(f"    After Elo 1200->2400->1800: {move4}")

            # Call 5: Repeated call
            move5 = test_func()
            print(f"    Repeated call: {move5}")

            if move1 == move2 == move3 == move4 == move5:
                print(f"  ✓ {test_name} CONSISTENT: Always returns {move1}")
            else:
                print(f"  ✗ {test_name} INCONSISTENT!")
                print(f"    Moves: {[move1, move2, move3, move4, move5]}")
                all_passed = False

        # Cleanup session
        clear_current_session(cleanup=True)

    return all_passed




def test_cross_contamination():
    """Test if analyzing different positions affects each other."""
    print("\n" + "="*60)
    print("TEST 2: Cross-Position Contamination Check")
    print("="*60)

    pos1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    pos2 = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

    all_passed = True

    # Get baseline moves
    print("\nBaseline moves:")
    session1 = ChessGameSession()
    session1.set_position(pos1)
    set_current_session(session1)
    baseline1 = elo_1800()
    print(f"  Position 1 baseline: {baseline1}")

    session2 = ChessGameSession()
    session2.set_position(pos2)
    set_current_session(session2)
    baseline2 = elo_1800()
    print(f"  Position 2 baseline: {baseline2}")

    # Test if analyzing pos2 affects pos1
    print("\nAfter analyzing other position:")
    set_current_session(session2)
    _ = elo_2400()  # Higher-strength analysis of position 2
    set_current_session(session1)
    test1 = elo_1800()  # Re-analyze position 1
    print(f"  Position 1 after analyzing pos2 at Elo 2400: {test1}")

    if baseline1 == test1:
        print(f"  ✓ No cross-contamination for position 1")
    else:
        print(f"  ✗ CONTAMINATION: Position 1 changed from {baseline1} to {test1}")
        all_passed = False

    # Test if analyzing pos1 affects pos2
    set_current_session(session1)
    _ = elo_2400()  # Higher-strength analysis of position 1
    set_current_session(session2)
    test2 = elo_1800()  # Re-analyze position 2
    print(f"  Position 2 after analyzing pos1 at Elo 2400: {test2}")

    if baseline2 == test2:
        print(f"  ✓ No cross-contamination for position 2")
    else:
        print(f"  ✗ CONTAMINATION: Position 2 changed from {baseline2} to {test2}")
        all_passed = False

    # Cleanup both sessions
    session1.cleanup()
    session2.cleanup()
    clear_current_session(cleanup=True)

    return all_passed


def test_repeated_calls():
    """Test that repeated calls to the same function give consistent results."""
    print("\n" + "="*60)
    print("TEST 3: Repeated Calls Consistency")
    print("="*60)

    test_fen = "r1bq1rk1/pp2npbp/2np2p1/2pN1pB1/2B1P2Q/3P1N2/PPP3PP/R4RK1 w - - 0 1"

    print(f"Testing position: Complex middlegame")
    print(f"FEN: {test_fen}")

    # Setup session
    session = ChessGameSession()
    session.set_position(test_fen)
    set_current_session(session)

    all_passed = True

    # Test Elo 1800 repeated calls
    print("\n  Testing 10 repeated calls to Elo 1800:")
    moves = []
    for i in range(10):
        move = elo_1800()
        moves.append(move)
        if i == 0:
            print(f"    First call: {move}")
        elif moves[-1] != moves[0]:
            print(f"    Call {i+1}: {move} (DIFFERENT!)")

    if len(set(moves)) == 1:
        print(f"  ✓ All 10 calls returned: {moves[0]}")
    else:
        print(f"  ✗ INCONSISTENT across repeated calls!")
        print(f"    Unique moves: {set(moves)}")
        all_passed = False

    # Test with interleaved different Elo levels
    print("\n  Testing Elo 1800 with interleaved other Elos:")
    moves = []
    for i in range(5):
        if i % 2 == 1:
            _ = elo_1200()  # Interleave with Elo 1200
        move = elo_1800()
        moves.append(move)

    if len(set(moves)) == 1:
        print(f"  ✓ Consistent with interleaving: {moves[0]}")
    else:
        print(f"  ✗ INCONSISTENT with interleaving!")
        print(f"    Moves: {moves}")
        all_passed = False

    # Cleanup
    clear_current_session(cleanup=True)

    return all_passed


def main():
    """Run all consistency tests."""
    print("="*60)
    print("CHESS TOOL CONSISTENCY TESTS")
    print("="*60)
    print("\nThese tests check if the chess tools give consistent results")
    print("regardless of the order in which they are called.")
    print("This helps detect caching/state issues in the chess engine.")

    try:
        results = []

        # Run all tests
        results.append(("Simple Consistency", test_elo_consistency_simple()))
        results.append(("Cross-contamination", test_cross_contamination()))
        results.append(("Repeated Calls", test_repeated_calls()))

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        all_passed = True
        for test_name, passed in results:
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {test_name}: {status}")
            if not passed:
                all_passed = False

        print("\n" + "="*60)
        if all_passed:
            print("✓ ALL TESTS PASSED - No consistency issues detected!")
            print("The ucinewgame command appears to be working correctly.")
        else:
            print("✗ CONSISTENCY ISSUES DETECTED!")
            print("The chess engine may be caching results between calls.")
            print("Consider strengthening the cache clearing mechanism.")
        print("="*60)

    finally:
        # Clean up
        cleanup_engines()
        print("\nEngines cleaned up.")


if __name__ == "__main__":
    main()
