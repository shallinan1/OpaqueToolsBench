"""
Evaluate whether chess agents are calling the optimal tools.

This script analyzes trajectory files to determine if agents are selecting
optimal tools using the same logic as the "best" baseline player.
In mirror mode, the agent may play as either white or black.

Usage:
    python -m src.datasets.chess.evaluate_tool_selection \
        --input_file <trajectory.json> \
        --config_file <tool_config.json>

Example:
    python -m src.datasets.chess.evaluate_tool_selection \
        --input_file runs/chess/test/v0_trajectories.json \
        --config_file src/datasets/chess/shared_tools/elo_tools_accurate.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import chess

# Import the baseline player logic for determining best tools
from src.datasets.chess.baseline_player import select_best_tool, classify_tools
from src.datasets.chess.process_data import get_game_phase_fast

def load_trajectories(trajectory_path: str) -> List[Dict]:
    """Load trajectory file."""
    with open(trajectory_path, 'r') as f:
        data = json.load(f)

    # Handle both formats: raw list or dict with 'trajectories' key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'trajectories' in data:
        return data['trajectories']
    else:
        raise ValueError("Unexpected trajectory file format")


def evaluate_tool_selection(trajectories: List[Dict], config: Dict, verbose: bool = False) -> Dict:
    """
    Evaluate tool selection for agents using the same logic as the "best" baseline.
    Evaluates all agent moves regardless of which color the agent is playing.

    Args:
        trajectories: List of game trajectories
        config: Tool configuration
        verbose: Whether to print detailed per-move analysis

    Returns:
        Dictionary with evaluation results
    """
    tools = config.get('tools', [])
    function_mapping = config.get('function_mapping', {})

    # Classify once and reuse for every move evaluation.
    classification = classify_tools(tools, function_mapping)

    # Infer which baseline strategy family is being evaluated.
    if classification.specialists:
        tool_type = "specialist"
    elif classification.depth_tools:
        tool_type = "depth"
    elif classification.elo_tools:
        tool_type = "elo"
    else:
        tool_type = "unknown"

    total_calls = 0
    optimal_calls = 0
    tool_usage_counts = defaultdict(int)  # Aggregate tool usage across analyzed games.

    phase_stats = defaultdict(lambda: {'total': 0, 'correct': 0})  # Used for specialist phase accuracy.
    game_accuracies = []  # Per-game accuracies for mean game-level reporting.

    for trajectory in trajectories:
        # Check if agent is playing (could be white or black in mirror mode)
        white_is_agent = trajectory.get('white_type') == 'agent'
        black_is_agent = trajectory.get('black_type') == 'agent'

        if not (white_is_agent or black_is_agent):
            continue  # Skip if no agent in this game

        tool_calls = trajectory.get('tool_calls', [])
        moves = trajectory.get('moves', [])

        # Build FEN history for the game
        board = chess.Board(trajectory.get('start_fen', chess.STARTING_FEN))
        fen_history = [board.fen()]  # FEN before move 1

        for move in moves:
            if 'move' in move and move['move']:
                try:
                    board.push_san(move['move'])
                    fen_history.append(board.fen())
                except Exception:
                    # If move parsing fails, keep previous FEN
                    fen_history.append(fen_history[-1])

        # Reset board for evaluation
        board = chess.Board(trajectory.get('start_fen', chess.STARTING_FEN))

        game_total = 0
        game_optimal = 0

        for call in tool_calls:
            # Evaluate agent's calls (could be white or black depending on mirror mode)
            call_color = call.get('color')

            if call_color == 'white' and not white_is_agent:
                continue
            if call_color == 'black' and not black_is_agent:
                continue

            tool_name = call.get('tool')
            move_number = call.get('move_number', 0)

            if not tool_name:
                continue

            game_total += 1
            total_calls += 1
            tool_usage_counts[tool_name] += 1

            # Convert (move_number, color) to a ply index in fen_history.
            # move_number is 1-indexed; fen_history starts from the initial position.
            if call_color == 'white':
                fen_index = (move_number - 1) * 2  # White moves are at even indices after start
            else:  # black
                fen_index = (move_number - 1) * 2 + 1  # Black moves are at odd indices

            if 0 <= fen_index < len(fen_history):
                board = chess.Board(fen_history[fen_index])
            else:
                # Fallback to start position if index is out of range
                board = chess.Board(trajectory.get('start_fen', chess.STARTING_FEN))

            try:
                optimal_tool = select_best_tool(board, tools, function_mapping)

                if tool_name == optimal_tool:
                    optimal_calls += 1
                    game_optimal += 1

                if tool_type == "specialist":
                    phase = get_game_phase_fast(board.fen())
                    phase_stats[phase]['total'] += 1
                    if tool_name == optimal_tool:
                        phase_stats[phase]['correct'] += 1

                if verbose:
                    status = "✓" if tool_name == optimal_tool else "✗"
                    real_name = function_mapping.get(tool_name, tool_name)
                    optimal_real_name = function_mapping.get(optimal_tool, optimal_tool)
                    print(f"  Move {move_number}: {status} Used: {real_name}, Optimal: {optimal_real_name}")

            except ValueError as e:
                # Could not determine optimal tool
                if verbose:
                    print(f"  Move {move_number}: Could not determine optimal tool: {e}")

        if game_total > 0:
            game_accuracy = game_optimal / game_total
            game_accuracies.append(game_accuracy)
            if verbose:
                print(f"Game {trajectory.get('game_id', '?')}: {game_optimal}/{game_total} optimal ({game_accuracy:.2%})")

    # Return a machine-readable summary used by scripts and optional JSON output.
    results = {
        'player': 'agent',
        'tool_type': tool_type,
        'total_tool_calls': total_calls,
        'optimal_tool_calls': optimal_calls,
        'optimal_selection_rate': optimal_calls / total_calls if total_calls > 0 else 0,
        'tool_usage': dict(tool_usage_counts),
        'num_games_analyzed': len(game_accuracies),
        'per_game_accuracy_mean': sum(game_accuracies) / len(game_accuracies) if game_accuracies else 0,
        'function_mapping': function_mapping
    }

    if tool_type == "specialist":
        phase_accuracy = {}
        for phase, stats in phase_stats.items():
            if stats['total'] > 0:
                phase_accuracy[phase] = {
                    'total_calls': stats['total'],
                    'correct_calls': stats['correct'],
                    'accuracy': stats['correct'] / stats['total']
                }
        results['phase_accuracy'] = phase_accuracy
        results['specialists'] = {phase: tool for phase, tool in classification.specialists.items()}

    elif tool_type == "depth":
        results['depth_tools'] = {depth: tool for depth, tool in classification.depth_tools.items()}
        results['optimal_depth'] = max(classification.depth_tools.keys()) if classification.depth_tools else None

    elif tool_type == "elo":
        results['elo_tools'] = {elo: tool for elo, tool in classification.elo_tools.items()}
        results['optimal_elo'] = max(classification.elo_tools.keys()) if classification.elo_tools else None

    return results


def print_results(results: Dict):
    """Print evaluation results in a readable format."""
    print("\n" + "="*60)
    print("AGENT TOOL SELECTION EVALUATION")
    print("="*60)

    print(f"\nTool Type: {results['tool_type']}")
    print(f"Games Analyzed: {results['num_games_analyzed']}")
    print(f"Total Tool Calls: {results['total_tool_calls']}")
    print(f"Optimal Tool Calls: {results['optimal_tool_calls']}")
    print(f"Optimal Selection Rate: {results['optimal_selection_rate']:.2%}")
    print(f"Per-Game Accuracy (mean): {results['per_game_accuracy_mean']:.2%}")

    print("\nTool Usage Breakdown:")
    function_mapping = results.get('function_mapping', {})
    for tool, count in sorted(results['tool_usage'].items()):
        real_name = function_mapping.get(tool, tool)
        percentage = count / results['total_tool_calls'] * 100 if results['total_tool_calls'] > 0 else 0
        print(f"  {tool} ({real_name}): {count} calls ({percentage:.1f}%)")

    tool_type = results['tool_type']

    if tool_type == "specialist":
        print("\nPhase-Specific Accuracy:")
        phase_accuracy = results.get('phase_accuracy', {})
        specialists = results.get('specialists', {})

        for phase in ['opening', 'middlegame', 'endgame', 'late_endgame']:
            if phase in phase_accuracy:
                stats = phase_accuracy[phase]
                optimal_tool = specialists.get(phase, 'unknown')
                optimal_real_name = function_mapping.get(optimal_tool, optimal_tool)
                print(f"  {phase:12} → {optimal_tool} ({optimal_real_name:20}): "
                      f"{stats['accuracy']:.2%} ({stats['correct_calls']}/{stats['total_calls']})")

    elif tool_type == "depth":
        depth_tools = results.get('depth_tools', {})
        optimal_depth = results.get('optimal_depth')
        if optimal_depth and optimal_depth in depth_tools:
            optimal_tool = depth_tools[optimal_depth]
            optimal_real_name = function_mapping.get(optimal_tool, optimal_tool)
            print(f"\nOptimal Tool: {optimal_tool} ({optimal_real_name}) with depth {optimal_depth}")

    elif tool_type == "elo":
        elo_tools = results.get('elo_tools', {})
        optimal_elo = results.get('optimal_elo')
        if optimal_elo and optimal_elo in elo_tools:
            optimal_tool = elo_tools[optimal_elo]
            optimal_real_name = function_mapping.get(optimal_tool, optimal_tool)
            print(f"\nOptimal Tool: {optimal_tool} ({optimal_real_name}) with Elo {optimal_elo}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate whether chess agents are calling optimal tools")
    parser.add_argument("--input_file", required=True, help="Path to trajectory JSON file")
    parser.add_argument("--config_file", required=True, help="Path to tool configuration JSON file")
    parser.add_argument("--output_file", help="Optional: Save results to JSON file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-move analysis")

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config_file}")
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    print(f"Loading trajectories from: {args.input_file}")
    trajectories = load_trajectories(args.input_file)
    print(f"Loaded {len(trajectories)} trajectories")

    results = evaluate_tool_selection(trajectories, config, verbose=args.verbose)

    print_results(results)

    if args.output_file:
        output_path = Path(args.output_file)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()