"""
Add Stockfish position evaluations to chess trajectories.

This script annotates existing game trajectories with centipawn evaluations.
It does NOT run new games or test new descriptions—it just adds position
quality scores to trajectories that were already played.

Note: For measuring optimal tool selection rate (% of best tool calls),
use evaluate_tool_selection.py instead.

Metrics added:
    Per move:
        - board_value_cp: Centipawn evaluation (positive = White advantage)
        - turn_to_move: Whose turn after this move

    Per trajectory:
        - initial_board_value_cp: Starting position evaluation
        - total_value_change_cp: Final - Initial evaluation (raw change)
        - agent_normalized_change_cp: Value change from agent's perspective
            * Positive = agent improved position
            * For White agent: same as total_value_change_cp
            * For Black agent: negated (since lower values favor Black)

    Summary statistics:
        - average_value_change_cp: Mean raw change across all games
        - average_agent_normalized_change_cp: Mean agent-perspective change

Input: v0_trajectories.json (from run.py)
Output: v0_scored.json (trajectories with evaluations added)

Example:
    python -m src.datasets.chess.evaluate \\
        --input runs/chess/.../v0_trajectories.json \\
        --num_workers 10 \\
        --depth 16
"""

import json
import sys
import chess
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from src.datasets.chess.utils.stockfish_eval import get_board_value

def score_single_trajectory(trajectory, depth=16):
    """
    Score a single trajectory. This function will be called in parallel.

    Args:
        trajectory: A single trajectory dict
        depth: Depth for board evaluation

    Returns:
        The trajectory dict with board values added
    """
    scored_trajectory = {}

    # Copy keys in order, inserting initial_board_value_cp after start_fen
    for key, value in trajectory.items():
        scored_trajectory[key] = value

        # If we just added start_fen, calculate and add the board value right after it
        if key == 'start_fen':
            board = chess.Board(value)
            scored_trajectory['initial_board_value_cp'] = get_board_value(board)
            # Placeholder for change values - will be calculated after scoring moves
            scored_trajectory['total_value_change_cp'] = None
            scored_trajectory['agent_normalized_change_cp'] = None

    # Score each move in the trajectory
    for step in scored_trajectory.get('moves', []):
        if 'fen_after' in step:
            board = chess.Board(step['fen_after'])
            # Use specified depth for consistency
            board_value = get_board_value(board, depth=depth)
            step['board_value_cp'] = board_value

            # Also store whose turn it is for context (after the move was made)
            step['turn_to_move'] = 'white' if board.turn == chess.WHITE else 'black'

    # Calculate value_change: difference between first and last board values
    initial_cp = scored_trajectory.get('initial_board_value_cp')
    moves = scored_trajectory.get('moves', [])

    if initial_cp is not None and moves:
        # Find the final board value (last move with a board_value_cp)
        final_cp = None
        for move in reversed(moves):
            if 'board_value_cp' in move and move['board_value_cp'] is not None:
                final_cp = move['board_value_cp']
                break

        if final_cp is not None:
            value_change = final_cp - initial_cp
            scored_trajectory['value_change'] = value_change
            scored_trajectory['total_value_change_cp'] = value_change

            # Calculate agent_normalized_value based on agent color
            # Board values are from White's perspective (positive = good for white)
            # Agent normalized value should be positive when the change benefits the agent
            white_type = scored_trajectory.get('white_type')
            black_type = scored_trajectory.get('black_type')

            if white_type == 'agent':
                # Agent is playing as white, positive value_change is good for agent
                agent_normalized = value_change
                scored_trajectory['agent_normalized_value'] = agent_normalized
                scored_trajectory['agent_normalized_change_cp'] = agent_normalized
            elif black_type == 'agent':
                # Agent is playing as black, negative value_change is good for agent
                agent_normalized = -value_change
                scored_trajectory['agent_normalized_value'] = agent_normalized
                scored_trajectory['agent_normalized_change_cp'] = agent_normalized
            else:
                # If neither white nor black is agent, don't set agent_normalized_value
                pass

    return scored_trajectory

def score_trajectory_file(input_path, num_workers=None, depth=16):
    """
    Load a trajectory file, calculate board values for each position, and save with scores.

    Args:
        input_path: Path to input trajectory JSON file
        num_workers: Number of parallel workers (defaults to CPU count)
        depth: Depth for board evaluation (default 16)
    """
    input_path = Path(input_path)
    # Derive scored filename from trajectories filename
    if input_path.name == "trajectories.json":
        # Improvement iteration: trajectories.json -> scored.json
        output_path = input_path.parent / "scored.json"
    else:
        # Base run: v0_trajectories.json -> v0_scored.json
        output_path = input_path.parent / input_path.name.replace('_trajectories.json', '_scored.json')

    print(f"Loading trajectories from: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    print(f"Found {len(data)} trajectories to score")

    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(data))  # Don't use more workers than trajectories

    print(f"Using {num_workers} parallel workers")

    # Process trajectories in parallel
    score_func = partial(score_single_trajectory, depth=depth)

    with Pool(num_workers) as pool:
        scored_data = list(tqdm(
            pool.imap(score_func, data),
            total=len(data),
            desc="Scoring trajectories"
        ))

    # Calculate statistics
    total_positions = sum(len(t.get('moves', [])) for t in scored_data)
    scored_positions = sum(
        sum(1 for step in t.get('moves', [])
            if 'board_value_cp' in step and step['board_value_cp'] is not None)
        for t in scored_data
    )

    # Calculate board value summary statistics
    value_changes = []
    agent_normalized_values = []

    for trajectory in scored_data:
        if 'value_change' in trajectory and trajectory['value_change'] is not None:
            value_changes.append(trajectory['value_change'])
        if 'agent_normalized_value' in trajectory and trajectory['agent_normalized_value'] is not None:
            agent_normalized_values.append(trajectory['agent_normalized_value'])

    avg_value_change = sum(value_changes) / len(value_changes) if value_changes else 0.0
    avg_agent_normalized = sum(agent_normalized_values) / len(agent_normalized_values) if agent_normalized_values else 0.0

    # Create output data with summary at the top
    output_data = {
        "summary": {
            "total_trajectories": len(scored_data),
            "total_positions": total_positions,
            "scored_positions": scored_positions,
            "average_value_change_cp": avg_value_change,
            "average_agent_normalized_change_cp": avg_agent_normalized,
            "value_change_count": len(value_changes),
            "agent_normalized_count": len(agent_normalized_values)
        },
        "trajectories": scored_data
    }

    print(f"\nSaving scored trajectories to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully scored and saved {len(scored_data)} trajectories")
    print(f"Scored {scored_positions}/{total_positions} positions")

    print(f"\nBoard value summary:")
    print(f"  Average value change: {avg_value_change:.2f} cp ({len(value_changes)} trajectories)")
    print(f"  Average agent normalized change: {avg_agent_normalized:.2f} cp ({len(agent_normalized_values)} trajectories)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score chess trajectory files with board evaluations")
    parser.add_argument("--input_file", required=True, help="Path to input trajectory JSON file")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel workers (defaults to CPU count)")
    parser.add_argument("--depth", type=int, default=16, help="Depth for board evaluation (default 16)")
    args = parser.parse_args()
    score_trajectory_file(args.input_file, num_workers=args.num_workers, depth=args.depth)