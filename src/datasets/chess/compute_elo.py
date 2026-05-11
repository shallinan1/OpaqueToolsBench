"""
Compute chess player ratings using two methods: Streaming Elo and Performance Rating.

This script analyzes chess game results against opponents with known Elo ratings and
computes two rating metrics:

1. STREAMING ELO: Processes games in shuffled order, updating rating after each game
   using standard Elo formulas. Tracks both final Elo and average Elo during the run.

   - Initial Elo: Uses 1500 if in opponent range, otherwise midpoint of opponent range
   - Bootstrap Mode (default): Runs multiple iterations with different orderings to
     compute confidence intervals (90%, 95%, 99%) for both final and average Elo
   - Single Run Mode: Use --bootstrap 0 to run once with --seed

2. PERFORMANCE RATING: Binary search to find rating where expected score equals actual
   score. Answers: "What rating produces this exact tournament result?"

3. PER-TRAJECTORY ANALYSIS (when available): If metadata contains per_trajectory_results,
   computes ELO independently for each trajectory, then calculates mean ± std across
   trajectories to measure experimental variance.

   - Each trajectory is treated as an independent trial
   - Bootstrap is applied within each trajectory for game-order uncertainty
   - Mean ± std across trajectories shows trial-to-trial variance
   - Provides both per-trajectory results and aggregate statistics

Opponents must follow elo_N format (e.g., elo_1500, elo_2000).

Usage:
    # Standard usage (aggregate analysis)
    python -m src.datasets.chess.compute_elo \
        --results-dir runs/chess/shared_tools/test_50q/my_agent_tools/agent \
        --bootstrap 100

    # With per-trajectory results (automatically detected if available)
    # Computes ELO for each trajectory independently, then reports mean ± std
"""

import json
import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def get_opponent_elo(opponent_name: str) -> int | None:
    """Get the Elo rating for an opponent.

    Args:
        opponent_name: Name of the opponent (e.g., "elo_1500", "elo_2000")

    Returns:
        Elo rating, or None if opponent name doesn't match elo_N format
    """
    if opponent_name.startswith("elo_"):
        try:
            return int(opponent_name[4:])
        except ValueError:
            return None
    return None


def update_elo(player_elo: float, opponent_elo: float, actual_score: float, k_factor: float = 32) -> float:
    """Update Elo rating based on a single game result.

    Args:
        player_elo: Current Elo rating of the player
        opponent_elo: Elo rating of the opponent
        actual_score: Game result (1.0 for win, 0.5 for draw, 0.0 for loss)
        k_factor: K-factor determining how much ratings change (default: 32 for rapid convergence)

    Returns:
        Updated Elo rating
    """
    expected_score = 1.0 / (1.0 + 10**((opponent_elo - player_elo) / 400.0))
    return player_elo + k_factor * (actual_score - expected_score)


def extract_player_from_path(results_dir: Path) -> str:
    """Extract the player name from the directory path.

    The player name is simply the last part of the path.
    E.g., .../best -> "best"

    Args:
        results_dir: Directory path

    Returns:
        The player name (last part of path)
    """
    return results_dir.name


def load_matchup_results(results_dir: Path, target_player: str | None = None) -> Dict[str, Dict]:
    """Load all matchup results from metadata files.

    Args:
        results_dir: Directory containing subdirectories with v0_metadata.json files
        target_player: Specific player to compute ELO for. If None, uses old behavior.

    Returns:
        Dictionary mapping opponent names to result statistics
    """
    matchups = defaultdict(lambda: {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_games": 0,
        "metadata_files": []
    })

    # Find all v0_metadata.json files
    for metadata_file in results_dir.rglob("v0_metadata.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Extract opponent info
        white_type = metadata.get("white_type")
        black_type = metadata.get("black_type")
        results = metadata.get("results", {})

        # Check if we have player-type-based results (new format with mirror mode support)
        white_type_name = results.get("white_type_name")
        black_type_name = results.get("black_type_name")
        has_player_type_results = "white_type_wins" in results and "black_type_wins" in results

        # Evaluate specific player regardless of naming pattern
        if white_type == target_player and black_type != target_player:
            # Target player is white, other is opponent
            opponent = black_type

            if has_player_type_results:
                # Use player-type-based results (accounts for mirror mode)
                # Verify that target_player matches white_type_name
                if white_type_name == target_player:
                    wins = results.get("white_type_wins", 0)
                    losses = results.get("black_type_wins", 0)
                else:
                    # Shouldn't happen, but fall back to color-based
                    wins = results.get("white_wins", 0)
                    losses = results.get("black_wins", 0)
            else:
                # Old format: use color-based results
                wins = results.get("white_wins", 0)
                losses = results.get("black_wins", 0)

            draws = results.get("draws", 0)

        elif black_type == target_player and white_type != target_player:
            # Target player is black, other is opponent
            opponent = white_type

            if has_player_type_results:
                # Use player-type-based results (accounts for mirror mode)
                # Verify that target_player matches black_type_name
                if black_type_name == target_player:
                    wins = results.get("black_type_wins", 0)
                    losses = results.get("white_type_wins", 0)
                else:
                    # Shouldn't happen, but fall back to color-based
                    wins = results.get("black_wins", 0)
                    losses = results.get("white_wins", 0)
            else:
                # Old format: use color-based results
                wins = results.get("black_wins", 0)
                losses = results.get("white_wins", 0)

            draws = results.get("draws", 0)

        else:
            # Either both are target player or neither is - skip
            continue

        # Aggregate results
        matchups[opponent]["wins"] += wins
        matchups[opponent]["losses"] += losses
        matchups[opponent]["draws"] += draws
        matchups[opponent]["total_games"] += wins + losses + draws
        matchups[opponent]["metadata_files"].append(str(metadata_file))

    return dict(matchups)


def load_trajectory_specific_matchups(results_dir: Path, target_player: str, trajectory_idx: int) -> Dict[str, Dict]:
    """Load matchup results for a specific trajectory.

    Args:
        results_dir: Directory containing subdirectories with v0_metadata.json files
        target_player: Specific player to compute ELO for
        trajectory_idx: Index of the trajectory to load (e.g., 0, 1, 2)

    Returns:
        Dictionary mapping opponent names to result statistics for this trajectory only
    """
    matchups = defaultdict(lambda: {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_games": 0,
        "metadata_files": []
    })

    trajectory_key = f"trajectory_{trajectory_idx}"

    # Find all v0_metadata.json files
    for metadata_file in results_dir.rglob("v0_metadata.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Check if per_trajectory_results exists
        per_trajectory_results = metadata.get("per_trajectory_results", {})
        if not per_trajectory_results or trajectory_key not in per_trajectory_results:
            continue

        # Get results for this specific trajectory
        traj_results = per_trajectory_results[trajectory_key]

        # Extract opponent info
        white_type = metadata.get("white_type")
        black_type = metadata.get("black_type")

        # Determine opponent
        if white_type == target_player and black_type != target_player:
            opponent = black_type
            # Use trajectory-specific wins/losses
            wins = traj_results.get("white_type_wins", 0)
            losses = traj_results.get("black_type_wins", 0)
        elif black_type == target_player and white_type != target_player:
            opponent = white_type
            # Use trajectory-specific wins/losses (reversed since target is black)
            wins = traj_results.get("black_type_wins", 0)
            losses = traj_results.get("white_type_wins", 0)
        else:
            # Either both are target player or neither is - skip
            continue

        draws = traj_results.get("draws", 0)

        # Aggregate results
        matchups[opponent]["wins"] += wins
        matchups[opponent]["losses"] += losses
        matchups[opponent]["draws"] += draws
        matchups[opponent]["total_games"] += wins + losses + draws
        matchups[opponent]["metadata_files"].append(str(metadata_file))

    return dict(matchups)


def check_trajectory_support(results_dir: Path) -> Tuple[bool, int]:
    """Check if per-trajectory results are available and count trajectories.

    Args:
        results_dir: Directory containing subdirectories with v0_metadata.json files

    Returns:
        Tuple of (has_trajectory_support, num_trajectories)
    """
    # Check the first metadata file to see if it has per_trajectory_results
    for metadata_file in results_dir.rglob("v0_metadata.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        per_trajectory_results = metadata.get("per_trajectory_results", {})
        if per_trajectory_results:
            # Count number of trajectories
            num_trajectories = len(per_trajectory_results)
            return True, num_trajectories
        else:
            return False, 0

    return False, 0


def compute_streaming_elo(matchups: Dict[str, Dict], initial_elo: float = None, k_factor: float = 32, seed: int = 0, verbose: bool = True) -> Tuple[float, float, float, Dict[str, Dict]]:
    """Compute Elo rating using streaming updates with shuffled game order.

    Starting from an initial Elo, processes all games in a random shuffled order,
    updating the Elo rating after each individual game.

    Args:
        matchups: Dictionary of opponent -> results
        initial_elo: Starting Elo rating (if None, uses 1500 or average of min/max opponent Elos, whichever is closer to opponent range)
        k_factor: K-factor for Elo updates (default: 32 for rapid convergence)
        seed: Random seed for shuffling games (default: 0)
        verbose: Whether to print progress (default: True)

    Returns:
        Tuple of (initial_elo, final_elo, average_elo, opponent_summaries)
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Collect all individual games into a single list and find opponent Elo range
    all_games = []
    opponent_elos = []

    for opponent, stats in matchups.items():
        opponent_elo = get_opponent_elo(opponent)
        if opponent_elo is not None and stats["total_games"] > 0:
            opponent_elos.append(opponent_elo)

            # Add individual wins
            for _ in range(stats["wins"]):
                all_games.append({
                    "opponent": opponent,
                    "opponent_elo": opponent_elo,
                    "result": "W",
                    "score": 1.0
                })
            # Add individual draws
            for _ in range(stats["draws"]):
                all_games.append({
                    "opponent": opponent,
                    "opponent_elo": opponent_elo,
                    "result": "D",
                    "score": 0.5
                })
            # Add individual losses
            for _ in range(stats["losses"]):
                all_games.append({
                    "opponent": opponent,
                    "opponent_elo": opponent_elo,
                    "result": "L",
                    "score": 0.0
                })

    # Calculate initial Elo if not provided
    # Strategy: Use 1500 as baseline, but if opponent range is far from 1500, use the midpoint instead
    if initial_elo is None:
        if opponent_elos:
            min_elo = min(opponent_elos)
            max_elo = max(opponent_elos)
            midpoint = (min_elo + max_elo) / 2

            # Use 1500 if it's within the opponent range, otherwise use midpoint
            if min_elo <= 1500 <= max_elo:
                initial_elo = 1500
            else:
                initial_elo = midpoint
        else:
            initial_elo = 1500  # Fallback if no valid opponents

    current_elo = initial_elo

    # Shuffle all games
    random.shuffle(all_games)

    if verbose:
        print(f"\nProcessing {len(all_games)} games in random order (shuffled with seed={seed})...")
        print("-" * 70)

    # Initialize opponent summaries
    opponent_summaries = {}
    for opponent, stats in matchups.items():
        opponent_elo = get_opponent_elo(opponent)
        if opponent_elo is not None:
            opponent_summaries[opponent] = {
                "opponent_elo": opponent_elo,
                "wins": 0,
                "draws": 0,
                "losses": 0
            }

    # Track running sum of ELOs to compute average
    elo_sum = 0.0

    # Process each game one by one
    for i, game in enumerate(all_games, 1):
        current_elo = update_elo(current_elo, game["opponent_elo"], game["score"], k_factor)
        elo_sum += current_elo

        # Update opponent summary
        opponent = game["opponent"]
        if game["result"] == "W":
            opponent_summaries[opponent]["wins"] += 1
        elif game["result"] == "D":
            opponent_summaries[opponent]["draws"] += 1
        else:
            opponent_summaries[opponent]["losses"] += 1

        # Print progress every 10 games
        if verbose and (i % 10 == 0 or i == len(all_games)):
            print(f"Game {i:3}/{len(all_games)}: Elo = {current_elo:.0f}")

    # Compute average ELO during the run
    average_elo = elo_sum / len(all_games) if len(all_games) > 0 else current_elo

    # Print summary by opponent
    if verbose:
        print("\n" + "-" * 70)
        print("Summary by opponent:")
        for opponent in sorted(opponent_summaries.keys(), key=lambda x: get_opponent_elo(x) or 0):
            summary = opponent_summaries[opponent]
            total = summary["wins"] + summary["draws"] + summary["losses"]
            score_pct = (summary["wins"] + 0.5 * summary["draws"]) / total * 100 if total > 0 else 0
            print(f"vs {opponent:10} (Elo {summary['opponent_elo']:4}): "
                  f"W:{summary['wins']:2} D:{summary['draws']:2} L:{summary['losses']:2} | "
                  f"Score: {score_pct:5.1f}%")

        print("-" * 70)
        print(f"Final Elo: {current_elo:.0f}")
        print(f"Average Elo: {average_elo:.0f}")

    return initial_elo, current_elo, average_elo, opponent_summaries


def compute_confidence_intervals(elos: List[float]) -> Dict[str, Tuple[float, float]]:
    """Compute confidence intervals for a list of ELO ratings.

    Args:
        elos: List of ELO ratings from bootstrap samples

    Returns:
        Dictionary mapping confidence level to (lower, upper) bounds
    """
    if not elos:
        return {}

    elos_array = np.array(elos)

    confidence_intervals = {}
    for conf_level in [90, 95, 99]:
        alpha = (100 - conf_level) / 100
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower = np.percentile(elos_array, lower_percentile)
        upper = np.percentile(elos_array, upper_percentile)

        confidence_intervals[conf_level] = (lower, upper)

    return confidence_intervals


def compute_bootstrap_elo(matchups: Dict[str, Dict], initial_elo: float = None, k_factor: float = 32, n_bootstrap: int = 100) -> Tuple[float, List[float], Dict[str, Tuple[float, float]], List[float], Dict[str, Tuple[float, float]]]:
    """Compute streaming ELO with bootstrap confidence intervals.

    Runs streaming ELO calculation multiple times with different random permutations
    of games to estimate uncertainty in the ELO rating.

    Args:
        matchups: Dictionary of opponent -> results
        initial_elo: Starting Elo rating (if None, uses 1500 or average of min/max opponent Elos)
        k_factor: K-factor for Elo updates (default: 32)
        n_bootstrap: Number of bootstrap iterations (default: 100)

    Returns:
        Tuple of (median_elo, all_elos, confidence_intervals, all_average_elos, average_confidence_intervals)
    """
    print(f"\nRunning {n_bootstrap} bootstrap iterations...")

    all_elos = []
    all_average_elos = []

    for i in tqdm(range(n_bootstrap), desc="Bootstrap progress", ncols=100):
        _, final_elo, average_elo, _ = compute_streaming_elo(
            matchups,
            initial_elo=initial_elo,
            k_factor=k_factor,
            seed=i,
            verbose=False
        )
        all_elos.append(final_elo)
        all_average_elos.append(average_elo)

    median_elo = np.median(all_elos)
    confidence_intervals = compute_confidence_intervals(all_elos)

    median_average_elo = np.median(all_average_elos)
    average_confidence_intervals = compute_confidence_intervals(all_average_elos)

    print(f"\nBootstrap Results ({n_bootstrap} iterations):")
    print(f"  Final ELO - Median: {median_elo:.0f}, Mean: {np.mean(all_elos):.0f}, Std Dev: {np.std(all_elos):.1f}")
    print(f"  Average ELO - Median: {median_average_elo:.0f}, Mean: {np.mean(all_average_elos):.0f}, Std Dev: {np.std(all_average_elos):.1f}")
    print(f"  Min Elo: {min(all_elos):.0f}")
    print(f"  Max Elo: {max(all_elos):.0f}")
    print(f"  Range: {max(all_elos) - min(all_elos):.0f}")
    print(f"\nFinal ELO Confidence Intervals:")
    for conf_level in [90, 95, 99]:
        lower, upper = confidence_intervals[conf_level]
        width = upper - lower
        print(f"  {conf_level}% CI: [{lower:.0f}, {upper:.0f}] (width: {width:.0f})")
    print(f"\nAverage ELO Confidence Intervals:")
    for conf_level in [90, 95, 99]:
        lower, upper = average_confidence_intervals[conf_level]
        width = upper - lower
        print(f"  {conf_level}% CI: [{lower:.0f}, {upper:.0f}] (width: {width:.0f})")

    return median_elo, all_elos, confidence_intervals, all_average_elos, average_confidence_intervals


def compute_performance_rating(matchups: Dict[str, Dict]) -> float:
    """Compute performance rating using binary search method.

    Finds the rating where expected tournament score equals actual score.
    This answers: "What rating would a player need to have for this
    tournament result to be perfectly expected?"

    Args:
        matchups: Dictionary of opponent -> results

    Returns:
        Performance rating
    """
    # Calculate actual score and collect opponent ratings for each game
    actual_score = 0.0
    opponent_ratings = []

    for opponent, stats in matchups.items():
        opponent_elo = get_opponent_elo(opponent)
        if opponent_elo is not None and stats["total_games"] > 0:
            # Each win = 1.0, draw = 0.5, loss = 0.0
            actual_score += stats["wins"] + 0.5 * stats["draws"]

            # Add opponent rating for each game played
            for _ in range(stats["total_games"]):
                opponent_ratings.append(opponent_elo)

    if not opponent_ratings:
        return 1500  # Default if no valid opponents

    total_games = len(opponent_ratings)

    def expected_score_at_rating(rating: float) -> float:
        """Calculate expected score if player had this rating."""
        total_expected = 0.0
        for opp_rating in opponent_ratings:
            expected = 1.0 / (1.0 + 10**((opp_rating - rating) / 400.0))
            total_expected += expected
        return total_expected

    # Binary search bounds
    min_rating = 0.0
    max_rating = 4000.0
    tolerance = 0.01

    # Binary search for rating where expected score = actual score
    while max_rating - min_rating > tolerance:
        mid_rating = (min_rating + max_rating) / 2
        expected = expected_score_at_rating(mid_rating)

        if expected < actual_score:
            # Need higher rating to get higher expected score
            min_rating = mid_rating
        else:
            # Need lower rating
            max_rating = mid_rating

    performance_rating = (min_rating + max_rating) / 2

    print(f"\nPerformance Rating Calculation:")
    print(f"  Actual score: {actual_score:.1f}/{total_games} ({actual_score/total_games*100:.1f}%)")
    print(f"  Performance rating: {performance_rating:.0f}")
    print(f"  (Rating where expected score = actual score)")

    return performance_rating


def print_results(matchups: Dict[str, Dict], initial_elo: float, final_elo: float, performance_rating: float,
                  bootstrap_median: float = None, confidence_intervals: Dict[str, Tuple[float, float]] = None):
    """Print formatted results summary.

    Args:
        matchups: Dictionary of opponent -> results
        initial_elo: Initial Elo rating before streaming updates
        final_elo: Final Elo rating after streaming updates (single run or bootstrap median)
        performance_rating: Performance rating from binary search method
        bootstrap_median: Median ELO from bootstrap (if bootstrap was run)
        confidence_intervals: Bootstrap confidence intervals (if bootstrap was run)
    """
    print("\n" + "="*70)
    print("CHESS PLAYER RATING EVALUATION")
    print("="*70)

    # Calculate overall statistics
    total_games = sum(s['total_games'] for s in matchups.values())
    total_wins = sum(s['wins'] for s in matchups.values())
    total_losses = sum(s['losses'] for s in matchups.values())
    total_draws = sum(s['draws'] for s in matchups.values())
    overall_score_pct = (total_wins + 0.5 * total_draws) / total_games * 100 if total_games > 0 else 0

    if bootstrap_median is not None:
        print(f"\n1. Bootstrap Streaming Elo Rating: {bootstrap_median:.0f}")
        print(f"   (Median of multiple game orderings)")
        if confidence_intervals:
            print(f"\n   Confidence Intervals:")
            for conf_level in [90, 95, 99]:
                if conf_level in confidence_intervals:
                    lower, upper = confidence_intervals[conf_level]
                    print(f"     {conf_level}% CI: [{lower:.0f}, {upper:.0f}]")
    else:
        print(f"\n1. Streaming Elo Rating: {final_elo:.0f}")
        print(f"   (Game-by-game updates with shuffled order)")

    print(f"\n2. Performance Rating: {performance_rating:.0f}")
    print(f"   (Rating where expected score = actual score)")

    print(f"\nTotal Record: W:{total_wins} L:{total_losses} D:{total_draws} ({total_games} games)")
    print(f"Overall Score: {overall_score_pct:.1f}%")

    # Show Elo progression summary
    if bootstrap_median is not None:
        print(f"\nStreaming Elo: {initial_elo:.0f} → {bootstrap_median:.0f} ({bootstrap_median - initial_elo:+.0f})")
    else:
        print(f"\nStreaming Elo Progression: {initial_elo:.0f} → {final_elo:.0f} ({final_elo - initial_elo:+.0f})")

    print("\n" + "="*70)
    print(f"Total games analyzed: {total_games}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Elo rating for chess player using streaming updates (supports agent, probabilistic, and other player types)"
    )
    parser.add_argument("--results-dir", required=True, type=Path, help="Directory containing matchup results (e.g., runs/chess/shared_tools/test/my_agent)")
    parser.add_argument("--initial-elo", type=float, default=None, help="Initial Elo rating (default: average of min and max opponent Elos)")
    parser.add_argument("--k-factor", type=float, default=32, help="K-factor for Elo updates (default: 32 for rapid convergence)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling games (default: 0, only used if --bootstrap is not specified)")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap iterations for confidence intervals (default: 10000). Set to 0 to disable and use single-run mode with --seed.")

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    # Extract player name from the path (last component)
    target_player = extract_player_from_path(args.results_dir)
    print(f"Computing ELO for player: '{target_player}'")

    # Find all vs_* subdirectories
    vs_dirs = [d for d in args.results_dir.iterdir() if d.is_dir() and d.name.startswith("vs_")]
    if not vs_dirs:
        print(f"No vs_* subdirectories found in {args.results_dir}")
        return

    print(f"Found {len(vs_dirs)} opponent directories: {[d.name for d in vs_dirs]}")
    print(f"Loading matchup results from: {args.results_dir}")

    matchups = load_matchup_results(args.results_dir, target_player)

    if not matchups:
        print("No matchup results found!")
        return

    print(f"Found results for {len(matchups)} opponent(s)")

    # Check for per-trajectory support
    has_trajectory_support, num_trajectories = check_trajectory_support(args.results_dir)

    # Get opponent Elo range for display
    opponent_elos = [get_opponent_elo(opp) for opp in matchups.keys()]
    opponent_elos = [elo for elo in opponent_elos if elo is not None]
    if opponent_elos:
        print(f"Opponent Elo range: {min(opponent_elos)} to {max(opponent_elos)}")

    # Determine initial ELO for display (will be same as what compute functions use)
    if args.initial_elo is None:
        if opponent_elos:
            min_elo = min(opponent_elos)
            max_elo = max(opponent_elos)
            if min_elo <= 1500 <= max_elo:
                determined_initial_elo = 1500
            else:
                determined_initial_elo = (min_elo + max_elo) / 2
        else:
            determined_initial_elo = 1500
    else:
        determined_initial_elo = args.initial_elo

    # Print initial Elo info
    print(f"\nStarting Elo: {determined_initial_elo:.0f}")
    if args.initial_elo is None:
        if opponent_elos and min(opponent_elos) <= 1500 <= max(opponent_elos):
            print("  (using standard 1500 baseline - falls within opponent range)")
        elif opponent_elos:
            midpoint = (min(opponent_elos) + max(opponent_elos)) / 2
            print(f"  (using midpoint {midpoint:.0f} - standard 1500 is outside opponent range [{min(opponent_elos)}, {max(opponent_elos)}])")
        else:
            print("  (using default 1500 - no valid opponents found)")
    else:
        print("  (user-specified)")

    # If per-trajectory results are available, compute ELO for each trajectory
    trajectory_streaming_elos = []
    trajectory_performance_ratings = []
    trajectory_details = {}

    if has_trajectory_support:
        print(f"\n{'='*70}")
        print(f"PER-TRAJECTORY ANALYSIS")
        print(f"Found {num_trajectories} trajectories - computing ELO for each independently")
        print(f"{'='*70}")

        for traj_idx in range(num_trajectories):
            print(f"\nProcessing trajectory {traj_idx}...")

            # Load matchups for this specific trajectory
            traj_matchups = load_trajectory_specific_matchups(args.results_dir, target_player, traj_idx)

            if not traj_matchups:
                print(f"  No matchups found for trajectory {traj_idx}")
                continue

            # Count games for this trajectory
            traj_total_games = sum(s['total_games'] for s in traj_matchups.values())
            print(f"  Games in trajectory: {traj_total_games}")

            # Compute streaming ELO for this trajectory
            if args.bootstrap and args.bootstrap > 0:
                # Bootstrap mode for this trajectory
                traj_median, traj_all_elos, traj_ci, traj_avg_elos, traj_avg_ci = compute_bootstrap_elo(
                    traj_matchups,
                    args.initial_elo,
                    args.k_factor,
                    args.bootstrap
                )
                trajectory_streaming_elos.append(traj_median)
                print(f"  Bootstrap Streaming ELO: {traj_median:.0f} (95% CI: [{traj_ci[95][0]:.0f}, {traj_ci[95][1]:.0f}])")

                trajectory_details[f"trajectory_{traj_idx}"] = {
                    "streaming_elo": traj_median,
                    "streaming_elo_ci_95": [traj_ci[95][0], traj_ci[95][1]],
                    "average_elo": float(np.median(traj_avg_elos))
                }
            else:
                # Single run mode for this trajectory
                _, traj_final_elo, traj_avg_elo, _ = compute_streaming_elo(
                    traj_matchups,
                    args.initial_elo,
                    args.k_factor,
                    args.seed,
                    verbose=False
                )
                trajectory_streaming_elos.append(traj_final_elo)
                print(f"  Streaming ELO: {traj_final_elo:.0f}")

                trajectory_details[f"trajectory_{traj_idx}"] = {
                    "streaming_elo": traj_final_elo,
                    "average_elo": traj_avg_elo
                }

            # Compute performance rating for this trajectory
            traj_perf_rating = compute_performance_rating(traj_matchups)
            trajectory_performance_ratings.append(traj_perf_rating)
            trajectory_details[f"trajectory_{traj_idx}"]["performance_rating"] = traj_perf_rating
            print(f"  Performance Rating: {traj_perf_rating:.0f}")

        # Calculate statistics across trajectories
        if trajectory_streaming_elos:
            print(f"\n{'='*70}")
            print("TRAJECTORY STATISTICS")
            print(f"{'='*70}")

            streaming_mean = np.mean(trajectory_streaming_elos)
            streaming_std = np.std(trajectory_streaming_elos)
            streaming_median = np.median(trajectory_streaming_elos)

            perf_mean = np.mean(trajectory_performance_ratings)
            perf_std = np.std(trajectory_performance_ratings)
            perf_median = np.median(trajectory_performance_ratings)

            print(f"\nStreaming ELO across {len(trajectory_streaming_elos)} trajectories:")
            print(f"  Mean ± Std: {streaming_mean:.0f} ± {streaming_std:.0f}")
            print(f"  Median: {streaming_median:.0f}")
            print(f"  Min: {min(trajectory_streaming_elos):.0f}, Max: {max(trajectory_streaming_elos):.0f}")
            print(f"  Individual values: {[f'{e:.0f}' for e in trajectory_streaming_elos]}")

            print(f"\nPerformance Rating across {len(trajectory_performance_ratings)} trajectories:")
            print(f"  Mean ± Std: {perf_mean:.0f} ± {perf_std:.0f}")
            print(f"  Median: {perf_median:.0f}")
            print(f"  Min: {min(trajectory_performance_ratings):.0f}, Max: {max(trajectory_performance_ratings):.0f}")
            print(f"  Individual values: {[f'{r:.0f}' for r in trajectory_performance_ratings]}")

    # Compute Elo using streaming updates with or without bootstrap (aggregate)
    bootstrap_median = None
    confidence_intervals = None
    all_elos = None
    all_average_elos = None
    average_confidence_intervals = None

    print(f"\n{'='*70}")
    print("AGGREGATE ANALYSIS (All trajectories combined)")
    print(f"{'='*70}")

    if args.bootstrap and args.bootstrap > 0:
        # Bootstrap mode
        print(f"\nBootstrap mode enabled with {args.bootstrap} iterations")
        bootstrap_median, all_elos, confidence_intervals, all_average_elos, average_confidence_intervals = compute_bootstrap_elo(
            matchups,
            args.initial_elo,
            args.k_factor,
            args.bootstrap
        )
        final_elo = bootstrap_median

        # Get initial elo from first bootstrap run for consistency
        initial_elo = determined_initial_elo
    else:
        # Single run mode
        print(f"\nSingle-run mode (bootstrap disabled)")
        print(f"Note: Different random seeds can produce ±30-50 Elo variation.")
        print(f"      Current seed: {args.seed}")
        print(f"      Consider using --bootstrap for confidence intervals.")

        initial_elo, final_elo, average_elo, opponent_summaries = compute_streaming_elo(
            matchups,
            args.initial_elo,
            args.k_factor,
            args.seed
        )

    # Compute performance rating using binary search
    performance_rating = compute_performance_rating(matchups)

    # Print results summary
    print_results(matchups, initial_elo, final_elo, performance_rating, bootstrap_median, confidence_intervals)

    # Save to file in results directory with player name
    output_file = args.results_dir / f"{target_player}_elo_evaluation.json"

    total_games = sum(s["total_games"] for s in matchups.values())

    # Build streaming ELO section
    streaming_elo_data = {
        "final_elo": final_elo,
        "initial_elo": initial_elo,
        "initial_elo_method": "user-specified" if args.initial_elo is not None else "auto (1500 or midpoint)",
        "k_factor": args.k_factor,
    }

    if args.bootstrap and args.bootstrap > 0:
        streaming_elo_data.update({
            "methodology": "Bootstrap with multiple game orderings",
            "n_bootstrap": args.bootstrap,
            "median_elo": float(bootstrap_median),
            "mean_elo": float(np.mean(all_elos)),
            "std_elo": float(np.std(all_elos)),
            "confidence_intervals": {
                f"{level}%": {
                    "lower": float(ci[0]),
                    "upper": float(ci[1])
                }
                for level, ci in confidence_intervals.items()
            },
            "median_average_elo": float(np.median(all_average_elos)),
            "mean_average_elo": float(np.mean(all_average_elos)),
            "std_average_elo": float(np.std(all_average_elos)),
            "average_elo_confidence_intervals": {
                f"{level}%": {
                    "lower": float(ci[0]),
                    "upper": float(ci[1])
                }
                for level, ci in average_confidence_intervals.items()
            }
        })
    else:
        streaming_elo_data.update({
            "methodology": "Game-by-game updates with shuffled order",
            "seed": args.seed,
            "seed_note": "Different seeds can produce ±30-50 Elo variation"
        })

    output_data = {
        "player_evaluated": target_player,
        "aggregate_streaming_elo": streaming_elo_data,
        "aggregate_performance_rating": {
            "rating": performance_rating,
            "methodology": "Binary search to find rating where expected score = actual score"
        },
        "total_games": total_games,
        "matchups": {
            opponent: {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "draws": stats["draws"],
                "total_games": stats["total_games"],
                "score_percentage": (stats["wins"] + 0.5 * stats["draws"]) / stats["total_games"] * 100 if stats["total_games"] > 0 else 0,
                "opponent_elo": get_opponent_elo(opponent)
            }
            for opponent, stats in matchups.items()
        }
    }

    # Add trajectory-level results if available
    if has_trajectory_support and trajectory_streaming_elos:
        output_data["trajectory_analysis"] = {
            "num_trajectories": num_trajectories,
            "per_trajectory_results": trajectory_details,
            "trajectory_statistics": {
                "streaming_elo": {
                    "mean": float(np.mean(trajectory_streaming_elos)),
                    "std": float(np.std(trajectory_streaming_elos)),
                    "median": float(np.median(trajectory_streaming_elos)),
                    "min": float(min(trajectory_streaming_elos)),
                    "max": float(max(trajectory_streaming_elos)),
                    "values": [float(e) for e in trajectory_streaming_elos]
                },
                "performance_rating": {
                    "mean": float(np.mean(trajectory_performance_ratings)),
                    "std": float(np.std(trajectory_performance_ratings)),
                    "median": float(np.median(trajectory_performance_ratings)),
                    "min": float(min(trajectory_performance_ratings)),
                    "max": float(max(trajectory_performance_ratings)),
                    "values": [float(r) for r in trajectory_performance_ratings]
                }
            },
            "interpretation": "Each trajectory represents an independent trial. Mean ± std shows experimental variance across trials."
        }

    # Add all_elos and all_average_elos at the very end if bootstrap was used
    if args.bootstrap and args.bootstrap > 0:
        output_data["all_elos"] = [float(elo) for elo in all_elos]
        output_data["all_average_elos"] = [float(elo) for elo in all_average_elos]

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
