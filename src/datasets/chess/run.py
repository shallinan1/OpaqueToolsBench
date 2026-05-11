"""
Chess dataset trajectory collection runner - simplified batch processing version.

This module plays chess games in parallel batches, collecting trajectories with all moves and tool calls.
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import random

import chess

from src.generation_utils.openai_parallel_generate import openai_parallel_generate, requests_url_dict, default_request_url
from src.generation_utils.rate_limits import get_rate_limit, get_token_limit
from src.datasets.chess.args import create_chess_parser
from src.datasets.chess.prompts import CHESS_AGENT_PROMPTS
from src.datasets.chess.chess_game_session_v2 import cleanup_engines
from src.datasets.chess.utils.session_management import SessionPool, GameState
from src.datasets.chess.utils.tool_execution import execute_tools_parallel, check_and_mark_game_over
from src.datasets.chess.utils.api_helpers import create_api_request, process_batch_responses
from src.datasets.chess.utils.path_utils import build_output_folder, parse_config_name, detect_improvement_context
from src.datasets.chess.probabilistic_player import (
    parse_tool_probabilities,
    select_tool_probabilistic,
    format_probabilities_for_path
)
from src.datasets.chess.baseline_player import (
    select_best_tool,
    select_worst_tool
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("src.generation_utils.openai").setLevel(logging.WARNING)
logging.getLogger("src.generation_utils.openai_parallel_generate").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

def play_games_batch(games: Dict[int, GameState], tools: List[Dict], args,
                     function_mapping: Optional[Dict] = None) -> Tuple[Dict[int, GameState], Dict[str, Any]]:
    """Play all games in parallel batches until completion.

    Args:
        games: Dictionary of game_id -> GameState
        tools: Available tools
        args: Command line arguments
        function_mapping: Optional tool name mapping

    Returns:
        Tuple of (Dictionary of completed games, Token usage statistics)
    """
    # Convert tools to OpenAI format once
    openai_tools = [{"type": "function", "function": tool} for tool in tools]
    max_requests_per_minute = get_rate_limit(args.model)
    max_tokens_per_minute = get_token_limit(args.model)

    pbar = tqdm(range(args.max_moves), desc="Playing moves", unit="move")
    for move_num in pbar:
        active_games = {gid: game for gid, game in games.items()
                       if not game.is_complete and game.move_count < args.max_moves}

        if not active_games:
            logger.info(f"All games complete after {move_num} moves")
            break

        # Note: Games may not be synchronized when using mixed player types
        # Each game will determine its own current color
        first_game = next(iter(active_games.values()))

        # Update tqdm description with active game count
        pbar.set_description(f"Playing moves ({len(active_games)} active games)")

        # Batch 1: Collect all moves needed (both non-agent and agent)
        api_requests = []
        non_agent_calls = []  # For parallel execution of non-agent tools
        non_agent_metadata = {}  # Store metadata for non-agent games

        for game_id, game in active_games.items():
            # Determine color for THIS specific game based on its board state
            game_color = "white" if game.board.turn == chess.WHITE else "black"

            # For specific tool players, execute tool directly
            player_type = game.white_type if game_color == "white" else game.black_type

            if player_type != "agent":  # Direct tool execution (no API call needed)
                if player_type == "probabilistic": # Handle probabilistic tool selection
                    if game_color == "white":
                        tool_probs = getattr(args, '_white_tool_probs', {})
                    else:
                        tool_probs = getattr(args, '_black_tool_probs', {})

                    # Select tool based on probabilities
                    selected_tool = select_tool_probabilistic(tool_probs)
                elif player_type == "best": # Select best tool for current position
                    selected_tool = select_best_tool(game.board, tools, function_mapping)
                elif player_type == "worst": # Select worst tool for current position
                    selected_tool = select_worst_tool(game.board, tools, function_mapping)
                else: # Use the specified tool directly
                    selected_tool = player_type

                # Add to parallel execution list
                non_agent_calls.append((game_id, selected_tool, game.board, function_mapping))
                non_agent_metadata[game_id] = {
                    "game": game,
                    "color": game_color,
                    "player_type": player_type,  # Keep original type (e.g., "probabilistic")
                    "selected_tool": selected_tool  # The actual tool used
                }
            else:  # Agent player - needs API call
                api_request = create_api_request(game, openai_tools, args, game_color)
                api_requests.append(api_request)

        # Execute all non-agent tool calls in parallel using session pool
        if non_agent_calls:
            max_workers = getattr(args, 'parallel_workers', 8)
            session_pool = getattr(args, '_session_pool')  # Session pool passed via args
            non_agent_results = execute_tools_parallel(non_agent_calls, session_pool, max_workers)

            # Apply non-agent results
            for game_id, move in non_agent_results.items():
                metadata = non_agent_metadata[game_id]
                game = metadata["game"]
                game_color = metadata["color"]
                player_type = metadata["player_type"]
                selected_tool = metadata["selected_tool"]

                if move and not move.startswith("Error"):
                    try:
                        move_obj = game.board.parse_san(move)
                        game.board.push(move_obj)
                        game.move_count += 1

                        # Record the move with the correct game-specific color; for probabilistic players, record both the type and actual tool used
                        move_data = {
                            "move_number": game.move_count,
                            "color": game_color,
                            "move": move,
                            "fen_after": game.board.fen(),
                            "tool_used": selected_tool
                        }
                        if player_type == "probabilistic":
                            move_data["player_type"] = "probabilistic"
                        game.moves.append(move_data)
                    except Exception as e:
                        logger.error(f"Game {game_id}: Failed to apply {player_type} move {move}: {e}")
                        game.is_complete = True
                        game.result = "0-1" if game_color == "white" else "1-0"
                        game.termination = f"Invalid move by {game_color}"
                else:
                    logger.error(f"Game {game_id}: {player_type} failed to produce valid move: {move}")
                    game.is_complete = True
                    game.result = "0-1" if game_color == "white" else "1-0"
                    game.termination = f"No valid move by {game_color}"
        
        if not api_requests:
            for game in active_games.values():
                check_and_mark_game_over(game, args.max_moves)
            continue

        if args.verbose:
            print(api_requests[0]["messages"][0]["content"])
            print(api_requests[0]["messages"][1]["content"])

        api_responses = asyncio.run(openai_parallel_generate(
            api_requests,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            request_url=requests_url_dict.get(args.model, default_request_url)))
        process_batch_responses(api_responses, games, args, function_mapping)

        if args.verbose:
            # for i in range(len(api_responses)):
            #     text_response = api_responses[i][1]["choices"][0]["message"]
            #     if "content" in text_response:
            #         print(text_response["content"])
            #     if "tool_calls" in text_response:
            #         print(text_response["tool_calls"])
            #     print("\n\n\n")
            text_response = api_responses[0][1]["choices"][0]["message"]
            if "content" in text_response:
                print(text_response["content"])
            if "tool_calls" in text_response:
                print(text_response["tool_calls"])

        # Check for game endings
        for game in active_games.values():
            check_and_mark_game_over(game, args.max_moves)

    # Aggregate token usage from all games
    total_prompt_tokens = sum(game.token_usage["prompt_tokens"] for game in games.values())
    total_completion_tokens = sum(game.token_usage["completion_tokens"] for game in games.values())
    total_tokens = sum(game.token_usage["total_tokens"] for game in games.values())

    token_usage = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "model": args.model
    }

    return games, token_usage


def save_trajectories(games: Dict[int, GameState], args, config: Dict[str, Any], processing_time: float,
                     token_usage: Dict[str, Any], mode: str, config_path: str, output_folder: Path,
                     is_improvement: bool = False, improvement_version: Optional[int] = None):
    """Save game trajectories to output directory.

    Args:
        games: Dictionary of completed games
        args: Command line arguments
        config: Tool configuration
        processing_time: Total processing time in seconds
        token_usage: Token usage statistics
        mode: Either "tool_config" or "tool_set" to indicate which mode was used
        config_path: Path to the configuration file used
        output_folder: Pre-computed output folder path
        is_improvement: Whether this is an improvement iteration (saves without v0_ prefix)
        improvement_version: Version number if this is an improvement iteration
    """
    # Create timestamps for metadata only (not directory)
    now = datetime.utcnow()
    ts_machine = now.strftime("%Y%m%dT%H%M%S%fZ")
    ts_human = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Convert games to trajectory format
    trajectories = []
    for game_id, game in games.items():
        trajectory = {
            "game_id": game_id,
            "start_fen": game.start_fen,
            "white_type": game.white_type,  # Use game-specific types (might be swapped)
            "black_type": game.black_type,  # Use game-specific types (might be swapped)
            "moves": game.moves,
            "tool_calls": game.tool_calls,
            "result": game.result,
            "termination": game.termination,
            "final_fen": game.board.fen(),
            "total_moves": game.move_count,
            "full_responses": game.full_responses,  # Include all API interactions
            "prompt_key": args.prompt_key,
            "prompt": CHESS_AGENT_PROMPTS[args.prompt_key],  # Include the actual prompt used
            "token_usage": game.token_usage  # Per-game token usage
        }
        trajectories.append(trajectory)

    # For improvement iterations, save without version prefix in the version directory
    # For base runs, save as v0_trajectories.json
    if is_improvement:
        version = f"v{improvement_version}" if improvement_version is not None else "v0"
        trajectories_path = output_folder / "trajectories.json"
    else:
        version = "v0"
        trajectories_path = output_folder / f"{version}_trajectories.json"
    with open(trajectories_path, 'w') as f:
        json.dump(trajectories, f, indent=2)

    logger.info(f"Saved {len(trajectories)} trajectories to {trajectories_path}")

    # Calculate statistics by color and by player type
    results = {"white_wins": 0, "black_wins": 0, "draws": 0}

    # Track wins by player type (accounts for mirror mode where same type plays both colors)
    white_type_wins = 0  # Wins by the args.white_type player
    black_type_wins = 0  # Wins by the args.black_type player

    for game in games.values():
        if game.result == "1-0":
            results["white_wins"] += 1
            # White pieces won - check which player type was playing white
            if game.white_type == args.white_type:
                white_type_wins += 1
            else:
                black_type_wins += 1
        elif game.result == "0-1":
            results["black_wins"] += 1
            # Black pieces won - check which player type was playing black
            if game.black_type == args.white_type:
                white_type_wins += 1
            else:
                black_type_wins += 1
        elif game.result == "1/2-1/2":
            results["draws"] += 1

    # Add player type results to the results dictionary
    results["white_type_wins"] = white_type_wins
    results["black_type_wins"] = black_type_wins
    results["white_type_name"] = args.white_type
    results["black_type_name"] = args.black_type

    # Calculate per-trajectory results (treating each trajectory as a separate trial)
    per_trajectory_results = {}
    for i in range(args.num_trajectories):
        per_trajectory_results[f"trajectory_{i}"] = {
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "white_type_wins": 0,
            "black_type_wins": 0,
            "total_games": 0,
            "game_ids": []
        }

    # Assign games to trajectories based on game_id
    for game_id, game in games.items():
        # Calculate which trajectory this game belongs to
        if args.mirror:
            # With mirror mode, every 2 games belong to the same trajectory
            trajectory_idx = (game_id // 2) % args.num_trajectories
        else:
            # Without mirror mode, use simple modulo
            trajectory_idx = game_id % args.num_trajectories

        traj_key = f"trajectory_{trajectory_idx}"
        per_trajectory_results[traj_key]["game_ids"].append(game_id)
        per_trajectory_results[traj_key]["total_games"] += 1

        # Count results by color
        if game.result == "1-0":
            per_trajectory_results[traj_key]["white_wins"] += 1
            # Count by player type
            if game.white_type == args.white_type:
                per_trajectory_results[traj_key]["white_type_wins"] += 1
            else:
                per_trajectory_results[traj_key]["black_type_wins"] += 1
        elif game.result == "0-1":
            per_trajectory_results[traj_key]["black_wins"] += 1
            # Count by player type
            if game.black_type == args.white_type:
                per_trajectory_results[traj_key]["white_type_wins"] += 1
            else:
                per_trajectory_results[traj_key]["black_type_wins"] += 1
        elif game.result == "1/2-1/2":
            per_trajectory_results[traj_key]["draws"] += 1

    avg_moves = sum(g.move_count for g in games.values()) / len(games) if games else 0

    # Save metadata with versioning and all hyperparameters
    metadata = {
        "version": version,
        "timestamp": ts_human,
        "timestamp_utc": ts_machine,
        "mode": mode,  # Clearly indicate which mode was used
        "data_source": "embedded_in_config" if mode == "tool_config" else "external_split",  # Clarify data source
        "config_path": config_path,  # Path to the configuration file
        "config_source": config_path,  # Also save as config_source for compatibility with generate_descriptions
        "model": args.model,
        "temperature": args.temperature if not (args.model.startswith('o') or 'gpt-5' in args.model) else None,
        "top_p": args.top_p if hasattr(args, 'top_p') and not (args.model.startswith('o') or 'gpt-5' in args.model) else None,
        "reasoning_effort": args.reasoning_effort if hasattr(args, 'reasoning_effort') and (args.model.startswith('o') or 'gpt-5' in args.model) else None,
        "max_tokens": args.max_tokens,
        "tool_choice": args.tool_choice,
        "prompt_key": args.prompt_key,
        "num_trajectories": args.num_trajectories,
        "mirror": args.mirror,  # Whether mirror mode was used
        "white_type": args.white_type,  # Original white type
        "black_type": args.black_type,  # Original black type
        "white_tool_probabilities": args._white_tool_probs if hasattr(args, '_white_tool_probs') else None,
        "black_tool_probabilities": args._black_tool_probs if hasattr(args, '_black_tool_probs') else None,
        "include_history": args.include_history if hasattr(args, 'include_history') else False,
        "split": args.split,
        "num_queries": args.num_queries,
        "num_unique_positions": len(set(g.start_fen for g in games.values())),
        "max_moves": args.max_moves,
        "max_retries": args.max_retries if hasattr(args, 'max_retries') else None,
        "timeout": args.timeout if hasattr(args, 'timeout') else None,
        "provider": args.provider if hasattr(args, 'provider') else "openai",
        "seed": args.seed,
        "results": results,
        "per_trajectory_results": per_trajectory_results,
        "average_moves": avg_moves,
        "processing_time_seconds": processing_time,
        "config_name": config.get("config_name"),
        "is_improvement": is_improvement,
        "command": " ".join(sys.argv),
        "output_dir": str(output_folder)
    }

    if is_improvement:
        metadata_path = output_folder / "metadata.json"
    else:
        metadata_path = output_folder / f"{version}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save comprehensive token usage (per-game and aggregate; collect per-game token usage
    per_game_usage = {}
    for game_id, game in games.items():
        per_game_usage[f"game_{game_id}"] = {
            "start_fen": game.start_fen,
            "white_type": game.white_type,
            "black_type": game.black_type,
            "total_moves": game.move_count,
            "result": game.result,
            "token_usage": game.token_usage
        }

    # Create comprehensive token usage file
    comprehensive_token_usage = {
        "aggregate": token_usage,  # Total across all games
        "per_game": per_game_usage,  # Individual game usage
        "summary": {
            "total_games": len(games),
            "games_with_agent_white": sum(1 for g in games.values() if g.white_type == "agent"),
            "games_with_agent_black": sum(1 for g in games.values() if g.black_type == "agent"),
            "average_tokens_per_game": token_usage["total_tokens"] / len(games) if games else 0,
            "average_tokens_per_move": token_usage["total_tokens"] / sum(g.move_count for g in games.values()) if sum(g.move_count for g in games.values()) > 0 else 0
        }
    }

    if is_improvement:
        token_usage_path = output_folder / "token_usage.json"
    else:
        token_usage_path = output_folder / f"{version}_token_usage.json"
    with open(token_usage_path, 'w') as f:
        json.dump(comprehensive_token_usage, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Chess games complete!")
    logger.info(f"{'='*50}")
    logger.info(f"Total games played: {len(games)}")
    logger.info(f"Results by color: {results['white_wins']} white wins, {results['black_wins']} black wins, {results['draws']} draws")
    logger.info(f"Results by player type:")
    logger.info(f"  {args.white_type}: {white_type_wins} wins")
    logger.info(f"  {args.black_type}: {black_type_wins} wins")
    logger.info(f"  Draws: {results['draws']}")
    logger.info(f"Average game length: {avg_moves:.1f} moves")
    logger.info(f"Processing time: {processing_time:.1f}s")
    logger.info(f"Token usage: {token_usage['total_tokens']:,} total tokens")
    logger.info(f"Output saved to: {output_folder}")

def load_fen_positions(args):
    """Load FEN positions based on arguments.

    Returns:
        List of FEN strings to use as starting positions
    """
    if args.split:
        # Load from train or test split
        split_file = f"src/datasets/chess/data/{args.split}.jsonl"
        logger.info(f"Loading FEN positions from {args.split} split: {split_file}")

        fen_positions = []
        with open(split_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                fen_positions.append(data['fen'])

        # Limit number of queries if specified
        if args.num_queries:
            fen_positions = fen_positions[:args.num_queries]
            logger.info(f"Using first {args.num_queries} positions from {args.split} split")
        else:
            logger.info(f"Using all {len(fen_positions)} positions from {args.split} split")
        return fen_positions
    elif args.start_fen: # Use provided FEN positions
        logger.info(f"Using {len(args.start_fen)} provided FEN position(s)")
        return args.start_fen
    else: # Use standard starting position
        logger.info("Using standard chess starting position")
        return [chess.STARTING_FEN]

def main(args):
    """Main execution function."""
    random.seed(args.seed); logger.info(f"Random seed set to: {args.seed}")

    # Parse tool probabilities if specified
    if args.white_type == "probabilistic":
        try:
            args._white_tool_probs = parse_tool_probabilities(args.white_tool_probabilities)
            logger.info(f"White probabilistic player: {args._white_tool_probs}")
            # If mirror mode, set black probabilities too so probabilistic player works as either color
            if args.mirror:
                args._black_tool_probs = args._white_tool_probs
                logger.info(f"Mirror mode: also set black probabilistic player to same probabilities")
        except Exception as e:
            logger.error(f"Failed to parse white tool probabilities: {e}")
            sys.exit(1)

    if args.black_type == "probabilistic":
        try:
            args._black_tool_probs = parse_tool_probabilities(args.black_tool_probabilities)
            logger.info(f"Black probabilistic player: {args._black_tool_probs}")
            # If mirror mode, set white probabilities too so probabilistic player works as either color
            if args.mirror:
                args._white_tool_probs = args._black_tool_probs
                logger.info(f"Mirror mode: also set white probabilistic player to same probabilities")
        except Exception as e:
            logger.error(f"Failed to parse black tool probabilities: {e}")
            sys.exit(1)

    # Create session pool for efficient parallel tool execution; pool size matches parallel workers to ensure one session per thread
    cpu_count = os.cpu_count() or 8
    max_workers = getattr(args, 'parallel_workers', 8)

    # Create session pool
    session_pool = SessionPool(pool_size=max_workers)
    args._session_pool = session_pool  # Pass via args to other functions
    logger.info(f"Created session pool with {max_workers} workers (CPUs: {cpu_count})")

    if args.tool_config: # Mode 1: Tool config with per-problem tools
        logger.info(f"Loading tool config from {args.tool_config}")
        config_path = args.tool_config
        mode = "tool_config"

        with open(config_path, 'r') as f:
            full_config = json.load(f)

        # Extract tools from the first test (they should all be the same in practice), but we'll use the full config for metadata
        if "tests" in full_config and full_config["tests"]:
            tools = full_config["tests"][0].get("tools", [])
        else:
            tools = full_config.get("tools", [])

        config = full_config
        function_mapping = config.get("function_mapping", {})
    elif args.tool_set: # Mode 2: Tool set with shared tools for all problems
        logger.info(f"Loading tool set from {args.tool_set}")
        config_path = args.tool_set
        mode = "tool_set"

        with open(config_path, 'r') as f:
            config = json.load(f)

        tools = config.get("tools", [])
        function_mapping = config.get("function_mapping", {})

    fen_positions = load_fen_positions(args)
    if args.mirror and args.white_type == args.black_type:
        logger.warning(f"Mirror mode disabled: white and black player types are the same ({args.white_type})")
        args.mirror = False

    # Calculate total games
    games_per_position = args.num_trajectories * (2 if args.mirror else 1)
    total_games = len(fen_positions) * games_per_position

    # Detect if this is an improvement config (e.g., .../improvements/.../v1/config.json)
    is_improvement, improvement_base, improvement_version = detect_improvement_context(Path(config_path))

    # Build output directory path early so we can log it before starting
    if is_improvement:
        # Save directly in the version directory (e.g., .../v1/)
        output_folder = improvement_base / f"v{improvement_version}"
    else:
        config_name = parse_config_name(Path(config_path))
        output_folder = build_output_folder(args, config_name, mode)

    logger.info(f"Starting chess trajectory collection")
    logger.info(f"Mode: {mode}")
    logger.info(f"Config: {config.get('config_name', 'unnamed')}")
    logger.info(f"Config path: {config_path}")
    logger.info(f"White player: {args.white_type}")
    logger.info(f"Black player: {args.black_type}")
    logger.info(f"Mirror mode: {args.mirror}")
    logger.info(f"FEN positions to play: {len(fen_positions)}")
    logger.info(f"Trajectories per position: {args.num_trajectories}")
    if args.mirror:
        logger.info(f"  (x2 for mirror mode)")
    logger.info(f"Total games: {total_games}")
    logger.info(f"Output directory: {output_folder}")

    start_time = time.time()

    # Initialize games for all FEN positions
    games = {}
    game_id = 0
    for fen in fen_positions:
        for traj_idx in range(args.num_trajectories):
            # Original configuration
            games[game_id] = GameState(game_id, fen, args.white_type, args.black_type)
            game_id += 1

            if args.mirror: # Mirror configuration (swapped roles) if enabled
                games[game_id] = GameState(game_id, fen, args.black_type, args.white_type)
                game_id += 1

    # Play games in batches
    completed_games, token_usage = play_games_batch(games, tools, args, function_mapping)
    processing_time = time.time() - start_time
    save_trajectories(completed_games, args, config, processing_time, token_usage, mode, config_path, output_folder,
                      is_improvement=is_improvement, improvement_version=improvement_version)

    # Clean up session pool
    session_pool.cleanup()
    cleanup_engines()  # Global cleanup for any stray engines
    logger.info("Session pool and engines cleaned up")

if __name__ == "__main__":
    parser = create_chess_parser()
    args = parser.parse_args()
    main(args)