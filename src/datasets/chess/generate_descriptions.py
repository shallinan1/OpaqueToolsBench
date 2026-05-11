#!/usr/bin/env python3
"""
Generate improved tool descriptions from chess evaluation results.

This is Step 3 of the iterative improvement pipeline. It analyzes scored game
trajectories to identify patterns in tool usage and generate improved descriptions.

How it works:
1. Loads scored trajectories (v0_scored.json) with board evaluations
2. Batches trajectories (default: 10 per batch) for parallel LLM analysis
3. Each batch is sent to the LLM with the analysis prompt
4. LLM identifies tool usage patterns, strengths, and weaknesses
5. Saves all LLM responses to llm_responses.json

Output format (llm_responses.json):
    {
        "generation_metadata": {...},
        "llm_responses": [
            {"batch_idx": 0, "num_trajectories_batch": 10, "content": "..."},
            {"batch_idx": 1, "num_trajectories_batch": 10, "content": "..."}
        ]
    }

The LLM response format (per the "detailed" prompt):
    **Tool: [name]**
    Observed patterns: [behaviors with trajectory evidence]
    Distinguishing characteristics: [what makes this tool different]
    Updated description: [new description]
    Reasoning: [justification]

Note: This script produces intermediate LLM responses, NOT the final config.
Use synthesize_descriptions.py to combine responses into final descriptions.

Key parameters:
    --num-trajectories-batch: Number of games per LLM request (default: 10)
        - More games = more patterns but longer context
        - Fewer games = more requests but diverse perspectives

    --show-agent-values: Include centipawn evaluations after agent moves
        - Helps LLM understand move quality
        - Shows whether agent's tool choice improved/worsened position

    --prompt-key: Analysis prompt template (default: "detailed")
        - "detailed": Comprehensive analysis with trajectory evidence

Example:
    python -m src.datasets.chess.generate_descriptions \\
        --result-dir runs/chess/.../v0_scored.json \\
        --model gpt-5 \\
        --prompt-key detailed \\
        --num-trajectories-batch 5 \\
        --show-agent-values
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()

from src.generation_utils.openai_parallel_generate import openai_parallel_generate, requests_url_dict, default_request_url
from src.generation_utils.rate_limits import get_rate_limit, get_token_limit
from src.generation_utils.token_tracker import aggregate_token_usage_from_responses, save_token_usage, combine_token_usage

from src.datasets.chess.prompts import CHESS_TOOL_DESCRIPTION_PROMPTS
from src.datasets.chess.utils.path_utils import get_next_version, create_editing_dirname, get_base_run_path
import random

def load_scored_results(result_file: Path) -> List[Dict]:
    """Load scored results from file."""
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, 'r') as f:
        data = json.load(f)

    # The scored file has a summary and trajectories structure
    if isinstance(data, dict) and 'trajectories' in data:
        return data['trajectories']
    else:
        # Fallback to returning the data as-is if it's already a list
        return data if isinstance(data, list) else []

def load_tool_config(result_dir: Path) -> Tuple[Dict, Path]:
    """Load the tool config that was used for this evaluation run.

    Returns:
        (config_dict, config_source_path)
    """
    # Check for metadata file to find config source
    metadata_files = list(result_dir.glob("*metadata.json"))
    if not metadata_files:
        raise ValueError(f"No metadata files found in {result_dir}")

    # Use the latest metadata file
    metadata_file = max(metadata_files, key=lambda f: f.stat().st_mtime)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    config_source = metadata.get("config_source")
    if not config_source:
        raise ValueError("No config_source found in metadata")

    config_path = Path(config_source)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f), config_path

def format_trajectories_for_prompt(trajectories: List[Dict], show_agent_values: bool = False) -> str:
    """Format game trajectories for inclusion in the prompt.

    Args:
        trajectories: List of game dictionaries with moves and tool calls
        show_agent_values: If True, show board values after agent moves

    Returns:
        Formatted string representation of trajectories
    """
    # Check if board values exist when show_agent_values is requested
    if show_agent_values:
        has_board_values = False
        for game in trajectories:
            for move in game.get("moves", []):
                if "board_value_cp" in move:
                    has_board_values = True
                    break
            if has_board_values:
                break

        if not has_board_values:
            raise ValueError(
                "--show-agent-values was set but no board_value_cp found in trajectories. "
                "Did you run evaluate.py first? Use v0_scored.json instead of v0_trajectories.json."
            )

    formatted_trajectories = []

    for idx, game in enumerate(trajectories, 1):
        trajectory_lines = [f"### Trajectory {idx}:"]

        # Add which side the agent is playing
        white_type = game.get("white_type", "unknown")
        black_type = game.get("black_type", "unknown")
        if white_type == "agent":
            trajectory_lines.append(f"Agent plays: White")
        elif black_type == "agent":
            trajectory_lines.append(f"Agent plays: Black")
            
        # Only show result if it's checkmate
        if "termination" in game and game.get("termination") == "checkmate":
            trajectory_lines.append(f"Result: {game['result']} (checkmate)")

        # Add initial board value
        if "initial_board_value_cp" in game:
            trajectory_lines.append(f"Initial board value: {game['initial_board_value_cp']} cp")

        # Add final board value (calculated from last move if available)
        moves = game.get("moves", [])
        if moves and "board_value_cp" in moves[-1]:
            trajectory_lines.append(f"Final board value: {moves[-1]['board_value_cp']} cp")

        # Format moves and tool calls
        moves = game.get("moves", [])
        tool_calls = game.get("tool_calls", [])

        # Group tool calls by move number for easier analysis
        tool_calls_by_move = defaultdict(list)
        for call in tool_calls:
            move_num = call.get("move_number", 0)
            tool_calls_by_move[move_num].append(call)

        trajectory_lines.append("\nMoves and Tool Usage:")

        # Add starting position
        if "start_fen" in game:
            trajectory_lines.append(f"\nStarting position (FEN): {game['start_fen']}")

        for i, move_data in enumerate(moves[:20], 1):  # Limit to first 20 moves for brevity
            move = move_data.get("move", "")
            color = move_data.get("color", "")
            fen_after = move_data.get("fen_after", "")

            # Check if this was an agent move
            player_type = game.get("white_type" if color == "white" else "black_type", "")

            if player_type == "agent":
                # For agent moves, show which tool was called
                trajectory_lines.append(f"\nMove {i} ({color}): {move}")
                if fen_after:
                    trajectory_lines.append(f"  Position after (FEN): {fen_after}")
                if i in tool_calls_by_move:
                    for call in tool_calls_by_move[i]:
                        tool_name = call.get("tool", "unknown")
                        trajectory_lines.append(f"  Tool called by agent: {tool_name}")
                # Optionally show board values before and after agent move
                if show_agent_values and "board_value_cp" in move_data:
                    # "Before" = previous move's board_value_cp, or initial_board_value_cp for first move
                    if i == 1:  # First move (i is 1-indexed)
                        before_cp = game.get("initial_board_value_cp")
                    else:
                        before_cp = moves[i - 2].get("board_value_cp")  # i-2 because i is 1-indexed
                    if before_cp is not None:
                        trajectory_lines.append(f"  Board value before move: {before_cp} cp")
                    trajectory_lines.append(f"  Board value after move: {move_data['board_value_cp']} cp")
            else:
                # For non-agent moves, just show the move (don't reveal which tool)
                trajectory_lines.append(f"\nMove {i} ({color}): {move}")
                if fen_after:
                    trajectory_lines.append(f"  Position after (FEN): {fen_after}")

        if len(moves) > 20:
            trajectory_lines.append(f"\n... ({len(moves) - 20} more moves)")

        formatted_trajectories.append("\n".join(trajectory_lines))

    return "\n\n".join(formatted_trajectories)


def analyze_tool_usage(scored_games: List[Dict]) -> Dict[str, Dict]:
    """Analyze tool usage patterns from scored games.

    Returns a dictionary mapping tool names to usage statistics and patterns.
    """
    tool_usage = defaultdict(lambda: {
        "total_calls": 0,
        "successful_moves": [],
        "failed_moves": [],
        "move_patterns": [],
        "game_phases": defaultdict(int),  # opening, middlegame, endgame
        "position_types": defaultdict(int),  # tactical, positional, etc.
    })

    for game in scored_games:
        # Get tool calls from game
        tool_calls = game.get("tool_calls", [])
        moves = game.get("moves", [])

        for call in tool_calls:
            tool_name = call.get("tool")
            if not tool_name:
                continue

            tool_usage[tool_name]["total_calls"] += 1

            # Find corresponding move
            move_number = call.get("move_number")
            move_result = call.get("result")

            # Determine game phase based on move number
            total_moves = len(moves)
            if move_number <= total_moves * 0.3:
                phase = "opening"
            elif move_number <= total_moves * 0.7:
                phase = "middlegame"
            else:
                phase = "endgame"

            tool_usage[tool_name]["game_phases"][phase] += 1

            # Store move patterns
            if move_result:
                tool_usage[tool_name]["move_patterns"].append({
                    "move": move_result,
                    "move_number": move_number,
                    "phase": phase
                })

    return dict(tool_usage)

def prepare_description_requests(
    tool_usage: Dict[str, Dict],
    original_config: Dict,
    prompt_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    trajectory_batches: Optional[List[List[Dict]]] = None,
    show_agent_values: bool = False,
    reasoning_effort: str = None
) -> List[Dict]:
    """Prepare API requests for generating tool descriptions.

    Args:
        tool_usage: Analysis of tool usage patterns
        original_config: Original config with tool definitions
        prompt_key: Which prompt template to use
        model: Model to use for generation
        temperature: Temperature for generation
        max_tokens: Max tokens for generation
        trajectory_batches: List of trajectory batches, each batch will become one API request
        reasoning_effort: Reasoning effort for reasoning models (gpt-5, o-series)
    """

    if prompt_key not in CHESS_TOOL_DESCRIPTION_PROMPTS:
        raise ValueError(f"Unknown prompt key: {prompt_key}")

    prompt_config = CHESS_TOOL_DESCRIPTION_PROMPTS[prompt_key]

    # Get tools from config
    tools = original_config.get("tools", [])

    # Create API requests - one for each batch of trajectories
    api_requests = []

    if not trajectory_batches:
        trajectory_batches = [[]]  # Create one empty batch if none provided

    for batch_idx, trajectories_data in enumerate(trajectory_batches):
        # Format trajectories for this batch
        if trajectories_data:
            trajectories_text = format_trajectories_for_prompt(trajectories_data, show_agent_values)
        else:
            trajectories_text = "No trajectories provided"


        # Format current tool definitions
        tools_text = []
        for tool in tools:
            tools_text.append(f"""
Tool Name: {tool.get('name', '')}
Current Description: {tool.get('description', '')}""")

        # Build prompt - split into system and user messages
        system_prompt = prompt_config["pre"]

        user_prompt = "## Current Tool Definitions:\n"
        user_prompt += "\n".join(tools_text)
        user_prompt += "\n\n## Game Trajectories:\n"
        user_prompt += trajectories_text
        user_prompt += "\n\n---\n\n"
        user_prompt += "Now follow the instructions and generate improved descriptions for each tool based on the trajectories above."

        # Create API request with system and user messages
        api_request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "metadata": {
                "tool_names": [tool.get("name", "") for tool in tools],
                "batch_idx": batch_idx,
                "num_trajectories_batch": len(trajectories_data),
                "show_agent_values": show_agent_values
            }
        }

        # Handle model-specific parameters
        if model.startswith('o') or 'gpt-5' in model:  # o-series and gpt-5 models
            api_request["max_completion_tokens"] = max_tokens
            if reasoning_effort is not None:
                api_request["reasoning_effort"] = reasoning_effort
        else:
            api_request["temperature"] = temperature
            api_request["max_tokens"] = max_tokens

        api_requests.append(api_request)

    return api_requests

def process_description_responses(responses: List) -> List[Dict[str, str]]:
    """Process API responses and extract tool descriptions.

    Returns a list of dictionaries, each mapping tool names to descriptions from one response.
    """
    all_descriptions = []

    for response_data in responses:
        try:
            request = response_data[0]
            response = response_data[1]
            metadata = response_data[2]

            # Extract descriptions from response
            content = response["choices"][0]["message"]["content"]

            # Parse the response to extract descriptions
            lines = content.split("\n")
            current_tool = None
            current_description = []
            descriptions = {}  # Initialize for this response

            for line in lines:
                if line.startswith("TOOL:"):
                    if current_tool and current_description:
                        descriptions[current_tool] = " ".join(current_description).strip()
                    current_tool = line.replace("TOOL:", "").strip()
                    current_description = []
                elif line.startswith("DESCRIPTION:"):
                    current_description.append(line.replace("DESCRIPTION:", "").strip())
                elif current_description and line.strip():
                    current_description.append(line.strip())

            # Don't forget the last one
            if current_tool and current_description:
                descriptions[current_tool] = " ".join(current_description).strip()

            all_descriptions.append(descriptions)

        except Exception as e:
            logger.error(f"Error processing response: {e}")

    return all_descriptions

def save_generation_results(
    result_dir: Path,
    model: str,
    prompt_key: str,
    temperature: float,
    max_tokens: int,
    raw_responses: Optional[List[Dict]] = None,
    show_agent_values: bool = False,
    token_usage: Optional[Dict] = None,
    reasoning_effort: Optional[str] = None
) -> Path:
    """Save generation results in improvements directory.

    Returns:
        Path to the saved results directory
    """
    # Extract base path from result_dir
    # For base runs: runs/chess/.../vs_{black}/v0_scored.json → base is the vs_{black}/ dir
    # For improvement runs: .../improvements/.../v1/scored.json → base is the vs_{black}/ dir (above improvements/)
    if result_dir.is_file():
        parent = result_dir.parent
    else:
        parent = result_dir

    # Check if this path is inside an improvements directory
    base_from_improvement = get_base_run_path(parent)
    if base_from_improvement is not None:
        base_run_path = base_from_improvement
    else:
        base_run_path = parent

    # Create improvements directory structure
    editing_dirname = create_editing_dirname(model, temperature, prompt_key, max_tokens, show_agent_values, reasoning_effort)
    improvements_dir = base_run_path / "improvements" / editing_dirname

    # Get next version (improvements start at v1)
    next_version = get_next_version(improvements_dir, is_improvement=True)
    version_dir = improvements_dir / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Version: v{next_version}")

    # Save generation metadata
    generation_metadata = {
        "model": model,
        "temperature": temperature,
        "prompt_key": prompt_key,
        "max_tokens": max_tokens,
        "source_result_dir": str(parent),
        "improvement_version": next_version,
        "generated_timestamp": datetime.utcnow().isoformat() + "Z",
        "show_agent_values": show_agent_values
    }

    metadata_file = version_dir / "generation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(generation_metadata, f, indent=2)

    # Save reasoning if raw responses are provided
    if raw_responses:
        # Extract all LLM responses
        llm_responses = []
        for response_data in raw_responses:
            try:
                response = response_data[1]
                llm_responses.append({
                    "batch_idx": response_data[2].get("batch_idx", 0),
                    "num_trajectories_batch": response_data[2].get("num_trajectories_batch", 0),
                    "content": response["choices"][0]["message"]["content"]
                })
            except (IndexError, KeyError) as e:
                logger.warning(f"Could not extract response: {e}")

        reasoning_data = {
            "generation_metadata": generation_metadata,
            "llm_responses": llm_responses,
            "num_batches": len(llm_responses)
        }

        reasoning_file = version_dir / "llm_responses.json"
        with open(reasoning_file, 'w') as f:
            json.dump(reasoning_data, f, indent=2)

        logger.info(f"Saved LLM responses with {len(llm_responses)} responses: {reasoning_file}")

    # Save token usage if provided
    if token_usage:
        token_usage_path = version_dir / "token_usage.json"
        save_token_usage(token_usage, token_usage_path)
        logger.info(f"Saved token usage: {token_usage_path}")

    return version_dir

def main():
    """Main function for generating descriptions."""

    parser = argparse.ArgumentParser(description="Generate improved tool descriptions from chess results")
    parser.add_argument("--result-dir", type=str, required=True,
                       help="Directory containing scored.json from evaluation or path to scored.json file")
    parser.add_argument("--model", type=str, default="gpt-5",
                       help="Model to use for generating descriptions")
    parser.add_argument("--prompt-key", type=str, default="detailed",
                       help="Prompt template to use (detailed or multi_trajectory)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=8192,
                       help="Maximum tokens for generation")
    parser.add_argument("--num-trajectories-batch", type=int, default=10,
                       help="Number of trajectories to analyze together (for multi_trajectory prompt)")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                       choices=["minimal", "low", "medium", "high"],
                       help="Reasoning effort for editing model (for gpt-5, o-series)")
    parser.add_argument("--show-agent-values", action="store_true",
                       help="Show board values after agent moves in trajectories")

    args = parser.parse_args()

    result_path = Path(args.result_dir)
    # Determine if path is a file or directory
    if result_path.is_file():
        scored_file = result_path
        result_dir = result_path.parent
    else:
        result_dir = result_path
        # Look for scored.json file
        scored_file = result_dir / "v0_scored.json"
        if not scored_file.exists():
            scored_file = result_dir / "scored.json"
            if not scored_file.exists():
                logger.error(f"No scored.json or v0_scored.json found in {result_dir}")
                sys.exit(1)

    # Load scored results
    logger.info(f"Loading results from {scored_file}")
    scored_games = load_scored_results(scored_file)
    logger.info(f"Found {len(scored_games)} evaluated games")

    # Load original tool config
    original_config, config_source_path = load_tool_config(result_dir)
    logger.info(f"Loaded config from {config_source_path}")


    # Analyze tool usage patterns
    logger.info("Analyzing tool usage patterns...")
    tool_usage = analyze_tool_usage(scored_games)

    # Log usage summary
    for tool_name, usage in tool_usage.items():
        logger.info(f"Tool '{tool_name}': {usage['total_calls']} calls")


    # Use games in order
    games_to_use = scored_games

    # Create batches of trajectories
    trajectory_batches = []
    for i in range(0, len(games_to_use), args.num_trajectories_batch):
        batch = games_to_use[i:i + args.num_trajectories_batch]
        if len(batch) > 0:  # Only add non-empty batches
            trajectory_batches.append(batch)

    logger.info(f"Created {len(trajectory_batches)} batches of trajectories")
    logger.info(f"Batch sizes: {[len(batch) for batch in trajectory_batches]}")

    # Prepare API requests - one for each batch
    api_requests = prepare_description_requests(
        tool_usage, original_config, args.prompt_key,
        args.model, args.temperature, args.max_tokens,
        trajectory_batches=trajectory_batches,
        show_agent_values=args.show_agent_values,
        reasoning_effort=args.reasoning_effort
    )
    logger.info(f"Generated {len(api_requests)} description request(s)")

    # Get rate limits
    max_requests_per_minute = get_rate_limit(args.model)
    max_tokens_per_minute = get_token_limit(args.model)

    # Process requests in parallel
    logger.info(f"Sending requests to {args.model}...")
    responses = asyncio.run(openai_parallel_generate(
        api_requests,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        request_url=requests_url_dict.get(args.model, default_request_url),
    ))

    # Track token usage
    token_usage = aggregate_token_usage_from_responses(responses, model=args.model)
    logger.info(f"Token usage: {token_usage['total_tokens']:,} tokens")

    # Process responses
    all_descriptions = process_description_responses(responses)
    logger.info(f"Received {len(all_descriptions)} sets of descriptions")

    # Aggregate descriptions from multiple responses (for now, just use the first non-empty one for each tool)
    aggregated_descriptions = {}
    for desc_set in all_descriptions:
        for tool_name, description in desc_set.items():
            if tool_name not in aggregated_descriptions:
                aggregated_descriptions[tool_name] = description

    logger.info(f"Aggregated descriptions for {len(aggregated_descriptions)} tools")

    # Save generation results
    results_dir = save_generation_results(
        result_dir,
        args.model,
        args.prompt_key,
        args.temperature,
        args.max_tokens,
        responses,
        args.show_agent_values,
        token_usage,
        args.reasoning_effort
    )

    logger.info(f"\nGeneration complete. Results saved to: {results_dir}")
    logger.info(f"LLM responses saved in: {results_dir / 'llm_responses.json'}")

if __name__ == "__main__":
    main()