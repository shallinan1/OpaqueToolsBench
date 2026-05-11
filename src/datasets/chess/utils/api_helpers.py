"""
API request/response helpers for chess trajectory collection.

This module provides functions for creating API requests and processing
batch responses for chess move generation.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from src.datasets.chess.prompts import CHESS_AGENT_PROMPTS
from src.datasets.chess.utils.session_management import GameState
from src.datasets.chess.utils.tool_execution import execute_tools_parallel

logger = logging.getLogger(__name__)


def create_api_request(game: GameState, tools: List[Dict], args, color: str) -> Dict:
    """Create an API request for a game's current move.

    Args:
        game: The game state
        tools: Available tools in OpenAI format
        args: Command line arguments
        color: "white" or "black"

    Returns:
        API request dictionary
    """
    fen = game.board.fen()
    system_content = CHESS_AGENT_PROMPTS[args.prompt_key]
    user_content = f"Current position (FEN): {fen}\n"

    if args.include_history and game.moves:
        # Build condensed move history with tool usage
        move_history = []
        for move_data in game.moves:
            move_str = f"{move_data['move']}"
            # Find if there was a tool call for this move
            tool_used = None
            for tool_call in game.tool_calls:
                if tool_call['move_number'] == move_data['move_number'] and tool_call['color'] == move_data['color']:
                    tool_used = tool_call['tool']
                    break
            if tool_used:
                move_str += f" ({tool_used})"
            move_history.append(move_str)

        # Extract starting move number from FEN
        fen_parts = game.start_fen.split()
        starting_move_num = int(fen_parts[-1]) if len(fen_parts) >= 6 else 1
        starting_color = 'white' if fen_parts[1] == 'w' else 'black' if len(fen_parts) >= 2 else 'white'

        # Format as standard chess notation (1. e4 e5 2. Nf3 Nc6 ...)
        formatted_moves = []
        current_move_num = starting_move_num
        i = 0

        # If we start mid-move (Black to play), handle first Black move specially
        if starting_color == 'black' and game.moves and game.moves[0]['color'] == 'black':
            formatted_moves.append(f"{current_move_num}. ... {move_history[0]}")
            i = 1
            current_move_num += 1  # After Black completes the move, increment

        # Process remaining moves in White-Black pairs
        while i < len(game.moves):
            # Collect White move (if present)
            white_move = None
            if i < len(game.moves) and game.moves[i]['color'] == 'white':
                white_move = move_history[i]
                i += 1

            # Collect Black move (if present)
            black_move = None
            if i < len(game.moves) and game.moves[i]['color'] == 'black':
                black_move = move_history[i]
                i += 1

            # Format the move pair
            if white_move and black_move:
                formatted_moves.append(f"{current_move_num}. {white_move} {black_move}")
                current_move_num += 1  # Increment only after both moves
            elif white_move:
                formatted_moves.append(f"{current_move_num}. {white_move}")
                # Don't increment yet - waiting for Black's move
            elif black_move:
                formatted_moves.append(f"{current_move_num}. ... {black_move}")
                current_move_num += 1  # Black completed the move pair

        user_content += f"\nMove history: {' '.join(formatted_moves)}\n"

    user_content += f"\nYou are playing as {color}. It's your turn. Please decide an appropriate tool to make your move."

    # Create messages for current move
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    # Build API request
    api_request = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": args.tool_choice,
        "metadata": {"game_id": game.game_id, "move_number": game.move_count + 1, "color": color}
    }

    # Handle model-specific parameters
    if args.model.startswith('o') or 'gpt-5' in args.model:
        if args.max_tokens is not None:
            api_request["max_completion_tokens"] = args.max_tokens
        # Add reasoning_effort for reasoning models (o1, o3, gpt-5)
        if hasattr(args, 'reasoning_effort'):
            api_request["reasoning_effort"] = args.reasoning_effort
    else:
        if args.temperature is not None:
            api_request["temperature"] = args.temperature
        if args.top_p is not None:
            api_request["top_p"] = args.top_p
        if args.max_tokens is not None:
            api_request["max_tokens"] = args.max_tokens

    return api_request


def process_batch_responses(api_responses: List, games: Dict[int, GameState],
                           args, function_mapping: Optional[Dict] = None):
    """Process a batch of API responses, handling tool calls if present.

    Args:
        api_responses: List of API responses from parallel generation
        games: Dictionary of game_id -> GameState
        args: Command line arguments
        function_mapping: Optional tool name mapping
    """
    # First pass: collect all tool calls and save response data
    tool_calls_to_execute = []
    game_metadata = {}  # Store metadata for each game

    with tqdm(total=len(api_responses), desc="   Processing API responses", leave=False, unit="game") as pbar:
        for response_tuple in api_responses:
            # response_tuple format: [request, response, metadata]
            metadata = response_tuple[2]
            response = response_tuple[1]
            game_id = metadata["game_id"]
            color = metadata["color"]

            game = games[game_id]
            message = response["choices"][0]["message"]

            # Save the full API response for this move
            game.full_responses.append({
                "move_number": game.move_count + 1,
                "color": color,
                "request": response_tuple[0],  # Include the request that was sent
                "response": response,  # Full API response
                "metadata": metadata
            })

            if "usage" in response:
                usage = response["usage"]
                for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    game.token_usage[key] += usage.get(key, 0)

            # Store metadata for later use
            game_metadata[game_id] = {
                "color": color,
                "message": message,
                "game": game
            }

            if message.get("tool_calls"):
                # Only process first tool call (chess only needs one move)
                tool_call = message["tool_calls"][0]
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]

                # Parse arguments if string
                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except json.JSONDecodeError:
                        pass

                # Add to parallel execution list
                tool_calls_to_execute.append((game_id, func_name, game.board, function_mapping))

            pbar.update(1)

    # Execute all tool calls in parallel using session pool
    max_workers = getattr(args, 'parallel_workers', 8)
    session_pool = getattr(args, '_session_pool')  # Session pool passed via args
    tool_results = execute_tools_parallel(tool_calls_to_execute, session_pool, max_workers)

    # Second pass: apply results
    with tqdm(total=len(game_metadata), desc="   Applying moves", leave=False, unit="game") as pbar:
        for game_id, metadata in game_metadata.items():
            game = metadata["game"]
            color = metadata["color"]
            message = metadata["message"]

            if game_id in tool_results:
                # Tool was called - apply the result
                func_name = None
                for tc in tool_calls_to_execute:
                    if tc[0] == game_id:
                        func_name = tc[1]
                        break

                move = tool_results[game_id]
                game.tool_calls.append({
                    "move_number": game.move_count + 1,
                    "color": color,
                    "tool": func_name,
                    "result": move
                })

                # Apply the move if valid
                if move and not move.startswith("Error"):
                    try:
                        move_obj = game.board.parse_san(move)
                        game.board.push(move_obj)
                        game.move_count += 1

                        # Record the move
                        game.moves.append({
                            "move_number": game.move_count,
                            "color": color,
                            "move": move,
                            "fen_after": game.board.fen()
                        })
                    except Exception as e:
                        logger.error(f"Game {game_id}: Failed to apply tool move {move}: {e}")
                        game.is_complete = True
                        game.result = "0-1" if color == "white" else "1-0"
                        game.termination = f"Invalid move by {color}"
                else:
                    logger.error(f"Game {game_id}: Tool {func_name} failed: {move}")
                    game.is_complete = True
                    game.result = "0-1" if color == "white" else "1-0"
                    game.termination = f"Tool error by {color}"
            else:
                # No tool call - this is an error
                logger.error(f"Game {game_id}: No tool call by {color}")
                game.is_complete = True
                game.result = "0-1" if color == "white" else "1-0"
                game.termination = f"No tool call by {color}"

            pbar.update(1)
