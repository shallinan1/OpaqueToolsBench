"""
Token usage tracking utilities.

Simple functions for aggregating and saving token usage from API responses.

Usage:
    After getting responses from openai_parallel_generate, aggregate token usage:

    ```python
    from src.generation_utils.openai_parallel_generate import openai_parallel_generate
    from src.generation_utils.token_tracker import aggregate_token_usage_from_responses, save_token_usage

    # Get responses from parallel API calls
    responses = await openai_parallel_generate(api_requests, ...)

    # Aggregate token usage from all responses
    token_usage = aggregate_token_usage_from_responses(responses, model="gpt-4")

    # Save to file
    save_token_usage(token_usage, Path("output/token_usage.json"))
    ```

    The responses from openai_parallel_generate are in format: [request, response, metadata]
    where response contains a "usage" field with prompt_tokens, completion_tokens, and total_tokens.

    To combine usage from multiple runs:

    ```python
    from src.generation_utils.token_tracker import load_token_usage, combine_token_usage

    # Load multiple token usage files
    usage1 = load_token_usage(Path("run1/token_usage.json"))
    usage2 = load_token_usage(Path("run2/token_usage.json"))

    # Combine them
    combined = combine_token_usage([usage1, usage2])
    ```
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


def aggregate_token_usage_from_responses(responses: List, model: str = None) -> Dict[str, Any]:
    """
    Aggregate token usage from a list of API responses returned by openai_parallel_generate.

    Args:
        responses: List of responses from openai_parallel_generate (format: [request, response, metadata] or [request, response])
        model: Optional model name for tracking

    Returns:
        Dictionary with aggregated token usage
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for item in responses:
        # Response format is [request, response] or [request, response, metadata]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            response = item[1]  # The actual API response is the second element

            if isinstance(response, dict) and "usage" in response:
                usage = response["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += usage.get("total_tokens", prompt_tokens + completion_tokens)

    result = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "num_requests": len(responses),
        "timestamp": datetime.utcnow().isoformat()
    }

    if model:
        result["model"] = model

    return result


def save_token_usage(usage_data: Dict[str, Any], output_path: Path) -> None:
    """
    Save token usage data to a JSON file.

    Args:
        usage_data: Token usage statistics
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(usage_data, f, indent=2)

    logger.info(f"Token usage saved to {output_path}: {usage_data.get('total_tokens', 0):,} total tokens")


def load_token_usage(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load token usage data from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Token usage data or None if file doesn't exist
    """
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        return json.load(f)


def combine_token_usage(usage_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple token usage dictionaries into a single summary.

    Args:
        usage_list: List of token usage dictionaries

    Returns:
        Combined token usage summary
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_requests = 0
    models_used = set()

    for usage in usage_list:
        if usage:
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
            total_requests += usage.get("num_requests", 0)
            if "model" in usage:
                models_used.add(usage["model"])

    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "num_requests": total_requests,
        "models": list(models_used),
        "timestamp": datetime.utcnow().isoformat()
    }