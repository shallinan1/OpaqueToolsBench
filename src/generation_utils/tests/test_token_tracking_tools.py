"""
Test token tracking with tool calling to understand token usage.

Example command: 
python3 -m src.generation_utils.tests.test_token_tracking_tools
"""

import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.generation_utils.openai_parallel_generate import openai_parallel_generate
from src.generation_utils.token_tracker import aggregate_token_usage_from_responses


def create_test_request():
    """Create a test request similar to BFCL but simpler."""

    # Simple function definitions - proper OpenAI format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "function_1",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "function_2",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]

    # Question from BFCL
    question = "I've decided to to cook pork for dinner. I need some inspiration on what to pair with it."

    # Create the API request
    request = {
        "model": "gpt-5",  # Use a cheaper model for testing
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided tools to answer questions. You must call at least one tool."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "tools": tools,
        "tool_choice": "required",  # Force tool use
        "max_completion_tokens": 10000  # Higher limit to see actual usage
    }

    return request


async def test_single_tool_call():
    """Test token usage for a single tool call."""

    request = create_test_request()

    # Make the API call
    print("Making API call with tool_choice='required'...")
    responses = await openai_parallel_generate(
        [request],
        max_requests_per_minute=100,
        max_tokens_per_minute=100000
    )

    # Check the response
    if responses and len(responses) > 0:
        response = responses[0][1]  # Get the actual response

        # Print the response
        print("\nResponse received:")
        print(json.dumps(response, indent=2))

        # Aggregate token usage
        token_usage = aggregate_token_usage_from_responses(responses, model=request["model"])

        print("\n" + "="*50)
        print("TOKEN USAGE SUMMARY:")
        print("="*50)
        print(f"Prompt tokens: {token_usage['prompt_tokens']:,}")
        print(f"Completion tokens: {token_usage['completion_tokens']:,}")
        print(f"Total tokens: {token_usage['total_tokens']:,}")
        print(f"Model: {token_usage.get('model', 'unknown')}")

        # Check if tool was actually called
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if "message" in choice and "tool_calls" in choice["message"]:
                tool_calls = choice["message"]["tool_calls"]
                print(f"\nTool calls made: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"  - {tc['function']['name']}")
            else:
                print("\nNo tool calls in response!")
    else:
        print("No response received!")


async def test_with_max_completion_tokens():
    """Test if max_completion_tokens is affecting the response."""

    # Test with different max_completion_tokens values
    for max_completion_tokens in [50, 100, 500, 8192]:
        print(f"\n{'='*60}")
        print(f"Testing with max_completion_tokens={max_completion_tokens}")
        print('='*60)

        request = create_test_request()
        request["max_completion_tokens"] = max_completion_tokens

        responses = await openai_parallel_generate(
            [request],
            max_requests_per_minute=100,
            max_tokens_per_minute=100000
        )

        if responses:
            token_usage = aggregate_token_usage_from_responses(responses, model=request["model"])
            print(f"Completion tokens used: {token_usage['completion_tokens']}")

            # Check actual response length
            response = responses[0][1]
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice:
                    # Get the actual content/tool calls
                    message = choice["message"]
                    content_len = len(json.dumps(message))
                    print(f"Response message size (chars): {content_len}")


if __name__ == "__main__":
    print("Testing token usage for tool calling...")
    print("="*60)

    # Run the single test
    asyncio.run(test_single_tool_call())

    print("\n\n" + "="*60)
    print("Testing effect of max_completion_tokens parameter...")
    print("="*60)
    asyncio.run(test_with_max_completion_tokens())