# Tool call tests for openai_parallel_generate across multiple models
# Run with: python3 -m src.generation_utils.tests.test_tool_calls
import asyncio
from src.generation_utils.openai_parallel_generate import openai_parallel_generate, requests_url_dict

test_models = ["gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-4.1", "gpt-4o-2024-08-06"]
outputs = {}

# Define simple tools for testing
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tip",
            "description": "Calculate tip amount for a bill",
            "parameters": {
                "type": "object",
                "properties": {
                    "bill_amount": {
                        "type": "number",
                        "description": "The total bill amount"
                    },
                    "tip_percentage": {
                        "type": "number",
                        "description": "The tip percentage (e.g., 15 for 15%)"
                    }
                },
                "required": ["bill_amount", "tip_percentage"]
            }
        }
    }
]

async def test_tool_selection():
    for model in test_models:
        print(f"Testing tool selection with model: {model}")
        
        # Request that should trigger weather tool with metadata for ordering
        requests = [
            {
                "model": model,
                "messages": [{"role": "user", "content": "What's the weather like in New York?"}],
                "tools": tools,
                "tool_choice": "auto",
                "metadata": {"request_id": 0}
            }
        ]
        
        results = await openai_parallel_generate(
            requests,
            max_requests_per_minute=10,
            max_tokens_per_minute=1000,
            request_url=requests_url_dict.get(model, requests_url_dict.get("gpt-3.5-turbo-0125"))
        )
        
        outputs[f"weather_{model}"] = results
        assert isinstance(results, list), f"Results should be a list for {model}"
        assert len(results) == 1, f"Should return one result for {model}"
        print(f"Weather test passed for {model}")

async def test_multiple_tool_options():
    for model in test_models:
        print(f"Testing multiple tool options with model: {model}")
        
        # Request that should trigger tip calculator with metadata for ordering
        requests = [
            {
                "model": model,
                "messages": [{"role": "user", "content": "Calculate a 20% tip for a $45.50 bill"}],
                "tools": tools,
                "tool_choice": "auto",
                "metadata": {"request_id": 0}
            }
        ]
        
        results = await openai_parallel_generate(
            requests,
            max_requests_per_minute=10,
            max_tokens_per_minute=1000,
            request_url=requests_url_dict.get(model, requests_url_dict.get("gpt-3.5-turbo-0125"))
        )
        
        outputs[f"tip_{model}"] = results
        assert isinstance(results, list), f"Results should be a list for {model}"
        assert len(results) == 1, f"Should return one result for {model}"
        print(f"Tip calculation test passed for {model}")

# Run tests
asyncio.run(test_tool_selection())
asyncio.run(test_multiple_tool_options())

print("\n" + "="*60)
print("TOOL CALL CONVERSATION RESULTS")
print("="*60)

for model in test_models:
    print(f"\n--- {model.upper()} ---")
    
    # Weather test conversation
    weather_result = outputs.get(f"weather_{model}")
    if weather_result and len(weather_result) > 0:
        print("\nWeather Test:")
        print("User: What's the weather like in New York?")
        # Results are in format [request, response] or [request, response, metadata]
        result_data = weather_result[0]
        if isinstance(result_data, (list, tuple)) and len(result_data) >= 2:
            response = result_data[1]  # The response is at index 1
            if isinstance(response, dict) and 'choices' in response:
                choice = response['choices'][0]
                if 'message' in choice:
                    message = choice['message']
                    if 'tool_calls' in message and message['tool_calls']:
                        tool_call = message['tool_calls'][0]
                        func_name = tool_call['function']['name']
                        func_args = tool_call['function']['arguments']
                        print(f"Assistant: [Called tool: {func_name} with arguments: {func_args}]")
                    elif 'content' in message and message['content']:
                        print(f"Assistant: {message['content']}")
                    else:
                        print(f"Assistant: [No content or tool calls]")
            else:
                print(f"Assistant: {response}")
        else:
            print(f"Assistant: [Unexpected format]")
    
    # Tip test conversation
    tip_result = outputs.get(f"tip_{model}")
    if tip_result and len(tip_result) > 0:
        print("\nTip Calculation Test:")
        print("User: Calculate a 20% tip for a $45.50 bill")
        # Results are in format [request, response] or [request, response, metadata]
        result_data = tip_result[0]
        if isinstance(result_data, (list, tuple)) and len(result_data) >= 2:
            response = result_data[1]  # The response is at index 1
            if isinstance(response, dict) and 'choices' in response:
                choice = response['choices'][0]
                if 'message' in choice:
                    message = choice['message']
                    if 'tool_calls' in message and message['tool_calls']:
                        tool_call = message['tool_calls'][0]
                        func_name = tool_call['function']['name']
                        func_args = tool_call['function']['arguments']
                        print(f"Assistant: [Called tool: {func_name} with arguments: {func_args}]")
                    elif 'content' in message and message['content']:
                        print(f"Assistant: {message['content']}")
                    else:
                        print(f"Assistant: [No content or tool calls]")
            else:
                print(f"Assistant: {response}")
        else:
            print(f"Assistant: [Unexpected format]")
    
    print("-" * 40)

