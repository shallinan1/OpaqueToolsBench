"""
Basic tests for together_parallel_generate

python3 -m src.generation_utils.tests.test_together_generations
"""
import asyncio
from src.generation_utils.togetherai_parallel_generate import together_parallel_generate, default_request_url

test_models = ["openai/gpt-oss-20b"]
outputs = {}

async def test_together_parallel_generate_basic():
    for model in test_models:
        print(f"Testing basic request with model: {model}")
        # Minimal request for current model with metadata for ordering
        requests = [
            {
                "model": model,
                "messages": [{"role": "user", "content": "What model are you?"}],
                "metadata": {"request_id": 0}
            }
        ]
        results = await together_parallel_generate(
            requests,
            max_requests_per_minute=10,
            max_tokens_per_minute=1000,
            request_url=default_request_url
        )
        outputs[f"basic_{model}"] = results
        assert isinstance(results, list), f"Results should be a list for {model}"
        assert len(results) == 1, f"Should return one result for {model}"
        print(f"Test 1 passed: Basic single request for {model}.")

async def test_together_parallel_generate_multiple():
    for model in test_models:
        print(f"Testing multiple requests with model: {model}")
        # Multiple requests with metadata for ordering
        requests = [
            {
                "model": model,
                "messages": [{"role": "user", "content": f"Say hello #{i}!"}],
                "metadata": {"request_id": i}
            }
            for i in range(3)
        ]
        results = await together_parallel_generate(
            requests,
            max_requests_per_minute=10,
            max_tokens_per_minute=1000,
            request_url=default_request_url
        )

        # Sort results by request_id to preserve order
        sorted_results = sorted(results, key=lambda x: x[2]["request_id"] if len(x) > 2 else 0)
        outputs[f"multiple_{model}"] = sorted_results

        assert isinstance(results, list), f"Results should be a list for {model}"
        assert len(results) == 3, f"Should return three results for {model}"
        print(f"Test 2 passed: Multiple requests for {model}.")

# Run tests
asyncio.run(test_together_parallel_generate_basic())
asyncio.run(test_together_parallel_generate_multiple())

print("\n" + "="*60)
print("CONVERSATION RESULTS")
print("="*60)

for model in test_models:
    print(f"\n--- {model.upper()} ---")

    # Basic test conversation
    basic_result = outputs.get(f"basic_{model}")
    if basic_result and len(basic_result) > 0:
        print("\nBasic Test:")
        print("User: What model are you?")
        # Results are in format [request, response] or [request, response, metadata]
        result_data = basic_result[0]
        if isinstance(result_data, (list, tuple)) and len(result_data) >= 2:
            response = result_data[1]  # The response is at index 1
            if isinstance(response, dict) and 'choices' in response:
                choice = response['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    assistant_msg = choice['message']['content']
                    print(f"Assistant: {assistant_msg}")
                else:
                    print(f"Assistant: [No content in message]")
            else:
                print(f"Assistant: {response}")
        else:
            print(f"Assistant: [Unexpected format]")

    # Multiple test conversations
    multiple_results = outputs.get(f"multiple_{model}")
    if multiple_results and len(multiple_results) > 0:
        print(f"\nMultiple Tests:")
        for i, result_data in enumerate(multiple_results):
            print(f"User: Say hello #{i}!")
            if isinstance(result_data, (list, tuple)) and len(result_data) >= 2:
                response = result_data[1]  # The response is at index 1
                if isinstance(response, dict) and 'choices' in response:
                    choice = response['choices'][0]
                    if 'message' in choice and 'content' in choice['message']:
                        assistant_msg = choice['message']['content']
                        print(f"Assistant: {assistant_msg}")
                    else:
                        print(f"Assistant: [No content in message]")
                else:
                    print(f"Assistant: {response}")
            else:
                print(f"Assistant: [Unexpected format]")

    print("-" * 40)

