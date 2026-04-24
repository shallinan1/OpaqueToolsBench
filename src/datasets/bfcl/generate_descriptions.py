"""
Generate improved function descriptions from BFCL evaluation results.

This script analyzes function call results and generates better descriptions
based on observed usage patterns.

Example usage:
    python -m src.datasets.bfcl.generate_descriptions \
        --result-dir runs/bfcl/tool_observer/<config_name>/<hyperparam_dirname> \
        --model gpt-5 \
        --prompt-key basic_improved
"""

import sys
import json
import asyncio
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import shared utilities
from src.generation_utils.openai_parallel_generate import openai_parallel_generate, requests_url_dict, default_request_url
from src.generation_utils.rate_limits import get_rate_limit, get_token_limit
from src.generation_utils.token_tracker import aggregate_token_usage_from_responses, save_token_usage, combine_token_usage
from datetime import datetime

# Import BFCL components
from src.datasets.bfcl.prompts import BFCL_FUNCTION_DESCRIPTION_PROMPTS
from src.datasets.bfcl.utils.function_utils import (
    format_function_call,
    format_function_definition,
    functions_are_identical
)
from src.datasets.bfcl.utils.path_utils import (
    create_editing_dirname,
    get_next_version,
    parse_editing_dirname,
    get_base_run_path
)

def process_error_for_prompt(error_info, execution_results):
    """Process error information for the prompt.

    Uses the raw executed_result directly instead of BFCL processed errors.
    This shows what the model's code actually produced/errored with.
    """
    if not execution_results:
        return ("error", "No execution results available")

    # Check if execution_results contains error information
    # After our changes, execution_results always contains the raw outputs/errors
    results = []
    errors = []

    for result in execution_results:
        if isinstance(result, dict) and "error" in result:
            # This is a raw execution error
            errors.append(result["error"])
        else:
            # This is a successful execution output
            results.append(result)

    if errors:
        # Show the raw Python error(s)
        error_msg = "; ".join(errors)
        return ("error", error_msg)
    else:
        # Show the raw execution output(s)
        return ("result", results)

def load_converged_tests(result_dir: Path) -> set:
    """Load the set of test indices that have converged from previous iterations.

    A test is considered converged if its functions didn't change in the previous iteration.
    """
    converged_tests = set()

    # Check if this is an improvement directory
    if (result_dir / "config.json").exists():
        # Prefer convergence info from the current improvement directory
        convergence_file = result_dir / "converged_tests.json"
        if convergence_file.exists():
            with open(convergence_file, 'r') as f:
                data = json.load(f)
                converged_tests = set(data.get("converged_tests", []))
                logger.info(f"Loaded {len(converged_tests)} previously converged tests from {result_dir.name}")
                return converged_tests

    return converged_tests


def detect_newly_converged_tests(
    original_config: Dict,
    improved_config: Dict,
    previously_converged: set
) -> set:
    """Detect which tests have newly converged (functions unchanged) in this iteration.

    Returns the set of test indices that didn't change in this iteration.
    """
    newly_converged = set()

    original_tests = original_config.get("tests", [])
    improved_tests = improved_config.get("tests", [])

    for idx in range(min(len(original_tests), len(improved_tests))):
        # Skip if already converged
        if idx in previously_converged:
            continue

        # Check if functions are identical
        if functions_are_identical(
            original_tests[idx].get("tools", []),
            improved_tests[idx].get("tools", [])
        ):
            newly_converged.add(idx)
            logger.info(f"Test {idx} has converged (functions unchanged)")

    return newly_converged


def prepare_description_requests(
    evaluations: List[Dict],
    original_config: Dict,
    prompt_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    converged_tests: set = None,
    reasoning_effort: str = None
) -> List[Dict]:
    """Prepare API requests for generating function descriptions - one per problem.

    Args:
        evaluations: List of evaluation results
        original_config: Original config with function definitions
        prompt_key: Which prompt template to use
        model: Model to use for generation
        temperature: Temperature for generation
        max_tokens: Max tokens for generation
        converged_tests: Set of test indices that have converged and should be skipped
        reasoning_effort: Reasoning effort for reasoning models (gpt-5, o-series)
    """

    if prompt_key not in BFCL_FUNCTION_DESCRIPTION_PROMPTS:
        raise ValueError(f"Unknown prompt key: {prompt_key}")

    if converged_tests is None:
        converged_tests = set()

    prompt_config = BFCL_FUNCTION_DESCRIPTION_PROMPTS[prompt_key]

    # Build a map of test index to functions from config
    test_functions = {}
    for idx, test in enumerate(original_config.get("tests", [])):
        test_functions[idx] = test.get("tools", [])

    # Create one API request per evaluation/problem
    api_requests = []
    skipped_count = 0

    for eval_item in evaluations:
        # Get test index
        test_index = eval_item.get("index", 0)

        # Skip if this test has converged
        if test_index in converged_tests:
            logger.debug(f"Skipping test {test_index} (converged)")
            skipped_count += 1
            continue

        # Get function definitions from the original config based on index
        functions = test_functions.get(test_index, [])
        question = eval_item.get("question", "")
        
        # Note: We no longer use ground_truth_execution as we don't want to expose the expected answer
        
        # Get model's result from evaluation
        model_result = eval_item.get("model_result", "[]")
        if isinstance(model_result, str):
            try:
                model_calls = json.loads(model_result)
            except json.JSONDecodeError:
                model_calls = []
        else:
            model_calls = model_result
        
        # Get execution results and errors from evaluation
        execution_results = eval_item.get("executed_result", [])
        is_valid = eval_item.get("valid", False)
        error_info = eval_item.get("error", [])

        # Format execution output
        if is_valid:
            execution_output = f"SUCCESS: {execution_results}"
        else:
            # Process the error to determine if it's a wrong result or real error
            processed = process_error_for_prompt(error_info, execution_results)
            error_type, error_value = processed
            if error_type == "result":
                # This is just a wrong result - show the model output
                execution_output = f"Model output: {error_value}"
            else: # This is a real error
                execution_output = f"ERROR: {error_value}"

        
        # Format all function calls for this problem
        function_calls_str = ""
        for call in model_calls:
            if isinstance(call, dict):
                func_name = call.get("function", "")
                function_calls_str += format_function_call({func_name: json.dumps(call.get("args", {}))}) + "\n"
        
        if not function_calls_str:
            continue  # Skip if no function calls
        
        # Format all available functions for this problem
        functions_str = "\n\n".join([format_function_definition(func) for func in functions])
        
        # Build prompt for this problem
        prompt_parts = [prompt_config["pre"]]

        # Add function definitions
        prompt_parts[0] = prompt_parts[0].replace(
            "{available_functions}",
            functions_str
        )

        # Add the example for this problem
        middle_part = prompt_config["middle"].format(
            example_num=1,
            question=question,
            function_call=function_calls_str.strip(),
            function_output=execution_output
        )
        prompt_parts.append(middle_part)
        
        # Add post section
        prompt_parts.append(prompt_config["post"])
        
        full_prompt = "\n".join(prompt_parts)
        
        # Create API request for this problem
        api_request = {
            "model": model,
            "messages": [{"role": "user", "content": full_prompt}],
            "metadata": {
                "test_index": test_index,
                "functions": [func.get("name", "") for func in functions]
            }
        }
        
        # Handle model-specific parameters (same as run.py)
        if model.startswith('o') or 'gpt-5' in model:  # o-series and gpt-5 models
            # These models don't support temperature/top_p, only max_completion_tokens
            api_request["max_completion_tokens"] = max_tokens
            if reasoning_effort is not None:
                api_request["reasoning_effort"] = reasoning_effort
        else:
            # Regular models support temperature and max_tokens
            api_request["temperature"] = temperature
            api_request["max_tokens"] = max_tokens
        
        api_requests.append(api_request)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} converged tests (won't attempt to improve)")

    return api_requests


def process_description_responses(responses: List) -> Tuple[Dict[int, Dict[str, str]], List[Dict]]:
    """Process API responses and extract function descriptions mapped by test index.

    Returns:
        (descriptions_by_test, raw_responses_by_index)
    """

    # Map test_index -> {function_name -> description}
    descriptions_by_test = {}
    raw_responses_by_index = []

    for response_data in responses:
        try:
            request = response_data[0]
            response = response_data[1]
            metadata = response_data[2]
            
            test_index = metadata.get("test_index", -1)
            
            # Extract description from response
            content = response["choices"][0]["message"]["content"]
            
            # Parse the response to extract the descriptions for all functions
            lines = content.split("\n")
            current_function = None
            current_description = []
            
            test_descriptions = {}
            
            for line in lines:
                if line.startswith("FUNCTION:"):
                    if current_function and current_description:
                        test_descriptions[current_function] = " ".join(current_description).strip()
                    current_function = line.replace("FUNCTION:", "").strip()
                    current_description = []
                elif line.startswith("DESCRIPTION:"):
                    current_description.append(line.replace("DESCRIPTION:", "").strip())
                elif current_description and line.strip():
                    current_description.append(line.strip())
            
            # Don't forget the last one
            if current_function and current_description:
                test_descriptions[current_function] = " ".join(current_description).strip()
            
            # Store descriptions for this specific test
            if test_descriptions:
                descriptions_by_test[test_index] = test_descriptions

            # Store raw response indexed by test_index
            while len(raw_responses_by_index) <= test_index:
                raw_responses_by_index.append(None)
            raw_responses_by_index[test_index] = response

        except Exception as e:
            logger.error(f"Error processing response: {e}")

    return descriptions_by_test, raw_responses_by_index


def load_config_from_result_dir(result_dir: Path) -> Tuple[Dict, Path]:
    """Load the config that was used for this evaluation run.

    Returns:
        (config_dict, config_source_path)
    """
    # Check if this is a base run (v{N}_metadata.json) or improvement (metadata.json)
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


def save_improved_config(
    result_dir: Path,
    original_config: Dict,
    descriptions_by_test: Dict[int, Dict[str, str]],
    model: str,
    prompt_key: str,
    temperature: float,
    max_tokens: int,
    existing_editing_dir: Optional[str] = None,
    raw_responses: Optional[List[Dict]] = None,
    converged_tests: set = None,
    reasoning_effort: str = None
) -> Path:
    """Save improved config with new descriptions in improvements directory.

    Args:
        result_dir: Directory containing the results to improve from
        original_config: The original config dictionary
        descriptions_by_test: Descriptions by test index
        model: Model used for generation
        prompt_key: Prompt key used
        temperature: Temperature used
        max_tokens: Max tokens used
        existing_editing_dir: If provided, verify it matches current params
        raw_responses: Raw LLM responses for reasoning
        converged_tests: Set of test indices that have converged

    Returns:
        Path to the saved config file
    """
    # Get the base run path (remove any v{N} suffix if present)
    base_run_path = get_base_run_path(result_dir)
    if not base_run_path:
        # If we can't detect base run path, use result_dir itself
        base_run_path = result_dir

    # Create editing hyperparameter directory name
    editing_dirname = create_editing_dirname(model, temperature, prompt_key, max_tokens, reasoning_effort=reasoning_effort)

    # If existing_editing_dir provided, verify it matches
    if existing_editing_dir:
        if existing_editing_dir != editing_dirname:
            # Parse the existing directory to show what's different
            try:
                existing_params = parse_editing_dirname(existing_editing_dir)
                logger.error(f"Editing hyperparameters mismatch!")
                logger.error(f"Existing: {existing_editing_dir}")
                logger.error(f"Current:  {editing_dirname}")
                logger.error(f"Existing params: {existing_params}")
                logger.error(f"Current params: model={model}, temp={temperature}, prompt={prompt_key}, max_tokens={max_tokens}")
                raise ValueError("Cannot increment version with different editing hyperparameters")
            except Exception as e:
                logger.error(f"Failed to parse existing directory name: {e}")
                raise

    # Create improvements directory structure
    improvements_dir = base_run_path / "improvements" / editing_dirname

    # Get next version
    next_version = get_next_version(improvements_dir, is_improvement=True)
    version_dir = improvements_dir / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Create a copy of the original config
    improved_config = json.loads(json.dumps(original_config))  # Deep copy

    # Update descriptions in the config - use test index to match
    updated_count = 0
    for idx, test in enumerate(improved_config.get("tests", [])):
        if idx in descriptions_by_test:
            test_descriptions = descriptions_by_test[idx]
            logger.debug(f"Test {idx} has descriptions for: {list(test_descriptions.keys())}")
            for tool in test.get("tools", []):
                tool_name = tool.get("name", "")
                if tool_name in test_descriptions:
                    tool["description"] = test_descriptions[tool_name]
                    updated_count += 1
                    logger.debug(f"Updated {tool_name} in test {idx}")
        else:
            logger.debug(f"No descriptions found for test index {idx}")

    logger.info(f"Updated {updated_count} function descriptions")

    # Don't add "_improved" suffix anymore - the directory structure shows it's improved
    # Keep original config_name
    improved_config["generation_metadata"] = {
        "model": model,
        "prompt_key": prompt_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "source_result_dir": str(result_dir),
        "improvement_version": next_version,
        "generated_timestamp": datetime.utcnow().isoformat() + "Z"
    }

    # Save the config
    config_file = version_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(improved_config, f, indent=2)

    logger.info(f"Saved improved config: {config_file}")
    logger.info(f"Version: v{next_version}")

    # Detect newly converged tests and save convergence information
    if converged_tests is None:
        converged_tests = set()

    # Detect which tests newly converged (functions unchanged) in this iteration
    newly_converged = detect_newly_converged_tests(original_config, improved_config, converged_tests)

    # Combine all converged tests
    all_converged = converged_tests.union(newly_converged)

    if all_converged:
        # Save convergence information
        convergence_file = version_dir / "converged_tests.json"
        convergence_data = {
            "converged_tests": sorted(list(all_converged)),
            "newly_converged_this_iteration": sorted(list(newly_converged)),
            "previously_converged": sorted(list(converged_tests)),
            "total_tests": len(improved_config.get("tests", [])),
            "convergence_rate": len(all_converged) / len(improved_config.get("tests", [])) if improved_config.get("tests") else 0
        }
        with open(convergence_file, 'w') as f:
            json.dump(convergence_data, f, indent=2)

        logger.info(f"Convergence status: {len(all_converged)}/{len(improved_config.get('tests', []))} tests converged")
        if newly_converged:
            logger.info(f"Newly converged in this iteration: {sorted(list(newly_converged))}")

    # Also save generation metadata separately for easy tracking
    generation_metadata = {
        "editing_model": model,
        "editing_temperature": temperature,
        "editing_prompt_key": prompt_key,
        "editing_max_tokens": max_tokens,
        "source_result_dir": str(result_dir),
        "improvement_version": next_version,
        "num_functions_updated": updated_count,
        "generated_timestamp": datetime.utcnow().isoformat() + "Z"
    }

    metadata_file = version_dir / "generation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(generation_metadata, f, indent=2)

    # Save reasoning if raw responses are provided
    if raw_responses:
        reasoning_data = []
        for idx, test in enumerate(improved_config.get("tests", [])):
            test_reasoning = {
                "test_index": idx,
                "question": test.get("question", ""),
                "functions_updated": []
            }

            # Only process function updates if descriptions exist for this test
            if idx in descriptions_by_test:
                # Get the original descriptions for comparison
                original_test = original_config.get("tests", [])[idx] if idx < len(original_config.get("tests", [])) else {}

                for tool in test.get("tools", []):
                    tool_name = tool.get("name", "")
                    if tool_name in descriptions_by_test[idx]:
                        # Find original description
                        original_desc = ""
                        for orig_tool in original_test.get("tools", []):
                            if orig_tool.get("name", "") == tool_name:
                                original_desc = orig_tool.get("description", "")
                                break

                        function_reasoning = {
                            "function_name": tool_name,
                            "original_description": original_desc,
                            "new_description": descriptions_by_test[idx][tool_name],
                            "changed": original_desc != descriptions_by_test[idx][tool_name]
                        }
                        test_reasoning["functions_updated"].append(function_reasoning)

            # Add the raw LLM response if available (for ALL tests, regardless of updates)
            if idx < len(raw_responses) and raw_responses[idx] is not None:
                test_reasoning["llm_reasoning"] = raw_responses[idx].get("choices", [{}])[0].get("message", {}).get("content", "")

            reasoning_data.append(test_reasoning)

        # Save reasoning file
        reasoning_file = version_dir / "improvement_reasoning.json"
        with open(reasoning_file, 'w') as f:
            json.dump({
                "generation_metadata": generation_metadata,
                "prompt_key_used": prompt_key,
                "test_reasoning": reasoning_data
            }, f, indent=2)

        logger.info(f"Saved improvement reasoning: {reasoning_file}")

    return config_file


def main():
    """Main function for generating descriptions."""
    
    parser = argparse.ArgumentParser(description="Generate improved function descriptions from BFCL results")
    parser.add_argument("--result-dir", type=str, required=True,
                       help="Directory containing scored.json from evaluation")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06",
                       help="Model to use for generating descriptions")
    parser.add_argument("--prompt-key", type=str, default="basic_improved",
                       choices=list(BFCL_FUNCTION_DESCRIPTION_PROMPTS.keys()),
                       help="Prompt template to use")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=8192,
                       help="Maximum tokens for generation")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                       choices=["none", "minimal", "low", "medium", "high"],
                       help="Reasoning effort for reasoning models (gpt-5, o-series)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        logger.error(f"Result directory not found: {result_dir}")
        sys.exit(1)
    
    # Look for scored.json file (improvements) or v0_scored.json (base runs)
    scored_file = result_dir / "scored.json"
    if not scored_file.exists():
        # Must be a base run - look for v0_scored.json
        scored_file = result_dir / "v0_scored.json"
        if not scored_file.exists():
            logger.error(f"No scored.json or v0_scored.json found in {result_dir}")
            logger.info("Please run evaluation first: python -m src.datasets.bfcl.evaluate --result-dir <dir>")
            sys.exit(1)
    
    # Load scored results
    with open(scored_file, 'r') as f:
        scored_data = json.load(f)
    detailed_evaluations = scored_data.get("detailed_evaluations", [])
    
    if not detailed_evaluations:
        logger.error("No detailed evaluations found in scored.json")
        sys.exit(1)
    
    logger.info(f"Found {len(detailed_evaluations)} evaluated results")
    
    # Group by category
    evaluations_by_category = {}
    for eval_item in detailed_evaluations:
        category = eval_item.get("category", "unknown")
        if category not in evaluations_by_category:
            evaluations_by_category[category] = []
        evaluations_by_category[category].append(eval_item)
    
    # Map test_index -> {function_name -> description}
    all_descriptions_by_test = {}
    all_raw_responses = []  # Collect all raw responses

    # Load original config from metadata first
    try:
        original_config, config_source_path = load_config_from_result_dir(result_dir)
    except Exception as e:
        logger.error(f"Failed to load original config: {e}")
        logger.info("Cannot proceed without original config")
        return

    # Load converged tests from previous iterations
    converged_tests = load_converged_tests(result_dir)

    if converged_tests:
        logger.info(f"Found {len(converged_tests)} converged tests from previous iterations")
        logger.info(f"These tests will be skipped: {sorted(list(converged_tests))}")

    # Track token usage across all batches
    all_token_usage = []

    # Process each category
    for test_category, category_evaluations in evaluations_by_category.items():
        logger.info(f"\nProcessing {test_category} ({len(category_evaluations)} results)...")

        # Prepare API requests using the original config, skipping converged tests
        api_requests = prepare_description_requests(
            category_evaluations, original_config, args.prompt_key,
            args.model, args.temperature, args.max_tokens,
            converged_tests=converged_tests,
            reasoning_effort=args.reasoning_effort
        )
        
        if not api_requests:
            logger.warning(f"No requests generated for {test_category}")
            continue
        
        logger.info(f"Generated {len(api_requests)} description requests")
        
        # Get rate limits
        max_requests_per_minute = get_rate_limit(args.model)
        max_tokens_per_minute = get_token_limit(args.model)
                
        # Process requests in parallel
        responses = asyncio.run(openai_parallel_generate(
            api_requests,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            request_url=requests_url_dict.get(args.model, default_request_url),
        ))

        # Track token usage for this batch
        batch_token_usage = aggregate_token_usage_from_responses(responses, model=args.model)
        all_token_usage.append(batch_token_usage)
        logger.info(f"Batch token usage: {batch_token_usage['total_tokens']:,} tokens")

        # Process responses - returns {test_index: {func_name: description}} and raw responses
        category_descriptions_by_test, raw_responses = process_description_responses(responses)
        logger.info(f"Descriptions by test indices: {list(category_descriptions_by_test.keys())}")

        # Merge with all descriptions
        for test_idx, test_descs in category_descriptions_by_test.items():
            if test_idx not in all_descriptions_by_test:
                all_descriptions_by_test[test_idx] = {}
            all_descriptions_by_test[test_idx].update(test_descs)

        # Merge raw responses (ensure we have the right size)
        if len(all_raw_responses) < len(raw_responses):
            all_raw_responses.extend([None] * (len(raw_responses) - len(all_raw_responses)))
        for i, resp in enumerate(raw_responses):
            if resp is not None:
                all_raw_responses[i] = resp
        
        # Count functions for logging
        num_functions = sum(len(test_descs) for test_descs in category_descriptions_by_test.values())
        logger.info(f"Generated descriptions for {num_functions} functions across {len(category_descriptions_by_test)} tests")
    
    # Check if we're improving from an existing improvement
    existing_editing_dir = None
    if "improvements" in str(result_dir):
        # Extract the editing directory name from the path
        parts = result_dir.parts
        if "improvements" in parts:
            imp_idx = parts.index("improvements")
            if imp_idx + 1 < len(parts):
                existing_editing_dir = parts[imp_idx + 1]
                logger.info(f"Detected existing editing directory: {existing_editing_dir}")

    # Save improved config with new descriptions
    logger.info(f"About to save config with descriptions for tests: {list(all_descriptions_by_test.keys())}")
    config_file = save_improved_config(
        result_dir,
        original_config,
        all_descriptions_by_test,
        args.model,
        args.prompt_key,
        args.temperature,
        args.max_tokens,
        existing_editing_dir,
        all_raw_responses,  # Pass raw responses for reasoning
        converged_tests,  # Pass converged tests for tracking
        reasoning_effort=args.reasoning_effort
    )

    # Save aggregated token usage in the same directory as the config
    if all_token_usage:
        combined_usage = combine_token_usage(all_token_usage)
        token_usage_path = config_file.parent / "token_usage_descriptions.json"
        save_token_usage(combined_usage, token_usage_path)
        logger.info(f"Total token usage for description generation: {combined_usage['total_tokens']:,} tokens")

    # Count total functions with descriptions
    total_functions = sum(len(test_descs) for test_descs in all_descriptions_by_test.values())
    logger.info(f"\nTotal functions with descriptions: {total_functions} across {len(all_descriptions_by_test)} tests")

    # Check for complete convergence
    total_tests = len(original_config.get("tests", []))
    if converged_tests and len(converged_tests) == total_tests:
        logger.info("\n" + "="*60)
        logger.info("🎉 COMPLETE CONVERGENCE ACHIEVED!")
        logger.info(f"All {total_tests} tests have converged. No further improvements possible.")
        logger.info("Iterative improvement process is complete.")
        logger.info("="*60)
        # Exit with special code 2 to signal convergence to the parent process
        sys.exit(2)
    else:
        logger.info(f"Next step: Run with --config-source {config_file}")




if __name__ == "__main__":
    main()
