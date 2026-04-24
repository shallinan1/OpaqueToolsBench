"""
BFCL evaluation script.
Evaluates function call results against ground truth.

Example usage:
    python -m src.datasets.bfcl.evaluate \
        --result-dir runs/bfcl/tool_observer/<config_name>/<hyperparam_dirname>
"""

import os
import sys
import json
import glob
import argparse
import logging
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.datasets.bfcl.utils.function_utils import functions_are_identical
from src.datasets.bfcl.utils.path_utils import get_base_run_path
from src.datasets.bfcl.enhanced_metrics import (
    evaluate_enhanced_metrics,
    aggregate_enhanced_metrics
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Add leaderboard root to sys.path
leaderboard_root = os.path.abspath('src/vendor/gorilla_bfcl_v1/berkeley-function-call-leaderboard')
if leaderboard_root not in sys.path:
    sys.path.insert(0, leaderboard_root)

# Change to eval_checker directory before importing eval_runner
eval_checker_dir = os.path.join(leaderboard_root, 'eval_checker')
if eval_checker_dir not in sys.path:
    sys.path.insert(0, eval_checker_dir)
original_cwd = os.getcwd()
os.chdir(eval_checker_dir)

# Import BFCL evaluation components
from eval_runner import (
    get_handler,
    is_executable,
    extract_after_test,
    is_rest
)
from eval_runner_helper import (
    is_executable_format_output,
    is_rest_format_output
)
from eval_checker_constant import REAL_TIME_MATCH_ALLOWED_DIFFERENCE

# Cache configuration
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "function_call_cache.json")
_cache = {}

def load_cache():
    """Load function call cache from JSON file"""
    try:
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache():
    """Save function call cache to JSON file"""
    global _cache
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save cache file {CACHE_FILE}: {e}")

def initialize_cache():
    """Initialize the global cache"""
    global _cache
    _cache = load_cache()
    logger.info(f"Loaded cache with {len(_cache)} entries")

def get_cache_key(function_call: str) -> str:
    """Generate cache key from function call"""
    import hashlib
    return hashlib.md5(function_call.encode()).hexdigest()

def get_missing_functions_patch():
    """Return Python code to patch in missing functions that are referenced in tests but not defined."""
    return '''
# Missing function implementations for BFCL tests
from typing import List, Union

def calculate_total_price(room_price: float, nights: int, discount: float = 0) -> float:
    return room_price * nights - discount

def calculate_total(quantities: List[int], prices: List[float]) -> float:
    return sum(q * p for q, p in zip(quantities, prices))

def compound_interest(principal: int, rate: float, times_compounded: int, years: int) -> float:
    return principal * (1 + rate / times_compounded) ** (times_compounded * years)

def inflation_adjustment(amount: float, inflation_rate: float, years: int) -> float:
    return amount / ((1 + inflation_rate) ** years)

def adjust_for_inflation(investment_value: float, inflation_rates: List[float]) -> float:
    adjusted_value = investment_value
    for rate in inflation_rates:
        adjusted_value = adjusted_value / (1 + rate)
    return adjusted_value

def calculate_interest_rate(principal: float, rate: float, time: float) -> float:
    return principal * rate * time

def get_movie_genre(movie_name: str) -> str:
    movie_genres = {"Avatar": "Action, Adventure, Fantasy", "Pulp Fiction": "Crime, Drama", "The Matrix": "Action, Sci-Fi"}
    return movie_genres.get(movie_name, "Unknown")

def get_director_by_movie_name(movie_name: str) -> str:
    movie_directors = {"Avatar": "James Cameron", "Pulp Fiction": "Quentin Tarantino", "The Matrix": "The Wachowski Brothers"}
    return movie_directors.get(movie_name, "Unknown")

def calculate_basal_metabolic_rate(weight: float, height: float, age: float, gender: str) -> float:
    if gender.lower() == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    elif gender.lower() == 'female':
        return 10 * weight + 6.25 * height - 5 * age - 161
    else:
        return 10 * weight + 6.25 * height - 5 * age - 78

def calculate_daily_energy_expenditure(basal_metabolic_rate: float, activity_level: float) -> float:
    multipliers = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    return basal_metabolic_rate * multipliers.get(activity_level, 1.55)

def convert_binary_to_decimal(binary: str) -> int:
    return int(binary, 2)

def convert_decimal_to_hex(decimal: int) -> str:
    return hex(decimal)

def calculate_slope(x: List[int], y: List[int]) -> float:
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x_squared = sum(xi ** 2 for xi in x)
    return (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)

def calculate_intercept(x: List[int], y: List[int], slope: int) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    return mean_y - slope * mean_x

def predict_value(slope: int, intercept: int, x: int) -> float:
    return slope * x + intercept

def apply_discount(total: float, discount: float) -> float:
    return total * (1 - discount / 100)

def confirm_booking(customer_id: str, room_number: str, total_price: float) -> dict:
    return {"status": "confirmed", "customer_id": customer_id, "room_number": room_number, "total_price": total_price, "confirmation_code": f"CONF-{customer_id}-{room_number}"}

def generate_random_number(min: int, max: int) -> int:
    import random
    return random.randint(min, max)

def convert_temperature(temperature: float, unit_from: str, unit_to: str) -> float:
    if unit_from.lower() == "celsius" and unit_to.lower() == "fahrenheit":
        return (temperature * 9/5) + 32
    elif unit_from.lower() == "fahrenheit" and unit_to.lower() == "celsius":
        return (temperature - 32) * 5/9
    else:
        return temperature

def convert_coordinates(coordinates: List[tuple]) -> List[List[float]]:
    return [list(coord) for coord in coordinates]

def validate_polygon(vertices: List[List[float]]) -> bool:
    if len(vertices) < 3:
        return False
    for vertex in vertices:
        if len(vertex) != 2:
            return False
    for i in range(len(vertices)):
        current = vertices[i]
        next_vertex = vertices[(i + 1) % len(vertices)]
        if current[0] == next_vertex[0] and current[1] == next_vertex[1]:
            return False
    return True
'''

def apply_aliases(function_call: str, name_map: Dict[str, str]) -> str:
    """Convert obfuscated function call to actual function call using name_map.
    
    Args:
        function_call: The function call with potentially obfuscated names
        name_map: Dict mapping original names to obfuscated names
    
    Returns:
        The function call with original names restored
    """
    actual_call = function_call
    for original, obfuscated in name_map.items():
        # Replace obfuscated function name with original
        if actual_call.startswith(obfuscated + "("):
            actual_call = actual_call.replace(obfuscated + "(", original + "(", 1)
            break
    return actual_call


def execute_function_and_get_output(function_call: str, alias_code: str = "", name_map: Dict[str, str] = None) -> tuple[Any, Optional[str]]:
    """Execute a function call and return the output, with caching support.
    
    Note: This must be called while in the eval_checker directory for imports to work.
    
    Args:
        function_call: The function call string to execute
        alias_code: Code to create aliases for obfuscated names (e.g., "function_0 = original_function")
        name_map: Dict mapping original names to obfuscated names
    """
    global _cache
    
    if name_map is None:
        name_map = {}
    
    # Convert to actual function call for caching
    actual_function_call = apply_aliases(function_call, name_map)
    cache_key = get_cache_key(actual_function_call)
    
    # Check cache first
    if cache_key in _cache:
        cached_result = _cache[cache_key]
        return cached_result["output"], cached_result.get("error")
    
    # Execute the function call
    exec_dict = {}
    try:
        # Import original functions and patch in missing ones
        exec_code = "from executable_python_function import *\n"
        exec_code += get_missing_functions_patch()
        exec_code += alias_code + "\nresult = " + function_call
        exec(exec_code, exec_dict)
        
        exec_output = exec_dict["result"]
        # Convert tuple to list for JSON serialization
        if isinstance(exec_output, tuple):
            exec_output = list(exec_output)
        
        # Cache the successful result (using actual function call as key)
        _cache[cache_key] = {"output": exec_output, "error": None}
        save_cache()
        
        return exec_output, None
    except Exception as e:
        # Don't cache errors with obfuscated names as they can cause cross-contamination
        error_msg = str(e)
        # Replace original names with obfuscated names in error message
        for original, obfuscated in name_map.items():
            error_msg = error_msg.replace(original, obfuscated)
        return None, error_msg


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file (one JSON object per line)."""
    results = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
    return results

def load_test_prompt(result_dir: Path) -> List[Dict]:
    """Load test prompts from the config file referenced in metadata."""
    # Load metadata - either metadata.json (improvements) or v0_metadata.json (base runs)
    metadata_file = result_dir / "metadata.json"
    if not metadata_file.exists():
        # Check for v0_metadata.json (base run pattern)
        metadata_file = result_dir / "v0_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Neither metadata.json nor v0_metadata.json found in {result_dir}")

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    config_source = metadata.get("config_source")
    if not config_source:
        raise ValueError("No config_source found in metadata")
    
    config_path = Path(config_source)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the config file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract test items with ground truth
    test_items = []
    for test in config.get("tests", []):
        # Create a simplified prompt item with what we need for evaluation
        prompt_item = {
            "question": test.get("question", ""),
            "ground_truth": test.get("ground_truth", []),
            "function": test.get("tools", [])  # Function definitions
        }
        
        # Include execution results if already computed
        if "execution_result" in test:
            prompt_item["execution_result"] = test["execution_result"]
        if "execution_result_type" in test:
            prompt_item["execution_result_type"] = test["execution_result_type"]
            
        test_items.append(prompt_item)
    
    return test_items

def evaluate_single_function(function_call: str, expected_result: Any, expected_type: str, name_map: Dict[str, str] = None) -> Dict[str, Any]:
    """Evaluate a single function call against expected result.
    
    Args:
        function_call: The function call to evaluate
        expected_result: The expected result
        expected_type: Type of comparison (exact_match or real_time_match)
        name_map: Dict mapping original names to obfuscated names
    """
    
    if name_map is None:
        name_map = {}
    
    # Create alias code for obfuscated names
    alias_code = "\n"
    for original, obfuscated in name_map.items():
        alias_code += f"{obfuscated} = {original}\n"
    
    # Execute the function
    exec_output, exec_error = execute_function_and_get_output(function_call, alias_code, name_map)
    
    if exec_error:
        return {
            "valid": False,
            "error": [f"Error in execution: {repr(function_call)}. Error: {exec_error}"],
            "error_type": "executable_checker:execution_error",
            "executed_result": None
        }
    
    # Check result based on expected type
    if expected_type == "exact_match":
        if exec_output == expected_result:
            return {
                "valid": True,
                "error": [],
                "executed_result": exec_output
            }
        else:
            return {
                "valid": False,
                "error": [f"Wrong execution result for {repr(function_call)}. Expected: {expected_result}, but got: {exec_output}."],
                "error_type": "executable_checker:wrong_result",
                "executed_result": exec_output
            }
    elif expected_type == "real_time_match":
        # Allow for 5% difference for real-time values
        if isinstance(expected_result, (int, float)) and isinstance(exec_output, (int, float)):
            if not (
                expected_result * (1 - REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
                <= exec_output
                <= expected_result * (1 + REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
            ):
                return {
                    "valid": False,
                    "error": [f"Wrong execution result for {repr(function_call)}. Expected: {expected_result}, but got: {exec_output}."],
                    "error_type": "executable_checker:wrong_result",
                    "executed_result": exec_output
                }
        else:
            # For non-numeric types, fall back to exact match
            if exec_output != expected_result:
                return {
                    "valid": False,
                    "error": [f"Wrong execution result for {repr(function_call)}. Expected: {expected_result}, but got: {exec_output}."],
                    "error_type": "executable_checker:wrong_result",
                    "executed_result": exec_output
                }
        
        return {
            "valid": True,
            "error": [],
            "executed_result": exec_output
        }
    else:
        # Default to exact match
        if exec_output == expected_result:
            return {
                "valid": True,
                "error": [],
                "executed_result": exec_output
            }
        else:
            return {
                "valid": False,
                "error": [f"Wrong execution result for {repr(function_call)}. Expected: {expected_result}, but got: {exec_output}."],
                "error_type": "executable_checker:wrong_result",
                "executed_result": exec_output
            }


def evaluate_result(model_result: Dict, prompt_item: Dict, test_category: str) -> Dict[str, Any]:
    """Evaluate a single model result against ground truth."""
    
    # Get name mapping if available (for obfuscated tests)
    name_map = model_result.get("name_mapping", {})
    
    # Create alias code for obfuscated names
    alias_code = "\n"
    for original, obfuscated in name_map.items():
        alias_code += f"{obfuscated} = {original}\n"
    
    # Parse the model result
    result_str = model_result.get("result", "[]")
    try:
        if isinstance(result_str, str):
            decoded_result = json.loads(result_str)
        else:
            decoded_result = result_str
    except json.JSONDecodeError:
        return {
            "valid": False,
            "error": ["Failed to decode executable."],
            "error_type": "executable_decoder:decoder_failed",
            "executed_result": None
        }
    
    # Transform to function call format
    function_calls = []
    for call in decoded_result:
        if isinstance(call, dict):
            func_name = call.get("function", "")
            args = call.get("args", {})
            
            # Format as Python function call
            if args:
                args_str = ", ".join([f"{k}={repr(v)}" for k, v in args.items()])
                function_calls.append(f"{func_name}({args_str})")
            else:
                function_calls.append(f"{func_name}()")
    
    # Get expected results
    ground_truth = prompt_item.get("ground_truth", [])
    execution_results = prompt_item.get("execution_result", [])
    execution_types = prompt_item.get("execution_result_type", ["exact_match"] * len(execution_results))

    # Always execute ALL function calls first, regardless of count
    # This ensures we always capture the raw execution results/errors
    execution_results_list = []
    for func_call in function_calls:
        exec_output, exec_error = execute_function_and_get_output(func_call, alias_code, name_map)
        if exec_error:
            # Store the error message as the result
            execution_results_list.append({"error": exec_error, "function_call": func_call})
        else:
            # Store the successful output
            execution_results_list.append(exec_output)

    # Check if we have the right number of function calls
    if len(function_calls) != len(ground_truth):
        return {
            "valid": False,
            "error": [f"Wrong number of functions. Expected {len(ground_truth)}, got {len(function_calls)}."],
            "error_type": "simple_exec_checker:wrong_count",
            "executed_result": execution_results_list
        }

    # For multiple functions, we need to handle parallel/multiple categories
    if "multiple" in test_category or "parallel" in test_category:
        # Convert execution_results_list to the format expected by the matching logic
        execution_tuples = []
        for i, result in enumerate(execution_results_list):
            if isinstance(result, dict) and "error" in result:
                # This was an error
                execution_tuples.append((None, result["error"], result["function_call"]))
            else:
                # This was a successful execution
                execution_tuples.append((result, None, function_calls[i]))
        
        # Try to match each expected result with a model result (no order requirement)
        matched_indices = []
        for i in range(len(execution_results)):
            found_match = False
            all_errors = []

            for index in range(len(execution_tuples)):
                if index in matched_indices:
                    continue

                exec_output, exec_error, func_call = execution_tuples[index]
                
                # Check if this execution result matches the expected result
                if exec_error:
                    all_errors.append({
                        f"Model Result Index {index}": {
                            "sub_error": [f"Error in execution: {repr(func_call)}. Error: {exec_error}"],
                            "sub_error_type": "executable_checker:execution_error",
                            "executed_result": None
                        }
                    })
                    continue
                
                # Validate this result against expected result i
                expected_result = execution_results[i]
                expected_type = execution_types[i]
                
                matches = False
                if expected_type == "exact_match":
                    matches = (exec_output == expected_result)
                elif expected_type == "real_time_match":
                    if (type(expected_result) == float or type(expected_result) == int) and (
                        type(exec_output) == float or type(exec_output) == int
                    ):
                        matches = (
                            expected_result * (1 - REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
                            <= exec_output
                            <= expected_result * (1 + REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
                        )
                    else:
                        matches = (exec_output == expected_result)
                else:
                    matches = (exec_output == expected_result)
                
                if matches:
                    matched_indices.append(index)
                    found_match = True
                    break
                else:
                    all_errors.append({
                        f"Model Result Index {index}": {
                            "sub_error": [f"Wrong execution result for {repr(func_call)}. Expected: {expected_result}, but got: {exec_output}."],
                            "sub_error_type": "executable_checker:wrong_result",
                            "executed_result": exec_output
                        }
                    })
            
            if not found_match:
                return {
                    "valid": False,
                    "error": [f"Could not find a matching function among index {list(range(len(function_calls)))} of model output for index {i} of possible answers."] + all_errors,
                    "error_type": "executable_checker:cannot_find_match",
                    "executed_result": execution_results_list
                }

        return {
            "valid": True,
            "error": [],
            "executed_result": execution_results_list
        }
    else:
        # Single function - evaluate directly
        if len(function_calls) == 0:
            return {
                "valid": False,
                "error": ["Wrong number of functions."],
                "error_type": "simple_exec_checker:wrong_count",
                "executed_result": []
            }

        # For single function, we already executed it above
        # Just return the result with validation
        result = execution_results_list[0] if execution_results_list else None

        # Check if it's an error
        if isinstance(result, dict) and "error" in result:
            return {
                "valid": False,
                "error": [f"Error in execution: {repr(result['function_call'])}. Error: {result['error']}"],
                "error_type": "executable_checker:execution_error",
                "executed_result": [result]
            }

        # Compare with expected result
        expected_result = execution_results[0] if execution_results else None
        expected_type = execution_types[0] if execution_types else "exact_match"

        matches = False
        if expected_result is not None:
            if expected_type == "exact_match":
                matches = (result == expected_result)
            elif expected_type == "real_time_match":
                if (type(expected_result) == float or type(expected_result) == int) and (
                    type(result) == float or type(result) == int
                ):
                    matches = (
                        expected_result * (1 - REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
                        <= result
                        <= expected_result * (1 + REAL_TIME_MATCH_ALLOWED_DIFFERENCE)
                    )
                else:
                    matches = (result == expected_result)
            else:
                matches = (result == expected_result)

        if matches:
            return {
                "valid": True,
                "error": [],
                "executed_result": [result]
            }
        else:
            return {
                "valid": False,
                "error": [f"Wrong execution result for {repr(function_calls[0])}. Expected: {expected_result}, but got: {result}."],
                "error_type": "executable_checker:wrong_result",
                "executed_result": [result]
            }


def compute_ground_truth_results(prompt_items: List[Dict]) -> None:
    """Compute execution results for ground truth if not already present.
    
    Uses caching to avoid re-executing identical function calls.
    """
    for item in tqdm(prompt_items, desc="Computing ground truth execution results"):
        if "execution_result" in item:
            continue
        
        execution_results = []
        ground_truth = item.get("ground_truth", [])
        
        for gt in ground_truth:
            # Execute with caching (no name_map for ground truth)
            exec_output, exec_error = execute_function_and_get_output(gt, "", {})
            
            if exec_error:
                raise Exception(f"Error executing ground truth: {exec_error}")
            
            execution_results.append(exec_output)
        
        item["execution_result"] = execution_results
        item["execution_result_type"] = ["exact_match"] * len(execution_results)


def find_previous_scored(result_dir: Path, current_version: int) -> Optional[Path]:
    """Find the previous iteration's scored results."""
    if current_version == 1:
        # First improvement - look for v0_scored.json in base run
        base_path = get_base_run_path(result_dir)
        if base_path:
            v0_scored = base_path / "v0_scored.json"
            if v0_scored.exists():
                return v0_scored
    else:
        # Later improvements - look for previous version
        parent = result_dir.parent
        prev_scored = parent / f"v{current_version-1}" / "scored.json"
        if prev_scored.exists():
            return prev_scored
    return None


def evaluate_category_results(result_file: Path, test_category: str, result_dir: Path) -> Dict[str, Any]:
    """Evaluate all results in a category file."""

    # Load model results
    model_results = load_jsonl(result_file)
    if not model_results:
        logger.warning(f"No results found in {result_file}")
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    # Load test prompts from config
    prompt_items = load_test_prompt(result_dir)

    # Check if we can reuse previous evaluation results
    reused_evaluations = {}
    is_improvement = (result_dir / "config.json").exists()

    if is_improvement:
        # Try to find previous scored results
        improvement_match = re.match(r'.*/improvements/.*/v(\d+)$', str(result_dir))
        if improvement_match:
            version = int(improvement_match.group(1))
            if version > 0:
                prev_scored_path = find_previous_scored(result_dir, version)

                if prev_scored_path:
                    logger.info(f"Found previous scored results at {prev_scored_path}")

                    # Load previous scored results
                    with open(prev_scored_path, 'r') as f:
                        prev_scored = json.load(f)

                    # Load current and previous configs for comparison
                    with open(result_dir / "config.json", 'r') as f:
                        current_config = json.load(f)

                    # Find previous config
                    if version == 1:
                        # Get from v0 metadata
                        base_path = get_base_run_path(result_dir)
                        if base_path:
                            v0_metadata = base_path / "v0_metadata.json"
                            if v0_metadata.exists():
                                with open(v0_metadata, 'r') as f:
                                    prev_config_path = json.load(f).get("config_source")
                                    if prev_config_path and Path(prev_config_path).exists():
                                        with open(prev_config_path, 'r') as f:
                                            prev_config = json.load(f)
                                    else:
                                        prev_config = None
                            else:
                                prev_config = None
                    else:
                        # Get from previous version's config
                        prev_config_path = result_dir.parent / f"v{version-1}" / "config.json"
                        if prev_config_path.exists():
                            with open(prev_config_path, 'r') as f:
                                prev_config = json.load(f)
                        else:
                            prev_config = None

                    # Compare configs and reuse unchanged evaluations
                    if prev_config:
                        for prev_eval in prev_scored.get("detailed_evaluations", []):
                            test_idx = prev_eval["index"]

                            # Make sure indices are valid
                            if (test_idx < len(current_config.get("tests", [])) and
                                test_idx < len(prev_config.get("tests", []))):

                                current_test = current_config["tests"][test_idx]
                                prev_test = prev_config["tests"][test_idx]

                                # Compare function tools
                                if functions_are_identical(current_test.get("tools", []),
                                                          prev_test.get("tools", [])):
                                    reused_evaluations[test_idx] = prev_eval # Functions unchanged - reuse evaluation
                                    # logger.info(f"Test {test_idx}: Reusing evaluation (functions unchanged)")

    # Ensure we have execution results for ground truth
    # Need to be in eval_checker directory for executable_python_function imports
    os.chdir(eval_checker_dir)
    compute_ground_truth_results(prompt_items)
    os.chdir(original_cwd)
    
    # Evaluate each result
    correct = 0
    total = len(model_results)
    detailed_evaluations = []

    for i, model_result in enumerate(tqdm(model_results, desc=f"Evaluating {test_category}")):
        if i >= len(prompt_items):
            logger.warning(f"More model results than prompt items for {test_category}")
            break

        # Check if we have a reused evaluation for this index
        if i in reused_evaluations:
            detailed_eval = reused_evaluations[i]             # Use the reused evaluation

            if detailed_eval["valid"]:
                correct += 1
            detailed_evaluations.append(detailed_eval)
            logger.debug(f"Test {i}: Used reused evaluation")
        else:
            # Need to evaluate this test
            # Need to be in eval_checker directory for function execution
            os.chdir(eval_checker_dir)
            eval_result = evaluate_result(model_result, prompt_items[i], test_category)
            os.chdir(original_cwd)

            if eval_result["valid"]:
                correct += 1

            # Calculate enhanced metrics
            enhanced_metrics = evaluate_enhanced_metrics(
                model_result,
                prompt_items[i].get("ground_truth", []),
                prompt_items[i].get("function", []),  # Available functions
                model_result.get("name_mapping", {})  # Name mapping for obfuscated functions
            )

            # Store detailed evaluation for each test
            detailed_eval = {
                "index": i,
                "question": model_result.get("question", ""),
                "model_result": model_result.get("result", ""),
                "ground_truth": prompt_items[i].get("ground_truth", []),
                "ground_truth_execution": prompt_items[i].get("execution_result", []),  # Add the executed ground truth
                "valid": eval_result["valid"],
                "error": eval_result.get("error", []),
                "error_type": eval_result.get("error_type", ""),
                "executed_result": eval_result.get("executed_result"),
                "parameter_metrics": enhanced_metrics["parameter_metrics"],  # NEW
                "ast_metrics": enhanced_metrics["ast_metrics"]  # NEW
            }

            # Keep errors as list for consistency with bfcl/eval.py
            detailed_evaluations.append(detailed_eval)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "category": test_category,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "detailed_evaluations": detailed_evaluations
    }


def main():
    os.chdir(original_cwd)

    parser = argparse.ArgumentParser(description="Evaluate BFCL function call results")
    parser.add_argument("--result-dir", type=str, required=True,
                       help="Directory containing result files to evaluate")
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        logger.error(f"Result directory not found: {result_dir}")
        sys.exit(1)
    
    # Initialize the function execution cache
    initialize_cache()

    # Find results and metadata files based on new structure
    result_files = []
    metadata_file = None
    results_file = None

    # Check for improvement directory structure (has config.json)
    if (result_dir / "config.json").exists():
        # This is an improvement directory
        results_file = result_dir / "results.json"
        metadata_file = result_dir / "metadata.json"
        if results_file.exists() and metadata_file.exists():
            logger.info(f"Found improvement directory results")
        else:
            logger.error(f"Improvement directory missing results.json or metadata.json")
            sys.exit(1)
    else:
        # This is a base run directory - look for v{N}_results.json
        versioned_results = list(result_dir.glob("v*_results.json"))
        versioned_metadata = list(result_dir.glob("v*_metadata.json"))

        if versioned_results:
            # Find the latest version
            import re
            latest_version = -1
            latest_results = None
            latest_metadata = None

            for res_file in versioned_results:
                match = re.match(r'v(\d+)_results\.json', res_file.name)
                if match:
                    version = int(match.group(1))
                    if version > latest_version:
                        latest_version = version
                        latest_results = res_file
                        # Find corresponding metadata
                        meta_file = result_dir / f"v{version}_metadata.json"
                        if meta_file.exists():
                            latest_metadata = meta_file

            if latest_results and latest_metadata:
                results_file = latest_results
                metadata_file = latest_metadata
                logger.info(f"Found base run results: {results_file.name}")
            else:
                logger.error(f"Could not find matching results and metadata files")
                sys.exit(1)
        else:
            # Fallback: check for unversioned results.json
            results_file = result_dir / "results.json"
            metadata_file = result_dir / "metadata.json"
            if not results_file.exists():
                logger.error(f"No results files found in {result_dir}")
                sys.exit(1)

    # Load metadata to get test category
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        test_category = metadata.get("config_test_category", "unknown")

    result_files = [(results_file, test_category)]

    if not result_files:
        logger.error(f"No result files found in {result_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(result_files)} result file(s) to evaluate")
    
    # Evaluate each category
    all_results = {}
    total_correct = 0
    total_count = 0
    
    for result_file, test_category in result_files:
        logger.info(f"\nEvaluating {test_category}...")
        category_results = evaluate_category_results(result_file, test_category, result_dir)
        
        all_results[test_category] = category_results
        total_correct += category_results["correct"]
        total_count += category_results["total"]
        
        logger.info(f"{test_category}: {category_results['accuracy']:.2%} ({category_results['correct']}/{category_results['total']})")
    
    # Overall accuracy
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    # Collect all detailed evaluations and extract enhanced metrics
    all_detailed_evaluations = []
    all_enhanced_metrics = []
    for category, category_results in all_results.items():
        if "detailed_evaluations" in category_results:
            for eval_item in category_results["detailed_evaluations"]:
                eval_item["category"] = category  # Add category to each evaluation
                all_detailed_evaluations.append(eval_item)

                # Collect enhanced metrics for aggregation
                if "parameter_metrics" in eval_item and "ast_metrics" in eval_item:
                    all_enhanced_metrics.append({
                        "parameter_metrics": eval_item["parameter_metrics"],
                        "ast_metrics": eval_item["ast_metrics"]
                    })

    # Aggregate enhanced metrics
    aggregated_enhanced = aggregate_enhanced_metrics(all_enhanced_metrics) if all_enhanced_metrics else {}

    # Build summary with enhanced metrics
    summary = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_count,
        "by_category": {
            cat: {
                "accuracy": res["accuracy"],
                "correct": res["correct"],
                "total": res["total"]
            }
            for cat, res in all_results.items()
        }
    }

    # Add enhanced metrics to summary
    summary.update(aggregated_enhanced)
    
    # Save scored results with appropriate naming
    # Determine if this is a base run or improvement
    if (result_dir / "config.json").exists():
        # Improvement directory - save as scored.json
        output_path = result_dir / "scored.json"
    else:
        # Base run - save as v{N}_scored.json matching the results file
        # Extract version from the results file we evaluated
        import re
        results_file_name = result_files[0][0].name
        match = re.match(r'v(\d+)_results\.json', results_file_name)
        if match:
            version = match.group(1)
            output_path = result_dir / f"v{version}_scored.json"
        else:
            # Fallback for unversioned files
            output_path = result_dir / "scored.json"

    output = {
        "summary": summary,
        "detailed_evaluations": all_detailed_evaluations
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\n" + "="*50)
    logger.info(f"Overall accuracy: {overall_accuracy:.2%} ({total_correct}/{total_count})")
    logger.info(f"Parameter accuracy (avg): {aggregated_enhanced.get('parameter_accuracy_avg', 0):.2%}")
    logger.info(f"AST format score (avg): {aggregated_enhanced.get('ast_format_score_avg', 0):.2%}")

    # Log common errors if present
    common_errors = aggregated_enhanced.get('common_errors', {})
    if common_errors.get('missing_parameters'):
        logger.info(f"Most common missing parameters: {common_errors['missing_parameters'][:3]}")
    if common_errors.get('type_mismatches'):
        logger.info(f"Common type mismatches: {list(common_errors['type_mismatches'].keys())[:3]}")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()