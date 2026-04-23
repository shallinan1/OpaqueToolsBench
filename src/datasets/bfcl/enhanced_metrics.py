"""
Enhanced metrics for BFCL evaluation including parameter accuracy and AST/format analysis.

This module provides additional granular metrics beyond simple pass/fail:
- Parameter Accuracy: Measures how many parameters are correctly provided
- AST/Format Analysis: Measures structural correctness of function calls
"""

import json
import ast
from itertools import permutations
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter


@dataclass
class ParameterMetrics:
    """Container for parameter-level accuracy metrics."""
    total_params: int = 0
    correct_params: int = 0
    missing_params: List[str] = field(default_factory=list)
    extra_params: List[str] = field(default_factory=list)
    wrong_value_params: List[str] = field(default_factory=list)
    accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "total_params": self.total_params,
            "correct_params": self.correct_params,
            "missing_params": self.missing_params,
            "extra_params": self.extra_params,
            "wrong_value_params": self.wrong_value_params
        }


@dataclass
class ASTMetrics:
    """Container for AST/format analysis metrics."""
    format_valid: bool = False
    structure_valid: bool = False
    type_correctness: float = 0.0
    schema_compliant: bool = False
    has_hallucinated_params: bool = False
    type_errors: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0  # Overall AST score (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_valid": self.format_valid,
            "structure_valid": self.structure_valid,
            "type_correctness": self.type_correctness,
            "schema_compliant": self.schema_compliant,
            "has_hallucinated_params": self.has_hallucinated_params,
            "type_errors": self.type_errors,
            "score": self.score
        }


def parse_python_function_call(call_str: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse a Python function call string like "func(a=1, b='test')" into name and args.

    Args:
        call_str: String representation of a Python function call

    Returns:
        Tuple of (function_name, arguments_dict) or None if parse fails
    """
    try:
        # Parse the string as Python code
        tree = ast.parse(call_str)

        # Get the function call node
        if not isinstance(tree.body[0], ast.Expr):
            return None
        call_node = tree.body[0].value
        if not isinstance(call_node, ast.Call):
            return None

        # Get function name
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        else:
            return None

        # Extract arguments
        args = {}

        # Process keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg:  # Skip **kwargs
                # Evaluate the value
                try:
                    # Use ast.literal_eval for safe evaluation of literals
                    value = ast.literal_eval(keyword.value)
                except (ValueError, SyntaxError):
                    # For more complex expressions, try to evaluate them
                    # This handles cases like "p=1/6"
                    try:
                        # Create a restricted eval environment
                        value = eval(compile(ast.Expression(keyword.value), '', 'eval'))
                    except Exception:
                        # If evaluation fails, store as string representation
                        value = ast.unparse(keyword.value) if hasattr(ast, 'unparse') else str(keyword.value)

                args[keyword.arg] = value

        # Process positional arguments (less common in BFCL, but handle them)
        # Note: We'd need the function signature to map these properly
        # For now, just store them with index-based keys if present
        for i, arg in enumerate(call_node.args):
            try:
                value = ast.literal_eval(arg)
            except (ValueError, SyntaxError):
                try:
                    value = eval(compile(ast.Expression(arg), '', 'eval'))
                except Exception:
                    value = ast.unparse(arg) if hasattr(ast, 'unparse') else str(arg)
            args[f"_positional_{i}"] = value

        return func_name, args

    except (SyntaxError, AttributeError, TypeError):
        return None


def parse_function_call(call_data: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse a function call (either JSON dict or string) into function name and arguments.

    Args:
        call_data: Either a dict with 'function' and 'args' keys, or a JSON string

    Returns:
        Tuple of (function_name, arguments_dict) or None if parse fails
    """
    if isinstance(call_data, str):
        # First try to parse as JSON
        try:
            call_data = json.loads(call_data)
        except json.JSONDecodeError:
            # If JSON parsing fails, try parsing as Python function call
            return parse_python_function_call(call_data)

    if not isinstance(call_data, dict):
        return None

    func_name = call_data.get("function", "")
    args = call_data.get("args", {})

    if not isinstance(args, dict):
        return None

    return func_name, args


def calculate_parameter_accuracy(
    model_func: str,
    model_args: Dict[str, Any],
    ground_truth_func: str,
    ground_truth_args: Dict[str, Any],
    name_mapping: Optional[Dict[str, str]] = None
) -> ParameterMetrics:
    """
    Calculate detailed parameter-level accuracy metrics.

    Only calculates parameter accuracy if the function name is correct,
    otherwise returns 0% accuracy.

    Args:
        model_func: Function name from model (possibly obfuscated)
        model_args: Arguments from model's function call
        ground_truth_func: Function name from ground truth (real name)
        ground_truth_args: Arguments from ground truth function call
        name_mapping: Optional dict mapping real names to obfuscated names

    Returns:
        ParameterMetrics with detailed accuracy information
    """
    metrics = ParameterMetrics()

    # Handle name mapping for obfuscated function names
    if name_mapping:
        # Check if the obfuscated name maps to the ground truth function
        # name_mapping is {real_name: obfuscated_name}
        expected_obfuscated = name_mapping.get(ground_truth_func)
        if expected_obfuscated != model_func:
            metrics.total_params = len(ground_truth_args)
            metrics.accuracy = 0.0
            return metrics
    else:
        # No name mapping, compare directly
        if model_func != ground_truth_func:
            metrics.total_params = len(ground_truth_args)
            metrics.accuracy = 0.0
            return metrics

    model_keys = set(model_args.keys())
    gt_keys = set(ground_truth_args.keys())

    # Find missing, extra, and common parameters
    metrics.missing_params = list(gt_keys - model_keys)
    metrics.extra_params = list(model_keys - gt_keys)
    common_keys = model_keys & gt_keys

    # Check values for common parameters
    for key in common_keys:
        if model_args[key] == ground_truth_args[key]:
            metrics.correct_params += 1
        else:
            metrics.wrong_value_params.append(key)

    # Calculate total and accuracy
    metrics.total_params = len(gt_keys)
    if metrics.total_params > 0:
        metrics.accuracy = metrics.correct_params / metrics.total_params
    else:
        # No parameters expected - if model also has no params, that's correct
        metrics.accuracy = 1.0 if len(model_keys) == 0 else 0.0

    return metrics


def check_type_match(value1: Any, value2: Any) -> bool:
    """Check if two values have the same type structure."""
    if type(value1) != type(value2):
        return False

    # For nested structures, check recursively
    if isinstance(value1, dict) and isinstance(value2, dict):
        if set(value1.keys()) != set(value2.keys()):
            return False
        return all(check_type_match(value1[k], value2[k]) for k in value1.keys())

    if isinstance(value1, (list, tuple)) and isinstance(value2, (list, tuple)):
        if len(value1) != len(value2):
            return True  # Different lengths but same type is ok
        return all(check_type_match(v1, v2) for v1, v2 in zip(value1, value2))

    return True


def calculate_ast_metrics(
    model_data: Any,
    ground_truth_data: Any,
    available_functions: Optional[List[Dict]] = None,
    name_mapping: Optional[Dict[str, str]] = None
) -> ASTMetrics:
    """
    Calculate AST/format analysis metrics for a function call.

    This checks structural and format correctness, not value correctness.

    Args:
        model_data: The model's function call (dict or string)
        ground_truth_data: The ground truth function call (dict or string)
        available_functions: Optional list of available function definitions

    Returns:
        ASTMetrics with format analysis results
    """
    metrics = ASTMetrics()

    # Check if model output can be parsed
    model_parsed = parse_function_call(model_data)
    if model_parsed is None:
        metrics.format_valid = False
        metrics.score = 0.0
        return metrics

    model_func, model_args = model_parsed
    metrics.format_valid = True

    # Parse ground truth (can be string or dict)
    gt_parsed = parse_function_call(ground_truth_data)
    if gt_parsed is None:
        # Ground truth parsing failed - shouldn't happen
        return metrics

    gt_func, gt_args = gt_parsed

    # Check structure validity
    metrics.structure_valid = (
        isinstance(model_data, dict) and
        "function" in model_data and
        "args" in model_data and
        isinstance(model_data["args"], dict)
    )

    # Check for hallucinated parameters
    if available_functions:
        # Find the function definition
        # Need to map from obfuscated name back to real name to find in available_functions
        real_func_name = gt_func  # The ground truth has the real function name

        func_def = None
        for f in available_functions:
            if f.get("name") == real_func_name:
                func_def = f
                break

        if func_def:
            # Get expected parameters from function definition
            expected_params = set()
            if "parameters" in func_def:
                params = func_def["parameters"]
                if "properties" in params:
                    expected_params = set(params["properties"].keys())
                elif "required" in params:
                    expected_params = set(params["required"])

            # Check for hallucinated params
            actual_params = set(model_args.keys())
            hallucinated = actual_params - expected_params
            metrics.has_hallucinated_params = len(hallucinated) > 0

    # Type correctness check
    # Check if functions match (considering name mapping)
    functions_match = False
    if name_mapping:
        expected_obfuscated = name_mapping.get(gt_func)
        functions_match = (expected_obfuscated == model_func)
    else:
        functions_match = (model_func == gt_func)

    if functions_match:  # Only check types if function is correct
        total_types = len(gt_args)
        correct_types = 0

        for key, gt_value in gt_args.items():
            if key in model_args:
                model_value = model_args[key]
                if check_type_match(model_value, gt_value):
                    correct_types += 1
                else:
                    metrics.type_errors.append({
                        "param": key,
                        "expected_type": type(gt_value).__name__,
                        "got_type": type(model_value).__name__
                    })

        metrics.type_correctness = correct_types / total_types if total_types > 0 else 1.0
    else:
        metrics.type_correctness = 0.0

    # Schema compliance
    metrics.schema_compliant = (
        metrics.structure_valid and
        not metrics.has_hallucinated_params and
        metrics.type_correctness == 1.0
    )

    # Calculate overall AST score
    score_components = [
        metrics.format_valid,
        metrics.structure_valid,
        metrics.type_correctness,
        metrics.schema_compliant,
        not metrics.has_hallucinated_params
    ]
    metrics.score = sum(score_components) / len(score_components)

    return metrics


def evaluate_enhanced_metrics(
    model_result: Dict[str, Any],
    ground_truth: List[Any],
    available_functions: Optional[List[Dict]] = None,
    name_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Calculate enhanced metrics for a single test result.

    Args:
        model_result: The model's result containing 'result' field with function calls
        ground_truth: List of ground truth function calls
        available_functions: Optional list of available function definitions
        name_mapping: Optional dict mapping real function names to obfuscated names

    Returns:
        Dict containing parameter and AST metrics
    """
    # Parse model result
    result_str = model_result.get("result", "[]")
    try:
        if isinstance(result_str, str):
            model_calls = json.loads(result_str)
        else:
            model_calls = result_str
    except json.JSONDecodeError:
        # Invalid JSON - return zero metrics
        return {
            "parameter_metrics": ParameterMetrics().to_dict(),
            "ast_metrics": ASTMetrics().to_dict()
        }

    if not model_calls or not ground_truth:
        return {
            "parameter_metrics": ParameterMetrics().to_dict(),
            "ast_metrics": ASTMetrics().to_dict()
        }

    # Try all permutations of model calls against ground truth and pick the best.
    # Ground truth has at most ~4 calls (max 24 permutations), so this is cheap.
    n_gt = len(ground_truth)
    # Pad or truncate model_calls to match ground truth length for pairing
    padded_model_calls = list(model_calls[:n_gt]) + [{}] * max(0, n_gt - len(model_calls))

    best_param_metrics = None
    best_ast_metrics = None
    best_total_correct = -1

    for perm in permutations(range(len(padded_model_calls))):
        total_params = 0
        correct_params = 0
        all_missing = []
        all_extra = []
        all_wrong_value = []

        ast_scores = []
        ast_format_valid_all = True
        ast_structure_valid_all = True
        ast_schema_compliant_all = True
        ast_hallucinated_any = False
        ast_type_correctness_vals = []
        ast_type_errors_all = []

        for gt_idx in range(n_gt):
            model_call = padded_model_calls[perm[gt_idx]]
            gt_call = ground_truth[gt_idx]

            model_parsed = parse_function_call(model_call)
            gt_parsed = parse_function_call(gt_call)

            if gt_parsed:
                gt_func, gt_args = gt_parsed
            else:
                gt_func, gt_args = "", {}

            if model_parsed:
                model_func, model_args = model_parsed
                pm = calculate_parameter_accuracy(model_func, model_args, gt_func, gt_args, name_mapping)
                am = calculate_ast_metrics(model_call, gt_call, available_functions, name_mapping)
            else:
                pm = ParameterMetrics()
                am = ASTMetrics()

            total_params += pm.total_params
            correct_params += pm.correct_params
            all_missing.extend(pm.missing_params)
            all_extra.extend(pm.extra_params)
            all_wrong_value.extend(pm.wrong_value_params)

            ast_scores.append(am.score)
            ast_format_valid_all = ast_format_valid_all and am.format_valid
            ast_structure_valid_all = ast_structure_valid_all and am.structure_valid
            ast_schema_compliant_all = ast_schema_compliant_all and am.schema_compliant
            ast_hallucinated_any = ast_hallucinated_any or am.has_hallucinated_params
            ast_type_correctness_vals.append(am.type_correctness)
            ast_type_errors_all.extend(am.type_errors)

        if correct_params > best_total_correct:
            best_total_correct = correct_params

            best_param_metrics = ParameterMetrics(
                total_params=total_params,
                correct_params=correct_params,
                missing_params=all_missing,
                extra_params=all_extra,
                wrong_value_params=all_wrong_value,
                accuracy=correct_params / total_params if total_params > 0 else (1.0 if not all_extra else 0.0)
            )

            best_ast_metrics = ASTMetrics(
                format_valid=ast_format_valid_all,
                structure_valid=ast_structure_valid_all,
                type_correctness=sum(ast_type_correctness_vals) / len(ast_type_correctness_vals) if ast_type_correctness_vals else 0.0,
                schema_compliant=ast_schema_compliant_all,
                has_hallucinated_params=ast_hallucinated_any,
                type_errors=ast_type_errors_all,
                score=sum(ast_scores) / len(ast_scores) if ast_scores else 0.0
            )

    return {
        "parameter_metrics": best_param_metrics.to_dict(),
        "ast_metrics": best_ast_metrics.to_dict()
    }


def aggregate_enhanced_metrics(all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate enhanced metrics across all test results.

    Args:
        all_metrics: List of enhanced metrics for each test

    Returns:
        Dict with aggregated statistics
    """
    if not all_metrics:
        return {}

    total = len(all_metrics)

    # Extract parameter metrics
    param_accuracies = [m["parameter_metrics"]["accuracy"] for m in all_metrics]

    # Extract AST metrics
    ast_scores = [m["ast_metrics"]["score"] for m in all_metrics]
    format_valid_count = sum(1 for m in all_metrics if m["ast_metrics"]["format_valid"])
    structure_valid_count = sum(1 for m in all_metrics if m["ast_metrics"]["structure_valid"])

    # Collect common errors
    all_missing_params = []
    all_extra_params = []
    all_wrong_value_params = []
    all_type_errors = []

    for m in all_metrics:
        all_missing_params.extend(m["parameter_metrics"]["missing_params"])
        all_extra_params.extend(m["parameter_metrics"]["extra_params"])
        all_wrong_value_params.extend(m["parameter_metrics"]["wrong_value_params"])
        all_type_errors.extend(m["ast_metrics"]["type_errors"])

    # Count most common errors
    missing_counter = Counter(all_missing_params)
    extra_counter = Counter(all_extra_params)
    wrong_value_counter = Counter(all_wrong_value_params)

    # Type error patterns
    type_error_patterns = defaultdict(list)
    for error in all_type_errors:
        param = error["param"]
        pattern = f"{error['expected_type']} -> {error['got_type']}"
        type_error_patterns[param].append(pattern)

    return {
        "parameter_accuracy_avg": sum(param_accuracies) / total if total > 0 else 0.0,
        "parameter_accuracy_distribution": {
            "0%": sum(1 for a in param_accuracies if a == 0.0) / total,
            "1-25%": sum(1 for a in param_accuracies if 0 < a <= 0.25) / total,
            "26-50%": sum(1 for a in param_accuracies if 0.25 < a <= 0.5) / total,
            "51-75%": sum(1 for a in param_accuracies if 0.5 < a <= 0.75) / total,
            "76-99%": sum(1 for a in param_accuracies if 0.75 < a < 1.0) / total,
            "100%": sum(1 for a in param_accuracies if a == 1.0) / total,
        },
        "ast_format_score_avg": sum(ast_scores) / total if total > 0 else 0.0,
        "ast_format_valid_rate": format_valid_count / total,
        "ast_structure_valid_rate": structure_valid_count / total,
        "common_errors": {
            "missing_parameters": missing_counter.most_common(5),
            "extra_parameters": extra_counter.most_common(5),
            "wrong_value_parameters": wrong_value_counter.most_common(5),
            "type_mismatches": dict(list(type_error_patterns.items())[:5])
        }
    }