#!/usr/bin/env python3
"""
Iterative improvement script for BFCL function descriptions.

This script iteratively improves function descriptions by:
1. Running evaluation on a config
2. Generating improved descriptions based on results
3. Running evaluation with improved config
4. Repeating the process

New directory structure:
- Base runs: runs/bfcl/{method}/{config_name}/{hyperparam_dir}/v0_*
- Improvements: runs/bfcl/{method}/{config_name}/{hyperparam_dir}/improvements/{editing_hypers}/v{N}/

Usage:
    # Start fresh iteration
    python -m src.datasets.bfcl.iterative_improve \
        --config-source src/datasets/bfcl/tool_configs/executable_multiple_function_name[all:increasing_number]_param[all:remove_all]_config.json \
        --generation-model gpt-5 \
        --generation-prompt-key must_call_tool \
        --generation-tool-choice required \
        --editing-model gpt-5 \
        --editing-prompt-key basic_improved \
        --iterations 10 \

    # Continue from existing improvement
    python -m src.datasets.bfcl.iterative_improve \
        --config-source runs/bfcl/ours/executable_multiple_function_name[all:increasing_number]_param[all:remove_all]/gpt5mini_req_8k_must_call_tool/improvements/gpt5_basic_improved_8k/v4/config.json \
        --generation-model gpt-5-mini \
        --generation-prompt-key must_call_tool \
        --generation-tool-choice required \
        --editing-model gpt-5 \
        --editing-prompt-key basic_improved \
        --iterations 1 \
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our utilities
from src.datasets.bfcl.utils.path_utils import (
    parse_config_name,
    create_generation_dirname,
    create_editing_dirname,
    detect_improvement_context,
    get_base_run_path,
    validate_hyperparams_match
)


def run_generation(config_source: str, generation_args: Dict, output_dir: Path = Path("runs/bfcl/ours")) -> Tuple[bool, Optional[Path]]:
    """Run the generation step (run.py).

    Returns:
        (success, result_dir)
    """
    # First, predict where the results would be stored
    config_name = parse_config_name(Path(config_source))
    hyperparam_dirname = create_generation_dirname(argparse.Namespace(**generation_args))

    is_improvement, improvement_base, version = detect_improvement_context(Path(config_source))
    if is_improvement:
        result_dir = improvement_base / f"v{version}"
    else:
        result_dir = output_dir / config_name / hyperparam_dirname

    # Check if v0_results.json already exists
    v0_results_file = result_dir / "v0_results.json"
    if v0_results_file.exists():
        logger.info(f"Found existing v0_results.json at {v0_results_file}")
        logger.info("Skipping generation step - using existing results")
        return True, result_dir

    # Run generation if results don't exist
    cmd = [
        sys.executable, "-m", "src.datasets.bfcl.run",
        "--config-source", config_source,
        "--model", generation_args["model"],
        "--temperature", str(generation_args["temperature"]),
        "--tool-choice", generation_args.get("tool_choice", "required"),
        "--prompt-key", generation_args["prompt_key"],  # Required for directory naming
        "--max-tokens", str(generation_args.get("max_tokens", 8192)),
        "--seed", str(generation_args.get("seed", 0)),
        "--output-dir", str(output_dir)
    ]

    # Add --together flag if specified
    if generation_args.get("together", False):
        cmd.append("--together")

    if "top_p" in generation_args:
        cmd.extend(["--top-p", str(generation_args["top_p"])])

    if "reasoning_effort" in generation_args:
        cmd.extend(["--reasoning-effort", generation_args["reasoning_effort"]])

    if "num_queries" in generation_args:
        cmd.extend(["--num-queries", str(generation_args["num_queries"])])

    logger.info(f"Running generation: {' '.join(cmd)}")
    logger.info("="*60)

    try:
        # Run without capture_output to show live progress
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            logger.error(f"Generation failed with return code {result.returncode}")
            return False, None

        return True, result_dir

    except Exception as e:
        logger.error(f"Exception running generation: {e}")
        return False, None


def run_evaluation(result_dir: Path) -> Tuple[bool, Optional[Path]]:
    """Run the evaluation step.

    Returns:
        (success, scored_file_path)
    """
    # Determine which scored file to look for
    if (result_dir / "config.json").exists():
        # Improvement directory
        scored_file = result_dir / "scored.json"
    else:
        # Base directory - use v0_scored.json
        scored_file = result_dir / "v0_scored.json"

    # Check if scored file already exists
    if scored_file.exists():
        logger.info(f"Found existing scored file at {scored_file}")
        logger.info("Skipping evaluation step - using existing scores")
        return True, scored_file

    # Run evaluation if scored file doesn't exist
    cmd = [
        sys.executable, "-m", "src.datasets.bfcl.evaluate",
        "--result-dir", str(result_dir)
    ]

    logger.info(f"Running evaluation: {' '.join(cmd)}")
    logger.info("="*60)

    try:
        # Run without capture_output to show live progress
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            logger.error(f"Evaluation failed with return code {result.returncode}")
            return False, None

        if scored_file.exists():
            logger.info(f"Evaluation complete: {scored_file}")
            return True, scored_file
        else:
            logger.error(f"Scored file not found: {scored_file}")
            return False, None

    except Exception as e:
        logger.error(f"Exception running evaluation: {e}")
        return False, None


def run_description_generation(result_dir: Path, editing_args: Dict) -> Tuple[bool, Optional[Path]]:
    """Generate improved descriptions.

    Returns:
        (success, improved_config_path)
    """
    cmd = [
        sys.executable, "-m", "src.datasets.bfcl.generate_descriptions",
        "--result-dir", str(result_dir),
        "--model", editing_args["model"],
        "--temperature", str(editing_args["temperature"]),
        "--prompt-key", editing_args["prompt_key"],  # Required for directory naming
        "--max-tokens", str(editing_args.get("max_tokens", 8192))
    ]

    if editing_args.get("reasoning_effort") is not None:
        cmd.extend(["--reasoning-effort", editing_args["reasoning_effort"]])

    logger.info(f"Generating improved descriptions: {' '.join(cmd)}")
    logger.info("="*60)

    try:
        # Run without capture_output to show live progress
        result = subprocess.run(cmd, text=True)
        if result.returncode == 2:
            # Exit code 2 means convergence was achieved
            logger.info("Complete convergence detected - no further improvements possible")
            return True, "CONVERGED"  # Special marker for convergence
        elif result.returncode != 0:
            logger.error(f"Description generation failed with return code {result.returncode}")
            return False, None

        # If we can't parse, predict the path
        base_run_path = get_base_run_path(result_dir)
        if not base_run_path:
            base_run_path = result_dir

        editing_dirname = create_editing_dirname(
            editing_args["model"],
            editing_args["temperature"],
            editing_args["prompt_key"],  # Now required
            editing_args.get("max_tokens", 8192),
            reasoning_effort=editing_args.get("reasoning_effort")
        )

        improvements_dir = base_run_path / "improvements" / editing_dirname

        # Find the most recently created version (should be what was just generated)
        # Look for existing versions
        existing_versions = []
        if improvements_dir.exists():
            for item in improvements_dir.iterdir():
                if item.is_dir() and item.name.startswith('v'):
                    try:
                        existing_versions.append(int(item.name[1:]))
                    except ValueError:
                        pass

        if existing_versions:
            # Use the highest existing version (the one just created)
            latest_version = max(existing_versions)
            config_path = improvements_dir / f"v{latest_version}" / "config.json"
            if config_path.exists():
                return True, config_path
            else:
                logger.error(f"Expected config not found: {config_path}")
                return False, None
        else:
            logger.error(f"No improvement versions found in {improvements_dir}")
            return False, None

    except Exception as e:
        logger.error(f"Exception generating descriptions: {e}")
        return False, None


def run_iteration(
    config_source: str,
    generation_args: Dict,
    editing_args: Dict,
    iteration_num: int,
    output_dir: Path = Path("runs/bfcl/ours")
) -> Tuple[bool, Optional[str], Dict]:
    """Run one complete iteration.

    Returns:
        (success, next_config_source, metrics)
    """
    metrics = {
        "iteration": iteration_num,
        "start_time": datetime.utcnow().isoformat()
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting iteration {iteration_num}")
    logger.info(f"Config source: {config_source}")
    logger.info(f"{'='*60}")

    # Step 1: Generate results
    success, result_dir = run_generation(config_source, generation_args, output_dir=output_dir)
    if not success:
        logger.error(f"Generation failed for iteration {iteration_num}")
        metrics["failed_at"] = "generation"
        return False, None, metrics

    metrics["result_dir"] = str(result_dir)

    # Step 2: Evaluate results
    success, scored_file = run_evaluation(result_dir)
    if not success:
        logger.error(f"Evaluation failed for iteration {iteration_num}")
        metrics["failed_at"] = "evaluation"
        return False, None, metrics

    # Extract accuracy from scored file
    try:
        with open(scored_file, 'r') as f:
            scored_data = json.load(f)
            accuracy = scored_data.get("summary", {}).get("accuracy", 0.0)
            metrics["accuracy"] = accuracy
            logger.info(f"Iteration {iteration_num} accuracy: {accuracy:.2%}")
    except Exception as e:
        logger.warning(f"Could not extract accuracy: {e}")

    # Step 3: Generate improved descriptions
    success, improved_config = run_description_generation(result_dir, editing_args)
    if not success:
        logger.error(f"Description generation failed for iteration {iteration_num}")
        metrics["failed_at"] = "description_generation"
        return False, None, metrics

    # Check for convergence
    if improved_config == "CONVERGED":
        metrics["converged"] = True
        metrics["end_time"] = datetime.utcnow().isoformat()
        logger.info(f"Iteration {iteration_num}: Convergence achieved!")
        return True, None, metrics  # Return None for config to signal convergence

    metrics["improved_config"] = str(improved_config)
    metrics["end_time"] = datetime.utcnow().isoformat()

    logger.info(f"Iteration {iteration_num} complete. Next config: {improved_config}")
    return True, str(improved_config), metrics


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively improve BFCL function descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config source
    parser.add_argument("--config-source", required=True,
                       help="Initial config file or improved config from previous iteration")

    # Generation parameters
    parser.add_argument("--generation-model", required=True,
                       help="Model for generating function calls (e.g., gpt-5)")
    parser.add_argument("--generation-together", action="store_true",
                       help="Use Together AI API for generation model")
    parser.add_argument("--generation-temperature", type=float, default=0.001,
                       help="Temperature for generation")
    parser.add_argument("--generation-top-p", type=float, default=1.0,
                       help="Top-p for generation")
    parser.add_argument("--generation-tool-choice", default="required",
                       choices=["required", "auto", "none"])
    parser.add_argument("--generation-prompt-key", default="must_call_tool",
                       help="Prompt key for generation")
    parser.add_argument("--generation-max-tokens", type=int, default=8192,
                       help="Max tokens for generation")
    parser.add_argument("--generation-reasoning-effort", type=str, default="medium",
                       choices=["none", "minimal", "low", "medium", "high"],
                       help="Reasoning effort for generation model (gpt-5 default: medium, gpt-5.1 default: none)")

    # Editing parameters
    parser.add_argument("--editing-model", required=True,
                       help="Model for generating descriptions (e.g., gpt-4o)")
    parser.add_argument("--editing-temperature", type=float, default=1.0,
                       help="Temperature for description generation")
    parser.add_argument("--editing-prompt-key", default="reflective",
                       help="Prompt key for description generation")
    parser.add_argument("--editing-max-tokens", type=int, default=8192,
                       help="Max tokens for description generation")
    parser.add_argument("--editing-reasoning-effort", type=str, default=None,
                       choices=["none", "minimal", "low", "medium", "high"],
                       help="Reasoning effort for editing model (gpt-5, o-series)")

    # Iteration control
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations to run")
    parser.add_argument("--stop-on-perfect", action="store_true",
                       help="Stop if accuracy reaches 100%")
    parser.add_argument("--stop-on-decline", action="store_true",
                       help="Stop if accuracy decreases")

    # Other options
    parser.add_argument("--output-dir", type=Path, default=Path("runs/bfcl/ours"),
                       help="Base output directory for runs (default: runs/bfcl/ours)")
    parser.add_argument("--num-queries", type=int,
                       help="Limit number of test queries per iteration")
    parser.add_argument("--output-summary", type=str,
                       help="Path to save iteration summary JSON")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility (default: 0)")

    args = parser.parse_args()

    # Prepare generation and editing arguments
    generation_args = {
        "model": args.generation_model,
        "together": args.generation_together,  # Include Together AI flag
        "temperature": args.generation_temperature,
        "top_p": args.generation_top_p,
        "tool_choice": args.generation_tool_choice,
        "prompt_key": args.generation_prompt_key,
        "max_tokens": args.generation_max_tokens,
        "reasoning_effort": args.generation_reasoning_effort,
        "seed": args.seed
    }

    if args.num_queries:
        generation_args["num_queries"] = args.num_queries

    editing_args = {
        "model": args.editing_model,
        "temperature": args.editing_temperature,
        "prompt_key": args.editing_prompt_key,
        "max_tokens": args.editing_max_tokens,
        "reasoning_effort": args.editing_reasoning_effort
    }

    # Check if we're continuing from an improvement
    is_improvement, _, current_version = detect_improvement_context(Path(args.config_source))

    if is_improvement:
        logger.info(f"Continuing from improvement v{current_version}")
        # Validate that hyperparameters match the path structure
        try:
            validate_hyperparams_match(Path(args.config_source), generation_args, editing_args)
            logger.info("Hyperparameters validated successfully")
        except ValueError as e:
            logger.error(str(e))
            return 1
        start_iteration = current_version + 1
    else:
        logger.info("Starting fresh iteration from base config")
        start_iteration = 0
        args.iterations += 1

    # Track metrics across iterations
    all_metrics = []
    current_config = args.config_source
    previous_accuracy = None

    # Run iterations
    for i in range(start_iteration, start_iteration + args.iterations):
        success, next_config, metrics = run_iteration(
            current_config,
            generation_args,
            editing_args,
            i,
            output_dir=args.output_dir
        )

        all_metrics.append(metrics)

        if not success:
            logger.error(f"Iteration {i} failed. Stopping.")
            break

        # Check for convergence
        if metrics.get("converged", False):
            logger.info(f"\n{'='*60}")
            logger.info("🎉 CONVERGENCE ACHIEVED - Stopping iterations")
            logger.info(f"All tests have converged at iteration {i}")
            logger.info(f"{'='*60}\n")
            break

        # Check stopping conditions
        if args.stop_on_perfect and metrics.get("accuracy", 0) >= 1.0:
            logger.info(f"Perfect accuracy achieved! Stopping.")
            break

        if args.stop_on_decline and previous_accuracy is not None:
            if metrics.get("accuracy", 0) < previous_accuracy:
                logger.info(f"Accuracy declined from {previous_accuracy:.2%} to {metrics.get('accuracy', 0):.2%}. Stopping.")
                break

        previous_accuracy = metrics.get("accuracy")
        current_config = next_config

    # Save summary if requested
    if args.output_summary:
        summary = {
            "config_source": args.config_source,
            "generation_args": generation_args,
            "editing_args": editing_args,
            "iterations_completed": len(all_metrics),
            "iterations": all_metrics,
            "final_config": current_config if current_config != args.config_source else None,
            "final_accuracy": all_metrics[-1].get("accuracy") if all_metrics else None
        }

        with open(args.output_summary, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {args.output_summary}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Iteration Summary:")
    logger.info(f"{'='*60}")
    for metric in all_metrics:
        accuracy = metric.get("accuracy", "N/A")
        if isinstance(accuracy, float):
            accuracy = f"{accuracy:.2%}"
        converged = " (CONVERGED)" if metric.get("converged", False) else ""
        logger.info(f"Iteration {metric['iteration']}: {accuracy}{converged}")

    # Check if we converged
    converged = any(m.get("converged", False) for m in all_metrics)
    if converged:
        logger.info(f"\n✅ Process completed successfully with convergence")

    return 0 if all(m.get("accuracy") is not None for m in all_metrics) else 1


if __name__ == "__main__":
    sys.exit(main())