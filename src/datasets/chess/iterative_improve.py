#!/usr/bin/env python3
"""
Iterative improvement script for chess tool descriptions.

Automates the description improvement loop by orchestrating:
1. Game playing (run.py) - Collect trajectories with current descriptions
2. Evaluation (evaluate.py) - Score positions with Stockfish
3. Description generation (generate_descriptions.py) - LLM analyzes tool usage
4. Synthesis (synthesize_descriptions.py) - Combine analyses into final descriptions
5. Repeat with improved config

Directory structure created:
    runs/chess/tool_observer/shared_tools/{config_name}/{split}/{hyperparams}/{white}/vs_{black}/
    ├── v0_trajectories.json, v0_scored.json, v0_metadata.json
    └── improvements/{editing_hypers}/
        ├── v1/llm_responses.json, config.json
        ├── v2/...
        └── ...

Example - Start fresh iteration:
    python -m src.datasets.chess.iterative_improve \\
        --config-source src/datasets/chess/shared_tools/elo_tools_obfuscated.json \\
        --generation-model gpt-5 \\
        --editing-model gpt-5 \\
        --editing-prompt-key detailed \\
        --iterations 10 \\
        --black-type elo_1800

Example - Continue from existing improvement:
    python -m src.datasets.chess.iterative_improve \\
        --config-source runs/chess/tool_observer/.../improvements/gpt-5_detailed_0.7_8192/v4/config.json \\
        --generation-model gpt-5 \\
        --editing-model gpt-5 \\
        --iterations 1
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess
from datetime import datetime

from src.datasets.chess.utils.path_utils import (
    parse_config_name,
    create_generation_dirname,
    create_editing_dirname,
    detect_improvement_context,
    get_base_run_path,
    build_output_folder,
    validate_hyperparams_match,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_generation(config_source: str, generation_args: Dict) -> Tuple[bool, Optional[Path]]:
    """Run the generation step (run.py).

    Returns:
        (success, result_dir) where result_dir is the output directory containing trajectories.
        For base runs: result_dir contains v0_trajectories.json
        For improvement runs: result_dir is the version dir (e.g., v1/) containing trajectories.json
    """
    config_path = Path(config_source)

    # Check if this is an improvement config — if so, run.py will save directly in the version dir
    is_improvement, improvement_base, improvement_version = detect_improvement_context(config_path)
    if is_improvement:
        # run.py will save trajectories.json in the version directory
        result_dir = improvement_base / f"v{improvement_version}"
        trajectories_file = result_dir / "trajectories.json"
    else:
        # Predict result_dir by building what run.py will produce
        # We need to construct an args namespace matching what build_output_folder expects
        ns = argparse.Namespace(**generation_args)
        config_name = parse_config_name(config_path)
        result_dir = build_output_folder(ns, config_name, "tool_set")
        trajectories_file = result_dir / "v0_trajectories.json"

    # Check if trajectories already exist
    if trajectories_file.exists():
        logger.info(f"Found existing trajectories at {trajectories_file}")
        logger.info("Skipping generation step - using existing results")
        return True, result_dir

    # Build run.py command with correct flags
    cmd = [
        sys.executable, "-m", "src.datasets.chess.run",
        "--shared-tools", config_source,
        "--model", generation_args["model"],
        "--tool-choice", generation_args.get("tool_choice", "required"),
        "--prompt-key", generation_args["prompt_key"],
        "--white-type", generation_args.get("white_type", "agent"),
        "--black-type", generation_args["black_type"],
        "--max-moves", str(generation_args.get("max_moves", 120)),
        "--num-trajectories", str(generation_args.get("num_trajectories", 3)),
        "--max-tokens", str(generation_args.get("max_tokens", 8192)),
        "--seed", str(generation_args.get("seed", 0)),
        "--output-dir", generation_args.get("output_dir", "runs/chess/tool_observer"),
    ]

    # Temperature (only for non-reasoning models)
    model = generation_args["model"]
    if not (model.startswith('o') or 'gpt-5' in model):
        cmd.extend(["--temperature", str(generation_args.get("temperature", 1.0))])

    # Reasoning effort (only for reasoning models)
    if model.startswith('o') or 'gpt-5' in model:
        cmd.extend(["--reasoning-effort", generation_args.get("reasoning_effort", "medium")])

    if generation_args.get("split"):
        cmd.extend(["--split", generation_args["split"]])

    if generation_args.get("num_queries"):
        cmd.extend(["--num-queries", str(generation_args["num_queries"])])

    if generation_args.get("mirror"):
        cmd.append("--mirror")

    if generation_args.get("include_history"):
        cmd.append("--include-history")

    logger.info(f"Running generation: {' '.join(cmd)}")
    logger.info("=" * 60)

    try:
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            logger.error(f"Generation failed with return code {result.returncode}")
            return False, None
    except Exception as e:
        logger.error(f"Exception running generation: {e}")
        return False, None

    return True, result_dir


def run_evaluation(result_dir: Path) -> Tuple[bool, Optional[Path]]:
    """Run the evaluation step (evaluate.py).

    Auto-detects whether this is a base run or improvement run based on
    which trajectories file exists in result_dir.

    Returns:
        (success, scored_file_path)
    """
    # Auto-detect: improvement runs have trajectories.json, base runs have v0_trajectories.json
    if (result_dir / "trajectories.json").exists():
        trajectories_file = result_dir / "trajectories.json"
        scored_file = result_dir / "scored.json"
    else:
        trajectories_file = result_dir / "v0_trajectories.json"
        scored_file = result_dir / "v0_scored.json"

    # Check if scored file already exists
    if scored_file.exists():
        logger.info(f"Found existing scored file at {scored_file}")
        logger.info("Skipping evaluation step - using existing scores")
        return True, scored_file

    # Check if trajectories file exists
    if not trajectories_file.exists():
        logger.error(f"Trajectories file not found: {trajectories_file}")
        return False, None

    cmd = [
        sys.executable, "-m", "src.datasets.chess.evaluate",
        "--input_file", str(trajectories_file)
    ]

    logger.info(f"Running evaluation: {' '.join(cmd)}")
    logger.info("=" * 60)

    try:
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            logger.error(f"Evaluation failed with return code {result.returncode}")
            return False, None
    except Exception as e:
        logger.error(f"Exception running evaluation: {e}")
        return False, None

    if scored_file.exists():
        logger.info(f"Evaluation complete: {scored_file}")
        return True, scored_file
    else:
        logger.error(f"Scored file not found after evaluation: {scored_file}")
        return False, None


def run_description_generation(result_dir: Path, scored_file: Path, editing_args: Dict) -> Tuple[bool, Optional[Path]]:
    """Generate improved descriptions using the two-step process.

    Step 1: generate_descriptions.py — LLM analyzes trajectory batches
    Step 2: synthesize_descriptions.py — Combine batch analyses into final config

    Args:
        result_dir: Directory containing trajectories/scored files.
            For base runs: the base run directory (contains v0_scored.json)
            For improvement runs: the version directory (e.g., v1/, contains scored.json)
        scored_file: Path to the scored file
        editing_args: Editing hyperparameters

    Returns:
        (success, improved_config_path)
    """
    # Step 1: Generate LLM responses
    cmd_generate = [
        sys.executable, "-m", "src.datasets.chess.generate_descriptions",
        "--result-dir", str(scored_file),
        "--model", editing_args["model"],
        "--temperature", str(editing_args["temperature"]),
        "--prompt-key", editing_args["prompt_key"],
        "--max-tokens", str(editing_args.get("max_tokens", 8192))
    ]

    if "num_trajectories_batch" in editing_args:
        cmd_generate.extend(["--num-trajectories-batch", str(editing_args["num_trajectories_batch"])])
    if editing_args.get("show_agent_values", False):
        cmd_generate.append("--show-agent-values")
    if editing_args.get("reasoning_effort") is not None:
        cmd_generate.extend(["--reasoning-effort", editing_args["reasoning_effort"]])

    logger.info(f"Step 1 - Generating LLM responses: {' '.join(cmd_generate)}")
    logger.info("=" * 60)

    try:
        result = subprocess.run(cmd_generate, text=True)
        if result.returncode != 0:
            logger.error(f"Description generation failed with return code {result.returncode}")
            return False, None
    except Exception as e:
        logger.error(f"Exception running description generation: {e}")
        return False, None

    # Find the version directory that was just created
    # generate_descriptions.py saves improvements under the base run path
    editing_dirname = create_editing_dirname(
        editing_args["model"],
        editing_args["temperature"],
        editing_args["prompt_key"],
        editing_args.get("max_tokens", 8192),
        editing_args.get("show_agent_values", False),
        editing_args.get("reasoning_effort")
    )
    # If result_dir is inside an improvements dir, get the base run path
    base_run_path = get_base_run_path(result_dir)
    if base_run_path is None:
        base_run_path = result_dir
    improvements_dir = base_run_path / "improvements" / editing_dirname

    if not improvements_dir.exists():
        logger.error(f"Improvements directory not found: {improvements_dir}")
        return False, None

    # Find the latest version directory
    versions = []
    for item in improvements_dir.iterdir():
        if item.is_dir() and item.name.startswith('v'):
            try:
                versions.append(int(item.name[1:]))
            except ValueError:
                pass

    if not versions:
        logger.error(f"No version directories found in {improvements_dir}")
        return False, None

    latest_version = max(versions)
    version_dir = improvements_dir / f"v{latest_version}"

    llm_responses_file = version_dir / "llm_responses.json"
    if not llm_responses_file.exists():
        logger.error(f"LLM responses not found: {llm_responses_file}")
        return False, None

    logger.info(f"LLM responses generated: {llm_responses_file}")

    # Step 2: Synthesize descriptions
    synthesis_args = editing_args.get("synthesis", {})
    cmd_synthesize = [
        sys.executable, "-m", "src.datasets.chess.synthesize_descriptions",
        "--response-dir", str(version_dir),
        "--model", synthesis_args.get("model", editing_args["model"]),
        "--temperature", str(synthesis_args.get("temperature", 0.3)),
        "--prompt-key", synthesis_args.get("prompt_key", "v1"),
        "--max-tokens", str(synthesis_args.get("max_tokens", editing_args.get("max_tokens", 8192)))
    ]

    # Pass reasoning effort for reasoning models
    synthesis_re = synthesis_args.get("reasoning_effort", editing_args.get("reasoning_effort"))
    if synthesis_re is not None:
        cmd_synthesize.extend(["--reasoning-effort", synthesis_re])

    logger.info(f"Step 2 - Synthesizing descriptions: {' '.join(cmd_synthesize)}")
    logger.info("=" * 60)

    try:
        result = subprocess.run(cmd_synthesize, text=True)
        if result.returncode != 0:
            logger.error(f"Synthesis failed with return code {result.returncode}")
            return False, None
    except Exception as e:
        logger.error(f"Exception running synthesis: {e}")
        return False, None

    config_path = version_dir / "config.json"
    if config_path.exists():
        logger.info(f"Synthesis complete: {config_path}")
        return True, config_path
    else:
        logger.error(f"Config not found after synthesis: {config_path}")
        return False, None


def extract_metrics(scored_file: Path) -> Dict:
    """Extract evaluation metrics from scored file."""
    with open(scored_file, 'r') as f:
        scored_data = json.load(f)

    # Scored files have a 'trajectories' key
    if isinstance(scored_data, dict) and 'trajectories' in scored_data:
        trajectories = scored_data['trajectories']
    elif isinstance(scored_data, list):
        trajectories = scored_data
    else:
        logger.warning(f"Unexpected scored file format")
        return {}

    total_games = len(trajectories)
    if total_games == 0:
        return {"total_games": 0}

    white_wins = sum(1 for g in trajectories if g.get("result") == "1-0")
    draws = sum(1 for g in trajectories if g.get("result") == "1/2-1/2")
    black_wins = sum(1 for g in trajectories if g.get("result") == "0-1")

    win_rate = white_wins / total_games
    draw_rate = draws / total_games

    # Average centipawn value across all evaluated positions
    total_cp = 0
    cp_count = 0
    for game in trajectories:
        for move in game.get("moves", []):
            if "board_value_cp" in move:
                total_cp += move["board_value_cp"]
                cp_count += 1

    avg_cp = total_cp / cp_count if cp_count > 0 else 0

    return {
        "total_games": total_games,
        "white_wins": white_wins,
        "draws": draws,
        "black_wins": black_wins,
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "avg_centipawn": avg_cp
    }


def run_iteration(
    config_source: str,
    generation_args: Dict,
    editing_args: Dict,
    iteration_num: int
) -> Tuple[bool, Optional[str], Dict]:
    """Run one complete iteration of the improvement loop.

    Returns:
        (success, next_config_source, metrics)
    """
    metrics = {
        "iteration": iteration_num,
        "start_time": datetime.utcnow().isoformat()
    }

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting iteration {iteration_num}")
    logger.info(f"Config source: {config_source}")
    logger.info(f"{'=' * 60}")

    # Step 1: Play games
    success, result_dir = run_generation(config_source, generation_args)
    if not success:
        logger.error(f"Generation failed for iteration {iteration_num}")
        metrics["failed_at"] = "generation"
        return False, None, metrics

    metrics["result_dir"] = str(result_dir)

    # Step 2: Evaluate with Stockfish
    success, scored_file = run_evaluation(result_dir)
    if not success:
        logger.error(f"Evaluation failed for iteration {iteration_num}")
        metrics["failed_at"] = "evaluation"
        return False, None, metrics

    # Extract metrics
    eval_metrics = extract_metrics(scored_file)
    metrics.update(eval_metrics)
    logger.info(f"Iteration {iteration_num} win rate: {eval_metrics.get('win_rate', 0):.2%}")

    # Step 3: Generate + synthesize improved descriptions
    success, improved_config = run_description_generation(result_dir, scored_file, editing_args)
    if not success:
        logger.error(f"Description generation failed for iteration {iteration_num}")
        metrics["failed_at"] = "description_generation"
        return False, None, metrics

    metrics["improved_config"] = str(improved_config)
    metrics["end_time"] = datetime.utcnow().isoformat()

    logger.info(f"Iteration {iteration_num} complete. Next config: {improved_config}")
    return True, str(improved_config), metrics


def main():
    parser = argparse.ArgumentParser(
        description="Iteratively improve chess tool descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config source
    parser.add_argument("--config-source", required=True,
                       help="Initial config file or improved config from previous iteration")

    # Generation parameters (game playing via run.py)
    parser.add_argument("--generation-model", required=True,
                       help="Model for playing chess (e.g., gpt-5)")
    parser.add_argument("--generation-temperature", type=float, default=1.0,
                       help="Temperature for generation (ignored for reasoning models)")
    parser.add_argument("--generation-tool-choice", default="required",
                       choices=["required", "auto", "none"])
    parser.add_argument("--generation-prompt-key", default="optimized_single",
                       help="Prompt key for agent during play (default: optimized_single, paper canonical)")
    parser.add_argument("--generation-max-tokens", type=int, default=8192,
                       help="Max tokens for generation")
    parser.add_argument("--generation-reasoning-effort", default="medium",
                       choices=["minimal", "low", "medium", "high"],
                       help="Reasoning effort for generation model (default: medium)")

    # Game configuration
    parser.add_argument("--white-type", default="agent",
                       help="White player type (default: agent)")
    parser.add_argument("--black-type", default="elo_1800",
                       help="Black player type (default: elo_1800)")
    parser.add_argument("--split", default="train",
                       help="Data split to use (default: train)")
    parser.add_argument("--num-queries", type=int, default=10,
                       help="Number of FEN positions to play (default: 10)")
    parser.add_argument("--num-trajectories", type=int, default=3,
                       help="Trajectories per position (default: 3)")
    parser.add_argument("--max-moves", type=int, default=120,
                       help="Max moves per game (default: 120)")
    parser.add_argument("--mirror", action="store_true",
                       help="Run each position with swapped colors")
    parser.add_argument("--include-history", action="store_true",
                       help="Include move/tool history in conversation context")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")

    # Editing parameters (description improvement)
    parser.add_argument("--editing-model", required=True,
                       help="Model for analyzing trajectories (e.g., gpt-5)")
    parser.add_argument("--editing-temperature", type=float, default=0.7,
                       help="Temperature for description generation")
    parser.add_argument("--editing-prompt-key", default="detailed",
                       help="Prompt key for description generation")
    parser.add_argument("--editing-max-tokens", type=int, default=8192,
                       help="Max tokens for description generation")
    parser.add_argument("--num-trajectories-batch", type=int, default=10,
                       help="Number of trajectories per LLM analysis request")
    parser.add_argument("--editing-reasoning-effort", type=str, default=None,
                       choices=["minimal", "low", "medium", "high"],
                       help="Reasoning effort for editing model (for gpt-5, o-series)")
    parser.add_argument("--show-agent-values", action="store_true",
                       help="Show board values after agent moves in trajectories")

    # Synthesis parameters
    parser.add_argument("--synthesis-model", type=str,
                       help="Model for synthesis (defaults to editing-model)")
    parser.add_argument("--synthesis-temperature", type=float, default=0.3,
                       help="Temperature for synthesis")
    parser.add_argument("--synthesis-prompt-key", default="v1",
                       help="Prompt key for synthesis")
    parser.add_argument("--synthesis-max-tokens", type=int,
                       help="Max tokens for synthesis (defaults to editing-max-tokens)")

    # Iteration control
    parser.add_argument("--iterations", type=int, default=3,
                       help="Number of iterations to run")
    parser.add_argument("--stop-on-perfect", action="store_true",
                       help="Stop if win rate reaches 100%%")
    parser.add_argument("--stop-on-decline", action="store_true",
                       help="Stop if win rate decreases")

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("runs/chess/tool_observer"),
                       help="Base output directory for runs (default: runs/chess/tool_observer)")
    parser.add_argument("--output-summary", type=str,
                       help="Path to save iteration summary JSON")

    args = parser.parse_args()

    # Build generation args dict matching what run.py expects
    generation_args = {
        "model": args.generation_model,
        "temperature": args.generation_temperature,
        "tool_choice": args.generation_tool_choice,
        "prompt_key": args.generation_prompt_key,
        "max_tokens": args.generation_max_tokens,
        "reasoning_effort": args.generation_reasoning_effort,
        "white_type": args.white_type,
        "black_type": args.black_type,
        "split": args.split,
        "num_queries": args.num_queries,
        "num_trajectories": args.num_trajectories,
        "max_moves": args.max_moves,
        "mirror": args.mirror,
        "include_history": args.include_history,
        "seed": args.seed,
        # build_output_folder needs output_dir
        "output_dir": str(args.output_dir),
    }

    editing_args = {
        "model": args.editing_model,
        "temperature": args.editing_temperature,
        "prompt_key": args.editing_prompt_key,
        "max_tokens": args.editing_max_tokens,
        "reasoning_effort": args.editing_reasoning_effort,
        "num_trajectories_batch": args.num_trajectories_batch,
        "show_agent_values": args.show_agent_values,
        "synthesis": {
            "model": args.synthesis_model or args.editing_model,
            "temperature": args.synthesis_temperature,
            "prompt_key": args.synthesis_prompt_key,
            "max_tokens": args.synthesis_max_tokens or args.editing_max_tokens
        }
    }

    # Check if we're continuing from an improvement
    is_improvement, _, current_version = detect_improvement_context(Path(args.config_source))

    if is_improvement:
        logger.info(f"Continuing from improvement v{current_version}")
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
        args.iterations += 1  # iteration 0 is the base run, so add 1 to get the requested number of improvements

    # Track metrics across iterations
    all_metrics = []
    current_config = args.config_source
    previous_win_rate = None

    for i in range(start_iteration, start_iteration + args.iterations):
        success, next_config, metrics = run_iteration(
            current_config,
            generation_args,
            editing_args,
            i
        )

        all_metrics.append(metrics)

        if not success:
            logger.error(f"Iteration {i} failed. Stopping.")
            break

        # Check stopping conditions
        if args.stop_on_perfect and metrics.get("win_rate", 0) >= 1.0:
            logger.info("Perfect win rate achieved! Stopping.")
            break

        if args.stop_on_decline and previous_win_rate is not None:
            if metrics.get("win_rate", 0) < previous_win_rate:
                logger.info(f"Win rate declined from {previous_win_rate:.2%} to {metrics.get('win_rate', 0):.2%}. Stopping.")
                break

        previous_win_rate = metrics.get("win_rate")
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
            "final_win_rate": all_metrics[-1].get("win_rate") if all_metrics else None
        }

        with open(args.output_summary, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {args.output_summary}")

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Iteration Summary:")
    logger.info(f"{'=' * 60}")
    for metric in all_metrics:
        win_rate = metric.get("win_rate", "N/A")
        if isinstance(win_rate, float):
            win_rate = f"{win_rate:.2%}"
        logger.info(f"Iteration {metric['iteration']}: Win rate {win_rate}, Avg CP: {metric.get('avg_centipawn', 0):.1f}")

    return 0 if all(m.get("win_rate") is not None for m in all_metrics) else 1


if __name__ == "__main__":
    sys.exit(main())
