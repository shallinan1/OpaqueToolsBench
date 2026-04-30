#!/usr/bin/env python3
"""Iterative improvement loop for BrowseCompPlus shared-tool descriptions.

Pipeline per iteration:
1) run.py
2) evaluate.py
3) generate_improved_descriptions.py (minibatch trajectory analysis + synthesis)
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.datasets.BrowseCompPlus.prompts import (
    BROWSECOMP_AGENT_PROMPTS,
    BROWSECOMP_SYNTHESIS_DESCRIPTION_PROMPTS,
    BROWSECOMP_TOOL_DESCRIPTION_PROMPTS,
    resolve_prompt_key,
)
from src.datasets.BrowseCompPlus.utils.path_utils import (
    build_output_folder,
    config_uses_faiss,
    create_editing_dirname,
    detect_improvement_context,
    get_base_run_path,
    parse_config_name,
    validate_hyperparams_match,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("o") or "gpt-5" in model


def _find_latest_base_results(result_dir: Path) -> Optional[Path]:
    versioned = sorted(
        result_dir.glob("v*_results.json"),
        key=lambda p: int(p.name.split("_")[0][1:]) if p.name.startswith("v") else -1,
    )
    return versioned[-1] if versioned else None


def _scored_for_results(results_file: Path) -> Path:
    if results_file.name == "results.json":
        return results_file.parent / "scored.json"
    version = results_file.name.split("_")[0]
    return results_file.parent / f"{version}_scored.json"


def run_generation(config_source: str, generation_args: Dict) -> Tuple[bool, Optional[Path], Optional[Path]]:
    """Run generation step and return (ok, result_dir, results_file)."""
    config_path = Path(config_source)
    is_improvement, improvement_base, improvement_version = detect_improvement_context(config_path)

    if is_improvement:
        result_dir = improvement_base / f"v{improvement_version}"
        results_file = result_dir / "results.json"
        if results_file.exists():
            logger.info("Found existing improvement results at %s", results_file)
            return True, result_dir, results_file
    else:
        ns = argparse.Namespace(**generation_args)
        ns._resolved_prompt_key = resolve_prompt_key(
            prompt_key=generation_args.get("prompt_key"),
            include_get_document=bool(generation_args.get("include_get_document")),
        )
        ns._uses_faiss = config_uses_faiss(config_path)
        config_name = parse_config_name(config_path)
        result_dir = build_output_folder(ns, config_name, mode="shared_tools")
        v0_results = result_dir / "v0_results.json"
        if v0_results.exists():
            logger.info("Found existing base results at %s", v0_results)
            return True, result_dir, v0_results
        latest_existing = _find_latest_base_results(result_dir)
        if latest_existing is not None:
            logger.info("Found existing base results at %s", latest_existing)
            return True, result_dir, latest_existing

    cmd = [
        sys.executable,
        "-m",
        "src.datasets.BrowseCompPlus.run",
        "--config-source",
        config_source,
        "--model",
        generation_args["model"],
        "--tool-choice",
        generation_args["tool_choice"],
        "--max-tokens",
        str(generation_args["max_tokens"]),
        "--k",
        str(generation_args["k"]),
        "--snippet-max-tokens",
        str(generation_args["snippet_max_tokens"]),
        "--max-iterations",
        str(generation_args["max_iterations"]),
        "--output-dir",
        generation_args["output_dir"],
    ]

    if generation_args.get("num_queries") is not None:
        cmd.extend(["--num-queries", str(generation_args["num_queries"])])

    if generation_args.get("prompt_key"):
        cmd.extend(["--prompt-key", generation_args["prompt_key"]])

    if generation_args.get("include_get_document"):
        cmd.append("--include-get-document")

    if generation_args.get("hide_urls"):
        cmd.append("--hide-urls")

    if generation_args.get("split"):
        cmd.extend(["--split", generation_args["split"]])

    if generation_args.get("faiss_index_path"):
        cmd.extend(["--faiss-index-path", generation_args["faiss_index_path"]])

    if generation_args.get("categories_file"):
        cmd.extend(["--categories-file", generation_args["categories_file"]])

    if generation_args.get("url_mapping_file"):
        cmd.extend(["--url-mapping-file", generation_args["url_mapping_file"]])

    if generation_args.get("index_path"):
        cmd.extend(["--index-path", generation_args["index_path"]])

    if _is_reasoning_model(generation_args["model"]):
        cmd.extend(["--reasoning-effort", generation_args["reasoning_effort"]])
    else:
        cmd.extend(["--temperature", str(generation_args["temperature"])])
        cmd.extend(["--top-p", str(generation_args["top_p"])])

    logger.info("Running generation: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("Generation failed with code %d", result.returncode)
        return False, None, None

    # Resolve produced files
    if is_improvement:
        results_file = result_dir / "results.json"
    else:
        latest = _find_latest_base_results(result_dir)
        results_file = latest

    if results_file is None or not results_file.exists():
        logger.error("Could not locate generated results file in %s", result_dir)
        return False, None, None

    return True, result_dir, results_file


def run_evaluation(results_file: Path) -> Tuple[bool, Optional[Path]]:
    """Run evaluation and return scored file path."""
    scored_file = _scored_for_results(results_file)

    if scored_file.exists():
        logger.info("Found existing scored file at %s", scored_file)
        return True, scored_file

    cmd = [
        sys.executable,
        "-m",
        "src.datasets.BrowseCompPlus.evaluate",
        "--results-file",
        str(results_file),
    ]

    logger.info("Running evaluation: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("Evaluation failed with code %d", result.returncode)
        return False, None

    if scored_file.exists():
        return True, scored_file

    logger.error("Scored file missing after evaluation: %s", scored_file)
    return False, None


def run_description_generation(result_dir: Path, editing_args: Dict) -> Tuple[bool, Optional[Path]]:
    """Generate improved config from result directory and return config path."""
    cmd = [
        sys.executable,
        "-m",
        "src.datasets.BrowseCompPlus.generate_improved_descriptions",
        "--result-dir",
        str(result_dir),
        "--prompt-type",
        editing_args["prompt_type"],
        "--model",
        editing_args["model"],
        "--temperature",
        str(editing_args["temperature"]),
        "--max-tokens",
        str(editing_args["max_tokens"]),
        "--num-trajectories-batch",
        str(editing_args["num_trajectories_batch"]),
        "--synthesis-prompt-key",
        editing_args["synthesis_prompt_key"],
        "--synthesis-temperature",
        str(editing_args["synthesis_temperature"]),
        "--synthesis-max-tokens",
        str(editing_args["synthesis_max_tokens"]),
    ]

    if _is_reasoning_model(editing_args["model"]) and editing_args.get("reasoning_effort"):
        cmd.extend(["--reasoning-effort", editing_args["reasoning_effort"]])
    if editing_args.get("synthesis_model"):
        cmd.extend(["--synthesis-model", editing_args["synthesis_model"]])
    if editing_args.get("synthesis_reasoning_effort"):
        cmd.extend(["--synthesis-reasoning-effort", editing_args["synthesis_reasoning_effort"]])

    logger.info("Running description generation: %s", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error("Description generation failed with code %d", result.returncode)
        return False, None

    base_run_path = get_base_run_path(result_dir) or result_dir
    editing_dirname = create_editing_dirname(
        model=editing_args["model"],
        temperature=editing_args["temperature"],
        prompt_key=editing_args["prompt_type"],
        max_tokens=editing_args["max_tokens"],
        reasoning_effort=editing_args.get("reasoning_effort"),
        num_trajectories_batch=editing_args.get("num_trajectories_batch"),
        synthesis_model=editing_args.get("synthesis_model"),
        synthesis_temperature=editing_args.get("synthesis_temperature"),
        synthesis_prompt_key=editing_args.get("synthesis_prompt_key"),
        synthesis_max_tokens=editing_args.get("synthesis_max_tokens"),
        synthesis_reasoning_effort=editing_args.get("synthesis_reasoning_effort"),
    )

    improvements_dir = base_run_path / "improvements" / editing_dirname
    versions = sorted(
        [d for d in improvements_dir.glob("v*") if d.is_dir() and d.name[1:].isdigit()],
        key=lambda p: int(p.name[1:]),
    )
    if not versions:
        logger.error("No improvement versions found in %s", improvements_dir)
        return False, None

    config_path = versions[-1] / "config.json"
    if not config_path.exists():
        logger.error("Improved config missing: %s", config_path)
        return False, None

    return True, config_path


def extract_metrics(scored_file: Path) -> Dict:
    """Extract key metrics from BrowseCompPlus scored output."""
    with open(scored_file, "r") as f:
        scored = json.load(f)

    summary = scored.get("summary", {})
    return {
        "accuracy": summary.get("accuracy"),
        "correct": summary.get("correct"),
        "total": summary.get("total"),
        "avg_tool_calls": summary.get("tool_usage", {}).get("avg_calls_per_query"),
        "avg_citation_precision": summary.get("citation_metrics", {}).get("avg_precision"),
        "avg_retrieval_recall": summary.get("retrieval", {}).get("avg_recall"),
    }


def run_iteration(
    config_source: str,
    generation_args: Dict,
    editing_args: Dict,
    iteration_num: int,
) -> Tuple[bool, Optional[str], Dict]:
    """Run one full run->evaluate->improve iteration."""
    metrics = {"iteration": iteration_num, "start_time": datetime.utcnow().isoformat()}

    logger.info("\n%s", "=" * 60)
    logger.info("Starting iteration %s", iteration_num)
    logger.info("Config source: %s", config_source)
    logger.info("%s", "=" * 60)

    success, result_dir, results_file = run_generation(config_source, generation_args)
    if not success:
        metrics["failed_at"] = "generation"
        return False, None, metrics

    metrics["result_dir"] = str(result_dir)
    metrics["results_file"] = str(results_file)

    success, scored_file = run_evaluation(results_file)
    if not success:
        metrics["failed_at"] = "evaluation"
        return False, None, metrics

    metrics["scored_file"] = str(scored_file)
    metrics.update(extract_metrics(scored_file))

    success, improved_config = run_description_generation(result_dir, editing_args)
    if not success:
        metrics["failed_at"] = "description_generation"
        return False, None, metrics

    metrics["improved_config"] = str(improved_config)
    metrics["end_time"] = datetime.utcnow().isoformat()
    return True, str(improved_config), metrics


def main():
    parser = argparse.ArgumentParser(description="Iteratively improve BrowseCompPlus tool descriptions")

    parser.add_argument("--config-source", required=True, help="Initial config or previous iteration config")

    # Generation args
    parser.add_argument("--generation-model", required=True, help="Model used in run.py")
    parser.add_argument("--generation-temperature", type=float, default=1.0, help="Temperature for non-reasoning models")
    parser.add_argument("--generation-top-p", type=float, default=1.0, help="Top-p for non-reasoning models")
    parser.add_argument(
        "--generation-tool-choice",
        default="auto",
        choices=["required", "auto", "none"],
        help="Tool choice mode for run.py",
    )
    parser.add_argument(
        "--generation-reasoning-effort",
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for reasoning generation models",
    )
    parser.add_argument(
        "--generation-prompt-key",
        default=None,
        choices=sorted(BROWSECOMP_AGENT_PROMPTS.keys()),
        help="Prompt key from src/datasets/BrowseCompPlus/prompts.py.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Number of queries to run each iteration (default: all queries)",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k search results")
    parser.add_argument("--snippet-max-tokens", type=int, default=512, help="Max snippet tokens")
    parser.add_argument("--generation-max-tokens", type=int, default=10000, help="Max completion tokens per generation response")
    parser.add_argument("--generation-max-iterations", type=int, default=50, help="Max tool-call rounds per query")
    parser.add_argument("--include-get-document", action="store_true", help="Include get_document tool")
    parser.add_argument("--hide-urls", action="store_true", help="Hide URLs from search results")

    parser.add_argument("--split", type=str, choices=["train", "test"], default=None,
                        help="Load queries from train or test split. If omitted, uses all queries.")
    parser.add_argument("--faiss-index-path", type=str, default=None, help="Optional FAISS index glob")
    parser.add_argument("--categories-file", type=str, default=None, help="Optional categories JSON override")
    parser.add_argument("--url-mapping-file", type=str, default=None, help="Optional docid->URL mapping JSON override")
    parser.add_argument("--index-path", type=str, default=None, help="Optional BM25 index path override")

    # Editing args
    parser.add_argument("--editing-model", required=True, help="Model for description improvement")
    parser.add_argument("--editing-temperature", type=float, default=0.7, help="Temperature for non-reasoning editing models")
    parser.add_argument(
        "--editing-prompt-type",
        default="detailed_v2",
        choices=sorted(BROWSECOMP_TOOL_DESCRIPTION_PROMPTS.keys()),
        help="Prompt strategy for minibatch trajectory analysis",
    )
    parser.add_argument("--editing-max-tokens", type=int, default=8192, help="Max tokens per minibatch analysis response")
    parser.add_argument(
        "--editing-reasoning-effort",
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for reasoning editing models",
    )
    parser.add_argument(
        "--num-trajectories-batch",
        type=int,
        default=10,
        help="Number of trajectories per minibatch analysis request",
    )
    parser.add_argument(
        "--synthesis-model",
        default=None,
        help="Synthesis model (defaults to --editing-model)",
    )
    parser.add_argument(
        "--synthesis-temperature",
        type=float,
        default=0.3,
        help="Temperature for non-reasoning synthesis models",
    )
    parser.add_argument(
        "--synthesis-prompt-key",
        default="v2",
        choices=sorted(BROWSECOMP_SYNTHESIS_DESCRIPTION_PROMPTS.keys()),
        help="Prompt strategy used for synthesis over minibatch responses",
    )
    parser.add_argument(
        "--synthesis-max-tokens",
        type=int,
        default=None,
        help="Max tokens for synthesis response (defaults to --editing-max-tokens)",
    )
    parser.add_argument(
        "--synthesis-reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for reasoning synthesis models",
    )

    # Loop control
    parser.add_argument("--iterations", type=int, default=3, help="Number of improvements to run")
    parser.add_argument("--stop-on-perfect", action="store_true", help="Stop when accuracy reaches 1.0")
    parser.add_argument("--stop-on-decline", action="store_true", help="Stop when accuracy declines")

    parser.add_argument("--output-dir", type=Path, default=Path("runs/BrowseCompPlus/tool_observer"), help="Output root")
    parser.add_argument("--output-summary", type=str, default=None, help="Optional summary JSON output")

    args = parser.parse_args()

    generation_args = {
        "model": args.generation_model,
        "temperature": args.generation_temperature,
        "top_p": args.generation_top_p,
        "tool_choice": args.generation_tool_choice,
        "reasoning_effort": args.generation_reasoning_effort,
        "prompt_key": args.generation_prompt_key,
        "split": args.split,
        "num_queries": args.num_queries,
        "k": args.k,
        "snippet_max_tokens": args.snippet_max_tokens,
        "max_tokens": args.generation_max_tokens,
        "max_iterations": args.generation_max_iterations,
        "include_get_document": args.include_get_document,
        "hide_urls": args.hide_urls,
        "faiss_index_path": args.faiss_index_path,
        "categories_file": args.categories_file,
        "url_mapping_file": args.url_mapping_file,
        "index_path": args.index_path,
        "output_dir": str(args.output_dir),
    }

    editing_args = {
        "model": args.editing_model,
        "temperature": args.editing_temperature,
        "prompt_type": args.editing_prompt_type,
        "max_tokens": args.editing_max_tokens,
        "reasoning_effort": args.editing_reasoning_effort,
        "num_trajectories_batch": args.num_trajectories_batch,
        "synthesis_model": args.synthesis_model or args.editing_model,
        "synthesis_temperature": args.synthesis_temperature,
        "synthesis_prompt_key": args.synthesis_prompt_key,
        "synthesis_max_tokens": args.synthesis_max_tokens or args.editing_max_tokens,
        "synthesis_reasoning_effort": (
            args.synthesis_reasoning_effort
            if args.synthesis_reasoning_effort is not None
            else args.editing_reasoning_effort
        ),
    }

    is_improvement, _, current_version = detect_improvement_context(Path(args.config_source))
    if is_improvement:
        logger.info("Continuing from improvement v%s", current_version)
        try:
            validate_hyperparams_match(Path(args.config_source), generation_args, editing_args)
            logger.info("Hyperparameters validated successfully")
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        start_iteration = current_version + 1
    else:
        logger.info("Starting from base config")
        start_iteration = 0
        # Match BFCL semantics: run base iteration (v0) plus N improvements.
        args.iterations += 1

    all_metrics = []
    current_config = args.config_source
    previous_accuracy = None

    for i in range(start_iteration, start_iteration + args.iterations):
        success, next_config, metrics = run_iteration(
            config_source=current_config,
            generation_args=generation_args,
            editing_args=editing_args,
            iteration_num=i,
        )

        all_metrics.append(metrics)

        if not success:
            logger.error("Iteration %s failed; stopping", i)
            break

        accuracy = metrics.get("accuracy")
        if args.stop_on_perfect and accuracy is not None and accuracy >= 1.0:
            logger.info("Perfect accuracy reached; stopping")
            break

        if (
            args.stop_on_decline
            and previous_accuracy is not None
            and accuracy is not None
            and accuracy < previous_accuracy
        ):
            logger.info("Accuracy declined from %.4f to %.4f; stopping", previous_accuracy, accuracy)
            break

        previous_accuracy = accuracy
        current_config = next_config

    if args.output_summary:
        summary = {
            "config_source": args.config_source,
            "generation_args": generation_args,
            "editing_args": editing_args,
            "iterations_completed": len(all_metrics),
            "iterations": all_metrics,
            "final_config": current_config if current_config != args.config_source else None,
            "final_accuracy": all_metrics[-1].get("accuracy") if all_metrics else None,
        }
        with open(args.output_summary, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved to %s", args.output_summary)

    logger.info("\n%s", "=" * 60)
    logger.info("Iteration summary")
    logger.info("%s", "=" * 60)
    for metric in all_metrics:
        logger.info(
            "Iteration %s: accuracy=%s total=%s avg_tool_calls=%s",
            metric.get("iteration"),
            metric.get("accuracy"),
            metric.get("total"),
            metric.get("avg_tool_calls"),
        )

    return 0 if all(m.get("accuracy") is not None for m in all_metrics) else 1


if __name__ == "__main__":
    sys.exit(main())
