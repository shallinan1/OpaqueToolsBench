#!/usr/bin/env python3
"""Generate improved BrowseCompPlus tool descriptions from run outputs.

Improvements are generated in two phases:
1) Batch analysis over minibatches of trajectories.
2) Synthesis across all batch analyses into final descriptions.

Saves improved configs to:
  .../improvements/{editing_hypers}/vN/config.json
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from src.datasets.BrowseCompPlus.prompts import (
    BROWSECOMP_SYNTHESIS_DESCRIPTION_PROMPTS,
    BROWSECOMP_TOOL_DESCRIPTION_PROMPTS,
)
from src.datasets.BrowseCompPlus.utils.path_utils import (
    create_editing_dirname,
    get_base_run_path,
    get_next_version,
)
from src.generation_utils.openai_parallel_generate import (
    default_request_url,
    openai_parallel_generate,
    requests_url_dict,
)
from src.generation_utils.rate_limits import get_rate_limit, get_token_limit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 700
MAX_ANSWER_CHARS = 500
MAX_OUTPUT_CHARS = 1400
MAX_TOOL_CALLS_PER_TRAJECTORY = 10


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("o") or "gpt-5" in model


def _extract_version(filename: str) -> Optional[int]:
    if filename.startswith("v") and filename.endswith("_results.json"):
        try:
            return int(filename[1 : filename.index("_")])
        except ValueError:
            return None
    return None


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + "... [truncated]"


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _build_api_request(
    model: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    metadata: Optional[Dict] = None,
) -> Dict:
    req = {
        "model": model,
        "messages": messages,
        "metadata": metadata or {},
    }

    if _is_reasoning_model(model):
        req["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            req["reasoning_effort"] = reasoning_effort
    else:
        req["temperature"] = temperature
        req["max_tokens"] = max_tokens

    return req


def _extract_response_content(response_payload: Dict) -> str:
    return response_payload.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def resolve_run_files(result_dir: Path, version: Optional[int] = None) -> Tuple[Path, Optional[Path], Path]:
    """Resolve results/scored/metadata files from base or improvement result directories."""
    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    improvement_results = result_dir / "results.json"
    if improvement_results.exists():
        scored_file = result_dir / "scored.json"
        metadata_file = result_dir / "metadata.json"
        return improvement_results, scored_file if scored_file.exists() else None, metadata_file

    versioned_results = list(result_dir.glob("v*_results.json"))
    if not versioned_results:
        raise FileNotFoundError(f"No results file found in {result_dir}")

    if version is not None:
        results_file = result_dir / f"v{version}_results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Versioned results file not found: {results_file}")
        selected_version = version
    else:
        versioned_results.sort(key=lambda p: _extract_version(p.name) or -1)
        results_file = versioned_results[-1]
        selected_version = _extract_version(results_file.name)

    scored_candidate = result_dir / f"v{selected_version}_scored.json"
    scored_file = scored_candidate if scored_candidate.exists() else None

    metadata_candidate = result_dir / f"v{selected_version}_metadata.json"
    if metadata_candidate.exists():
        metadata_file = metadata_candidate
    else:
        fallback = result_dir / "metadata.json"
        if not fallback.exists():
            raise FileNotFoundError(f"Metadata file not found for {results_file}")
        metadata_file = fallback

    return results_file, scored_file, metadata_file


def load_config_from_metadata(metadata_file: Path) -> Tuple[Dict, Path]:
    """Load config used for the run from metadata."""
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    if "config_source" in metadata:
        config_path = Path(metadata["config_source"])
    elif "config_file" in metadata:
        config_path = Path(metadata["config_file"])
    else:
        raise ValueError(f"Cannot determine config source from metadata: {metadata_file}")

    if not config_path.exists() and not config_path.is_absolute():
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f), config_path


def _build_outcomes_by_query(scored: Dict) -> Dict[str, Dict]:
    outcomes: Dict[str, Dict] = {}
    for item in scored.get("detailed_evaluations", []):
        qid = str(item.get("query_id"))
        outcomes[qid] = {
            "correct": item.get("correct"),
            "num_tool_calls": item.get("num_tool_calls"),
            "retrieval_recall": item.get("retrieval_recall"),
            "citation_metrics": item.get("citation_metrics", {}),
        }
    return outcomes


def _format_trajectory(result: Dict, outcome: Optional[Dict], idx: int) -> str:
    lines = [f"### Trajectory {idx}"]
    lines.append(f"Query ID: {result.get('query_id')}")
    lines.append(f"Question: {_truncate(_safe_text(result.get('query', '')), MAX_QUERY_CHARS)}")
    lines.append(f"Final answer: {_truncate(_safe_text(result.get('answer', '')), MAX_ANSWER_CHARS)}")

    if outcome is not None:
        lines.append(f"Correct: {outcome.get('correct')}")
        lines.append(f"Retrieval recall: {outcome.get('retrieval_recall')}")
        lines.append(f"Citations precision: {outcome.get('citation_metrics', {}).get('precision')}")

    tool_calls = result.get("tool_calls", [])
    lines.append(f"Tool calls: {len(tool_calls)}")

    if not tool_calls:
        lines.append("No tool calls in this trajectory.")
        return "\n".join(lines)

    tool_names = [call.get("tool_name", "unknown") for call in tool_calls]
    lines.append(f"Tool sequence: {' -> '.join(tool_names)}")

    for i, call in enumerate(tool_calls[:MAX_TOOL_CALLS_PER_TRAJECTORY], 1):
        tool_name = call.get("tool_name", "unknown")
        output_text = _truncate(_safe_text(call.get("output", "")), MAX_OUTPUT_CHARS)
        lines.append(f"\nCall {i}: {tool_name}")
        lines.append(f"Output snippet:\n{output_text}")

    if len(tool_calls) > MAX_TOOL_CALLS_PER_TRAJECTORY:
        lines.append(
            f"\n... ({len(tool_calls) - MAX_TOOL_CALLS_PER_TRAJECTORY} more tool calls omitted)"
        )

    return "\n".join(lines)


def _format_trajectories_for_prompt(results_batch: List[Dict], outcomes: Dict[str, Dict]) -> str:
    sections: List[str] = []
    for idx, result in enumerate(results_batch, 1):
        outcome = outcomes.get(str(result.get("query_id")))
        sections.append(_format_trajectory(result, outcome, idx))
    return "\n\n---\n\n".join(sections)


def _prepare_batch_analysis_requests(
    config: Dict,
    results: List[Dict],
    scored: Dict,
    prompt_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    num_trajectories_batch: int,
) -> List[Dict]:
    if prompt_key not in BROWSECOMP_TOOL_DESCRIPTION_PROMPTS:
        raise ValueError(f"Unknown BrowseComp tool prompt key: {prompt_key}")

    if num_trajectories_batch <= 0:
        raise ValueError("--num-trajectories-batch must be positive")

    prompt_config = BROWSECOMP_TOOL_DESCRIPTION_PROMPTS[prompt_key]
    system_prompt = prompt_config["pre"]

    tools = config.get("tools", [])
    if not tools:
        raise ValueError("Config has no tools to improve")

    tool_lines = []
    for tool in tools:
        tool_def = tool.get("tool_definition", {})
        tool_lines.append(
            f"Tool Name: {tool_def.get('name', '')}\n"
            f"Current Description: {tool_def.get('description', '')}"
        )
    tool_block = "\n\n".join(tool_lines)

    outcomes = _build_outcomes_by_query(scored)

    requests = []
    for batch_idx, start in enumerate(range(0, len(results), num_trajectories_batch)):
        batch = results[start : start + num_trajectories_batch]
        trajectories_text = _format_trajectories_for_prompt(batch, outcomes)
        user_prompt = (
            "## Current Tool Definitions\n"
            f"{tool_block}\n\n"
            "## BrowseComp QA Trajectories\n"
            f"{trajectories_text}\n\n"
            "Now produce updated descriptions for every tool using the specified output format."
        )

        requests.append(
            _build_api_request(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                metadata={
                    "batch_idx": batch_idx,
                    "num_trajectories_batch": len(batch),
                    "prompt_key": prompt_key,
                },
            )
        )

    return requests


def _extract_updated_descriptions_from_analysis(content: str) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    lines = content.splitlines()

    current_tool: Optional[str] = None
    in_updated_desc = False
    desc_lines: List[str] = []

    def flush() -> None:
        nonlocal desc_lines, current_tool
        if current_tool and desc_lines:
            text = " ".join(line.strip() for line in desc_lines if line.strip()).strip()
            if text:
                descriptions[current_tool] = text

    for raw in lines:
        line = raw.strip()
        lower = line.lower()

        if line.startswith("**Tool:"):
            flush()
            current_tool = line.replace("**Tool:", "", 1).replace("**", "").strip()
            in_updated_desc = False
            desc_lines = []
            continue

        if lower.startswith("tool:"):
            flush()
            current_tool = line.split(":", 1)[1].strip()
            in_updated_desc = False
            desc_lines = []
            continue

        if lower.startswith("updated description:"):
            in_updated_desc = True
            desc_lines = []
            tail = line.split(":", 1)[1].strip()
            if tail:
                desc_lines.append(tail)
            continue

        if in_updated_desc:
            if not line:
                continue
            if (
                line.startswith("**Tool:")
                or lower.startswith("observed patterns:")
                or lower.startswith("distinguishing characteristics:")
                or lower.startswith("reasoning:")
                or line.startswith("**")
            ):
                in_updated_desc = False
                continue
            desc_lines.append(line)

    flush()
    return descriptions


def _prepare_synthesis_request(
    llm_responses: List[Dict],
    prompt_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
) -> Dict:
    if prompt_key not in BROWSECOMP_SYNTHESIS_DESCRIPTION_PROMPTS:
        raise ValueError(f"Unknown BrowseComp synthesis prompt key: {prompt_key}")

    response_chunks: List[str] = []
    for idx, response in enumerate(llm_responses, 1):
        content = response.get("content", "")
        if content:
            response_chunks.append(
                f"## Response {idx} (from batch {response.get('batch_idx', 0)}):\n{content}"
            )

    if not response_chunks:
        raise ValueError("No batch responses available for synthesis")

    user_prompt = (
        "Here are the N LLM responses analyzing different minibatches of BrowseComp trajectories:\n\n"
        + "\n\n---\n\n".join(response_chunks)
        + "\n\n---\n\nNow synthesize these into final tool descriptions using the specified format."
    )

    return _build_api_request(
        model=model,
        messages=[
            {"role": "system", "content": BROWSECOMP_SYNTHESIS_DESCRIPTION_PROMPTS[prompt_key]},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        metadata={
            "synthesis": True,
            "num_batch_responses": len(response_chunks),
            "prompt_key": prompt_key,
        },
    )


def _extract_synthesized_descriptions(content: str) -> Dict[str, str]:
    descriptions: Dict[str, str] = {}
    lines = content.splitlines()

    current_tool: Optional[str] = None
    in_final_description = False
    description_lines: List[str] = []

    def flush() -> None:
        nonlocal current_tool, description_lines
        if current_tool and description_lines:
            text = " ".join(line.strip() for line in description_lines if line.strip()).strip()
            if text:
                descriptions[current_tool] = text

    for raw in lines:
        line = raw.strip()
        lower = line.lower()

        if line.startswith("**Tool:"):
            flush()
            current_tool = line.replace("**Tool:", "", 1).replace("**", "").strip()
            in_final_description = False
            description_lines = []
            continue

        if lower.startswith("final description:"):
            in_final_description = True
            description_lines = []
            tail = line.split(":", 1)[1].strip()
            if tail:
                description_lines.append(tail)
            continue

        if in_final_description:
            if not line:
                continue
            if line.startswith("**Tool:") or lower.startswith("synthesis reasoning:"):
                in_final_description = False
                continue
            description_lines.append(line)

    flush()
    return descriptions


def _aggregate_batch_descriptions(llm_responses: List[Dict]) -> Dict[str, str]:
    by_tool: Dict[str, List[str]] = defaultdict(list)
    for response in llm_responses:
        parsed = response.get("parsed_descriptions", {})
        for tool_name, description in parsed.items():
            if description:
                by_tool[tool_name].append(description)

    aggregated: Dict[str, str] = {}
    for tool_name, candidates in by_tool.items():
        most_common = Counter(candidates).most_common(1)
        if most_common:
            aggregated[tool_name] = most_common[0][0]

    return aggregated


async def generate_improved_descriptions(
    config: Dict,
    results: List[Dict],
    scored: Dict,
    prompt_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str],
    num_trajectories_batch: int,
    synthesis_model: str,
    synthesis_temperature: float,
    synthesis_max_tokens: int,
    synthesis_prompt_key: str,
    synthesis_reasoning_effort: Optional[str],
) -> Tuple[Dict, List[Dict], Dict]:
    """Generate improved descriptions via minibatch analysis + synthesis."""
    if not results:
        raise ValueError("No trajectories found in results file")

    analysis_requests = _prepare_batch_analysis_requests(
        config=config,
        results=results,
        scored=scored,
        prompt_key=prompt_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        num_trajectories_batch=num_trajectories_batch,
    )

    logger.info("Running batch analysis with %d minibatches", len(analysis_requests))
    analysis_results = await openai_parallel_generate(
        analysis_requests,
        max_requests_per_minute=get_rate_limit(model),
        max_tokens_per_minute=get_token_limit(model),
        request_url=requests_url_dict.get(model, default_request_url),
    )

    llm_responses: List[Dict] = []
    for response in analysis_results:
        try:
            metadata = response[2]
            payload = response[1]
            content = _extract_response_content(payload)
            parsed = _extract_updated_descriptions_from_analysis(content)
            llm_responses.append(
                {
                    "batch_idx": metadata.get("batch_idx", 0),
                    "num_trajectories_batch": metadata.get("num_trajectories_batch", 0),
                    "content": content,
                    "parsed_descriptions": parsed,
                    "response": payload,
                }
            )
            logger.info(
                "Processed analysis batch %s with %d parsed descriptions",
                metadata.get("batch_idx", 0),
                len(parsed),
            )
        except Exception as exc:
            logger.error("Failed to process analysis response: %s", exc)

    if not llm_responses:
        raise ValueError("No valid batch analysis responses received")

    synthesis_request = _prepare_synthesis_request(
        llm_responses=llm_responses,
        prompt_key=synthesis_prompt_key,
        model=synthesis_model,
        temperature=synthesis_temperature,
        max_tokens=synthesis_max_tokens,
        reasoning_effort=synthesis_reasoning_effort,
    )

    logger.info("Running synthesis using %s", synthesis_model)
    synthesis_results = await openai_parallel_generate(
        [synthesis_request],
        max_requests_per_minute=get_rate_limit(synthesis_model),
        max_tokens_per_minute=get_token_limit(synthesis_model),
        request_url=requests_url_dict.get(synthesis_model, default_request_url),
    )

    if not synthesis_results:
        raise ValueError("Synthesis returned no response")

    synthesis_payload = synthesis_results[0][1]
    synthesis_content = _extract_response_content(synthesis_payload)
    synthesized_descriptions = _extract_synthesized_descriptions(synthesis_content)

    # Fallback to majority vote from batch-level parsed descriptions if synthesis misses tools.
    fallback_descriptions = _aggregate_batch_descriptions(llm_responses)
    final_descriptions = dict(fallback_descriptions)
    final_descriptions.update(synthesized_descriptions)

    improved_config = json.loads(json.dumps(config))
    updated_count = 0
    for tool in improved_config.get("tools", []):
        tool_def = tool.get("tool_definition", {})
        tool_name = tool_def.get("name")
        if tool_name in final_descriptions:
            tool_def["description"] = final_descriptions[tool_name]
            updated_count += 1

    logger.info("Applied synthesized descriptions to %d tools", updated_count)

    synthesis_artifacts = {
        "model": synthesis_model,
        "temperature": synthesis_temperature,
        "max_tokens": synthesis_max_tokens,
        "reasoning_effort": synthesis_reasoning_effort,
        "prompt_key": synthesis_prompt_key,
        "content": synthesis_content,
        "parsed_descriptions": synthesized_descriptions,
        "fallback_descriptions": fallback_descriptions,
        "final_descriptions": final_descriptions,
        "updated_count": updated_count,
        "raw_response": synthesis_payload,
        "num_batches": len(llm_responses),
    }

    return improved_config, llm_responses, synthesis_artifacts


def save_improved_config(
    result_dir: Path,
    improved_config: Dict,
    llm_responses: List[Dict],
    model: str,
    prompt_type: str,
    temperature: float,
    max_tokens: int,
    source_config: Path,
    source_results: Path,
    source_scored: Optional[Path],
    reasoning_effort: Optional[str],
    num_trajectories_batch: int,
    synthesis_artifacts: Dict,
) -> Path:
    """Save improved config and metadata in improvements/{editing}/vN."""
    base_run_path = get_base_run_path(result_dir) or Path(result_dir)

    editing_dirname = create_editing_dirname(
        model=model,
        temperature=temperature,
        prompt_key=prompt_type,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        num_trajectories_batch=num_trajectories_batch,
        synthesis_model=synthesis_artifacts.get("model"),
        synthesis_temperature=synthesis_artifacts.get("temperature"),
        synthesis_prompt_key=synthesis_artifacts.get("prompt_key"),
        synthesis_max_tokens=synthesis_artifacts.get("max_tokens"),
        synthesis_reasoning_effort=synthesis_artifacts.get("reasoning_effort"),
    )

    improvements_dir = base_run_path / "improvements" / editing_dirname
    next_version = get_next_version(improvements_dir, is_improvement=True)
    version_dir = improvements_dir / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    config_path = version_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(improved_config, f, indent=2)

    metadata = {
        "editing_model": model,
        "editing_prompt_type": prompt_type,
        "editing_temperature": temperature,
        "editing_max_tokens": max_tokens,
        "editing_reasoning_effort": reasoning_effort,
        "num_trajectories_batch": num_trajectories_batch,
        "source_config": str(source_config),
        "source_result_dir": str(result_dir),
        "source_results_file": str(source_results),
        "source_scored_file": str(source_scored) if source_scored else None,
        "improvement_version": next_version,
        "generated_timestamp": datetime.utcnow().isoformat() + "Z",
        "synthesis": {
            "model": synthesis_artifacts.get("model"),
            "temperature": synthesis_artifacts.get("temperature"),
            "max_tokens": synthesis_artifacts.get("max_tokens"),
            "reasoning_effort": synthesis_artifacts.get("reasoning_effort"),
            "prompt_key": synthesis_artifacts.get("prompt_key"),
            "num_batches": synthesis_artifacts.get("num_batches"),
            "updated_count": synthesis_artifacts.get("updated_count"),
        },
    }

    with open(version_dir / "generation_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(version_dir / "llm_responses.json", "w") as f:
        json.dump(llm_responses, f, indent=2)

    with open(version_dir / "synthesis_response.json", "w") as f:
        json.dump(synthesis_artifacts, f, indent=2)

    logger.info("Saved improved config: %s", config_path)
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Generate improved BrowseCompPlus tool descriptions")
    parser.add_argument("--result-dir", type=Path, required=True, help="Directory containing results files")
    parser.add_argument("--version", type=int, default=None, help="Optional base run version (vN) to improve")
    parser.add_argument(
        "--prompt-type",
        default="detailed_v2",
        choices=sorted(BROWSECOMP_TOOL_DESCRIPTION_PROMPTS.keys()),
        help="Batch analysis prompt type",
    )
    parser.add_argument("--model", default="gpt-5", help="Model for minibatch analysis")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for non-reasoning analysis models")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per minibatch analysis response")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for reasoning analysis models",
    )
    parser.add_argument(
        "--num-trajectories-batch",
        type=int,
        default=10,
        help="Number of trajectories per minibatch analysis request",
    )

    parser.add_argument("--synthesis-model", default=None, help="Model for synthesis (defaults to --model)")
    parser.add_argument(
        "--synthesis-temperature",
        type=float,
        default=0.3,
        help="Temperature for non-reasoning synthesis models",
    )
    parser.add_argument(
        "--synthesis-prompt-key",
        type=str,
        default="v2",
        choices=sorted(BROWSECOMP_SYNTHESIS_DESCRIPTION_PROMPTS.keys()),
        help="Synthesis prompt key",
    )
    parser.add_argument(
        "--synthesis-max-tokens",
        type=int,
        default=None,
        help="Max tokens for synthesis response (defaults to --max-tokens)",
    )
    parser.add_argument(
        "--synthesis-reasoning-effort",
        type=str,
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for reasoning synthesis models",
    )

    args = parser.parse_args()

    results_file, scored_file, metadata_file = resolve_run_files(args.result_dir, version=args.version)

    with open(results_file, "r") as f:
        results = json.load(f)

    if scored_file is None or not scored_file.exists():
        logger.error(
            "Scored file is required before generating improvements. Run evaluate first for %s",
            results_file,
        )
        return 1

    with open(scored_file, "r") as f:
        scored = json.load(f)

    config, config_path = load_config_from_metadata(metadata_file)

    synthesis_model = args.synthesis_model or args.model
    synthesis_max_tokens = args.synthesis_max_tokens or args.max_tokens
    synthesis_reasoning_effort = args.synthesis_reasoning_effort
    if synthesis_reasoning_effort is None and _is_reasoning_model(synthesis_model):
        synthesis_reasoning_effort = args.reasoning_effort

    improved_config, llm_responses, synthesis_artifacts = asyncio.run(
        generate_improved_descriptions(
            config=config,
            results=results,
            scored=scored,
            prompt_key=args.prompt_type,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            num_trajectories_batch=args.num_trajectories_batch,
            synthesis_model=synthesis_model,
            synthesis_temperature=args.synthesis_temperature,
            synthesis_max_tokens=synthesis_max_tokens,
            synthesis_prompt_key=args.synthesis_prompt_key,
            synthesis_reasoning_effort=synthesis_reasoning_effort,
        )
    )

    config_path = save_improved_config(
        result_dir=args.result_dir,
        improved_config=improved_config,
        llm_responses=llm_responses,
        model=args.model,
        prompt_type=args.prompt_type,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        source_config=config_path,
        source_results=results_file,
        source_scored=scored_file,
        reasoning_effort=args.reasoning_effort,
        num_trajectories_batch=args.num_trajectories_batch,
        synthesis_artifacts=synthesis_artifacts,
    )

    logger.info("Next step: python -m src.datasets.BrowseCompPlus.run --config-source %s ...", config_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
