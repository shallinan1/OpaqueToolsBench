"""Evaluate BrowseCompPlus run outputs with LLM-as-judge scoring."""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
import asyncio

load_dotenv()

vendor_path = Path("src/vendor/BrowseComp-Plus")
from src.datasets.BrowseCompPlus.evaluation_utils import (
    load_ground_truth,
    create_judge_prompt,
    parse_judge_response,
    extract_citations_from_response,
    compute_citation_metrics,
    load_qrel_data,
)

from src.generation_utils.openai_parallel_generate import (
    openai_parallel_generate,
    requests_url_dict,
    default_request_url,
)
from src.generation_utils.rate_limits import get_rate_limit, get_token_limit


def load_results(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)


def _extract_version_from_results_name(filename: str):
    import re
    match = re.match(r"v(\d+)_results\.json", filename)
    return int(match.group(1)) if match else None


def resolve_results_file(results_file: Path = None, result_dir: Path = None, version: int = None) -> Path:
    """Resolve the actual results file path from explicit file or result directory."""
    if results_file is not None and result_dir is not None:
        raise ValueError("Use only one of --results-file or --result-dir")

    if results_file is not None:
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        return results_file

    if result_dir is None:
        raise ValueError("One of --results-file or --result-dir is required")

    result_dir = Path(result_dir)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    # Improvement directory format
    improvement_results = result_dir / "results.json"
    if improvement_results.exists():
        return improvement_results

    # Base directory format
    versioned_results = list(result_dir.glob("v*_results.json"))
    if not versioned_results:
        raise FileNotFoundError(f"No results.json or v*_results.json found in {result_dir}")

    if version is not None:
        target = result_dir / f"v{version}_results.json"
        if not target.exists():
            raise FileNotFoundError(f"Requested version file not found: {target}")
        return target

    versioned_results.sort(key=lambda p: _extract_version_from_results_name(p.name) or -1)
    return versioned_results[-1]


def resolve_output_file(output_file: Path, results_file: Path) -> Path:
    """Choose default scored file name based on input results filename."""
    if output_file is not None:
        return output_file

    version = _extract_version_from_results_name(results_file.name)
    if version is None:
        return results_file.parent / "scored.json"
    return results_file.parent / f"v{version}_scored.json"


def _is_reasoning_model(model: str) -> bool:
    """Return True for models that require max_completion_tokens."""
    return model.startswith("o") or "gpt-5" in model


def prepare_judge_requests(
    results_to_eval,
    ground_truth,
    model="gpt-5",
    judge_max_tokens: int = 1024,
    judge_reasoning_effort: str = "medium",
):
    api_requests = []
    request_mapping = {}

    for idx, result in enumerate(results_to_eval):
        query_id = str(result["query_id"])
        if query_id not in ground_truth:
            print(f"Warning: Query {query_id} not in ground truth")
            continue

        gt = ground_truth[query_id]
        predicted = result.get("answer", "")
        gold = gt["answer"]
        prompt = create_judge_prompt(gt["question"], predicted, gold)

        request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "metadata": {"request_id": idx},
        }
        if _is_reasoning_model(model):
            request["max_completion_tokens"] = judge_max_tokens
            if judge_reasoning_effort:
                request["reasoning_effort"] = judge_reasoning_effort
        else:
            request["temperature"] = 0.7
            request["max_tokens"] = judge_max_tokens

        api_requests.append(request)
        request_mapping[idx] = {
            "result": result,
            "query_id": query_id,
            "predicted": predicted,
            "gold": gold,
        }

    return api_requests, request_mapping


def process_judge_responses(parallel_results, request_mapping, qrel_evidence):
    all_evaluations = []
    citation_metrics = []
    tool_call_totals = {}
    retrieval_recalls = []

    indexed_results = {}
    failed_judge_requests = 0
    for result in parallel_results:
        if len(result) < 3 or not isinstance(result[2], dict) or "request_id" not in result[2]:
            continue
        request_id = result[2]["request_id"]
        indexed_results[request_id] = result[1]

    for request_id, api_response in indexed_results.items():
        info = request_mapping[request_id]
        result = info["result"]
        query_id = info["query_id"]

        predicted = info["predicted"]
        gold = info["gold"]

        # Failed requests are returned as a list of errors by the async processor.
        if not isinstance(api_response, dict):
            failed_judge_requests += 1
            continue

        choices = api_response.get("choices")
        if not choices:
            failed_judge_requests += 1
            continue

        judge_response = choices[0].get("message", {}).get("content", "")
        if not judge_response:
            failed_judge_requests += 1
            continue

        try:
            judge_result = parse_judge_response(judge_response)
        except Exception:
            failed_judge_requests += 1
            continue

        is_correct = judge_result.get("correct", False)

        tool_calls_record = result.get("tool_calls", [])
        retrieved_docids = result.get("retrieved_docids", [])

        tool_counts = {}
        for call in tool_calls_record:
            tool_name = call.get("tool_name")
            if tool_name:
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        cited_docids = extract_citations_from_response(predicted)
        relevant_docids = qrel_evidence.get(query_id, [])
        citation_scores = compute_citation_metrics(cited_docids, relevant_docids)
        citation_metrics.append(citation_scores)

        for tool, count in tool_counts.items():
            tool_call_totals[tool] = tool_call_totals.get(tool, 0) + count

        retrieved_docs = set(retrieved_docids)
        retrieval_recall = 0
        if relevant_docids:
            retrieval_recall = len(retrieved_docs & set(relevant_docids)) / len(relevant_docids)
            retrieval_recalls.append(retrieval_recall)

        all_evaluations.append(
            {
                "query_id": query_id,
                "correct": is_correct,
                "exact_match": predicted.strip().lower() == gold.strip().lower(),
                "judge_result": judge_result,
                "citation_metrics": citation_scores,
                "tool_counts": tool_counts,
                "num_tool_calls": sum(tool_counts.values()),
                "retrieval_recall": retrieval_recall if relevant_docids else None,
            }
        )

    return all_evaluations, citation_metrics, tool_call_totals, retrieval_recalls, failed_judge_requests


def main(args):
    results_file = resolve_results_file(args.results_file, args.result_dir, args.version)
    output_file = resolve_output_file(args.output_file, results_file)

    print(f"Loading results from {results_file}")
    results = load_results(results_file)

    print(f"Loading ground truth from {args.ground_truth}")
    ground_truth = load_ground_truth(args.ground_truth)

    print(f"Loading qrel evidence from {args.qrel_evidence}")
    qrel_evidence = load_qrel_data(args.qrel_evidence)

    results_to_eval = results[: args.max_eval] if args.max_eval else results

    print(f"\nPreparing {len(results_to_eval)} queries for parallel evaluation...")
    eval_start_time = time.time()

    api_requests, request_mapping = prepare_judge_requests(
        results_to_eval,
        ground_truth,
        model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        judge_reasoning_effort=args.judge_reasoning_effort,
    )

    if not api_requests:
        print("No valid queries to evaluate")
        return None

    max_requests_per_minute = get_rate_limit(args.judge_model)
    max_tokens_per_minute = get_token_limit(args.judge_model)
    print(f"Using judge model: {args.judge_model}")
    print(
        f"Rate limits - Max requests/min: {max_requests_per_minute}, Max tokens/min: {max_tokens_per_minute}"
    )

    print(f"\nEvaluating {len(api_requests)} queries in parallel...")
    parallel_results = asyncio.run(
        openai_parallel_generate(
            api_requests,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            request_url=requests_url_dict.get(args.judge_model, default_request_url),
        )
    )

    print("Processing evaluation results...")
    all_evaluations, citation_metrics, tool_call_totals, retrieval_recalls, failed_judge_requests = process_judge_responses(
        parallel_results,
        request_mapping,
        qrel_evidence,
    )

    total_eval_time = time.time() - eval_start_time
    requested = len(api_requests)
    total = len(all_evaluations)
    correct = sum(1 for e in all_evaluations if e["correct"])
    accuracy = correct / total if total > 0 else 0

    if citation_metrics:
        avg_citation_precision = sum(m["precision"] for m in citation_metrics) / len(citation_metrics)
        avg_citation_recall = sum(m["recall"] for m in citation_metrics) / len(citation_metrics)
        avg_citations_per_query = sum(m["num_citations"] for m in citation_metrics) / len(citation_metrics)
        responses_with_citations = sum(1 for m in citation_metrics if m["num_citations"] > 0)
    else:
        avg_citation_precision = 0
        avg_citation_recall = 0
        avg_citations_per_query = 0
        responses_with_citations = 0

    total_tool_calls = sum(tool_call_totals.values())
    avg_tool_calls_per_query = total_tool_calls / total if total > 0 else 0
    avg_retrieval_recall = sum(retrieval_recalls) / len(retrieval_recalls) if retrieval_recalls else 0

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "requested": requested,
        "total": total,
        "failed_judge_requests": failed_judge_requests,
        "citation_metrics": {
            "avg_precision": avg_citation_precision,
            "avg_recall": avg_citation_recall,
            "avg_citations_per_query": avg_citations_per_query,
            "responses_with_citations": responses_with_citations,
        },
        "tool_usage": {
            "total_calls": total_tool_calls,
            "avg_calls_per_query": avg_tool_calls_per_query,
            "by_tool": tool_call_totals,
        },
        "retrieval": {"avg_recall": avg_retrieval_recall},
        "performance": {
            "total_evaluation_time": total_eval_time,
            "avg_time_per_query": total_eval_time / total if total > 0 else 0,
            "judge_model": args.judge_model,
        },
    }

    output = {"summary": summary, "detailed_evaluations": all_evaluations}
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {output_file}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BrowseCompPlus results")
    parser.add_argument("--results-file", type=Path, default=None, help="Path to a specific results JSON file")
    parser.add_argument("--result-dir", type=Path, default=None, help="Run directory containing results files")
    parser.add_argument("--version", type=int, default=None, help="When using --result-dir, evaluate a specific vN file")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=vendor_path / "data" / "browsecomp_plus_decrypted.jsonl",
        help="Path to ground truth JSONL",
    )
    parser.add_argument(
        "--qrel-evidence",
        type=Path,
        default=vendor_path / "topics-qrels" / "qrel_evidence.txt",
        help="Path to qrel evidence file",
    )
    parser.add_argument("--max-eval", type=int, default=None, help="Max queries to evaluate")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional output file path")
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5",
        help="Model used for judging answers",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
        help="Max output tokens for judge completion",
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        type=str,
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for reasoning judge models",
    )
    main(parser.parse_args())
