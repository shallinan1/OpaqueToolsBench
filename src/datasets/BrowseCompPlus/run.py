"""BrowseCompPlus evaluation runner with shared-tool config support.

Output structure (config mode):
- Base runs: runs/BrowseCompPlus/{method}/shared_tools/{config_name}/{generation_hypers}/
  - v0_results.json, v0_metadata.json, v0_token_usage_generation.json
- Improvement runs: .../improvements/{editing_hypers}/vN/
  - results.json, metadata.json, token_usage_generation.json
"""

import json
import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime

import openai
from tqdm import tqdm
from dotenv import load_dotenv

os.environ["TRANSFORMERS_DISABLE_FLASH_ATTN"] = "1"
load_dotenv()

from src.generation_utils.openai_parallel_generate import (
    openai_parallel_generate,
    requests_url_dict,
    default_request_url,
)
from src.generation_utils.rate_limits import get_rate_limit, get_token_limit
from src.datasets.BrowseCompPlus.args import create_browsecompplus_parser
from src.datasets.BrowseCompPlus.prompts import (
    BROWSECOMP_FINAL_ANSWER_PROMPT,
    build_agent_messages,
    resolve_prompt_key,
)
from src.datasets.BrowseCompPlus.utils.path_utils import (
    parse_config_name,
    build_output_folder,
    config_uses_faiss,
    detect_improvement_context,
    get_next_version,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.generation_utils.openai_parallel_generate").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

vendor_path = Path("src/vendor/BrowseComp-Plus")
sys.path.insert(0, str(vendor_path))
sys.path.insert(0, str(vendor_path / "search_agent"))

from search_agent.openai_client import SearchToolHandler
from search_agent.utils import extract_retrieved_docids_from_result


class MockArgs:
    """Mock args for searcher initialization."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_api_request(messages, tools, args, request_id, tool_choice_override=None):
    """Create a properly formatted API request."""
    api_request = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice_override if tool_choice_override is not None else args.tool_choice,
        "metadata": {"request_id": request_id},
    }

    is_reasoning_model = args.model.startswith("o") or "gpt-5" in args.model
    if is_reasoning_model:
        if getattr(args, "max_tokens", None) is not None:
            api_request["max_completion_tokens"] = args.max_tokens
        if getattr(args, "reasoning_effort", None):
            api_request["reasoning_effort"] = args.reasoning_effort
    else:
        if getattr(args, "max_tokens", None) is not None:
            api_request["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            api_request["temperature"] = args.temperature
        if args.top_p is not None:
            api_request["top_p"] = args.top_p

    return api_request


def prepare_initial_requests(queries, tool_handler, args, prompt_key):
    """Prepare all initial API requests for parallel processing."""
    api_requests = []
    query_id_mapping = {}

    raw_tools = tool_handler.get_tool_definitions()
    tools = []
    for tool in raw_tools:
        tool_copy = tool.copy()
        tool_type = tool_copy.pop("type", "function")
        tools.append({"type": tool_type, "function": tool_copy})

    for idx, query in enumerate(queries):
        messages = build_agent_messages(
            query["text"],
            prompt_key=prompt_key,
            max_turns=args.max_iterations,
        )

        api_request = create_api_request(messages, tools, args, idx)
        api_requests.append(api_request)
        query_id_mapping[idx] = {
            "query_id": query["id"],
            "query_text": query["text"],
            "messages": messages.copy(),
            "tools": tools,
            "iteration": 0,
            "tool_calls_record": [],
        }

    return api_requests, query_id_mapping


def map_parallel_results(parallel_results, query_id_mapping):
    """Map parallel API results back to their original queries."""
    indexed_results = {}
    unknown_id_results = []

    for result in parallel_results:
        try:
            request_id = result[2]["request_id"]
            indexed_results[request_id] = result[1]
        except (KeyError, IndexError):
            unknown_id_results.append(result[1] if len(result) > 1 else result)

    if unknown_id_results:
        logger.warning("Found %d results without request IDs", len(unknown_id_results))
        for request_id in query_id_mapping:
            if request_id not in indexed_results and unknown_id_results:
                indexed_results[request_id] = unknown_id_results.pop(0)

    return indexed_results


def _extract_api_error_message(api_response):
    """Extract a human-readable error from non-success API payloads."""
    if isinstance(api_response, dict):
        error = api_response.get("error")
        if isinstance(error, dict):
            return error.get("message") or str(error)
        if "choices" not in api_response:
            return f"Unexpected API payload keys: {list(api_response.keys())[:5]}"
        return None

    if isinstance(api_response, list):
        messages = []
        for item in api_response:
            if not isinstance(item, dict):
                continue
            error = item.get("error")
            if isinstance(error, dict):
                msg = error.get("message")
                if msg:
                    messages.append(msg)
        if messages:
            return " | ".join(messages[:2])
        return "API payload was a list (likely failed attempts)."

    return f"Unexpected API payload type: {type(api_response).__name__}"


def _extract_assistant_message(api_response):
    """Safely extract assistant message from a successful completion payload."""
    if not isinstance(api_response, dict):
        return None

    choices = api_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    return message if isinstance(message, dict) else None


def process_batch_responses(indexed_results, query_id_mapping, tool_handler):
    """Process a batch of API responses and execute tool calls."""
    queries_needing_followup = {}
    all_tool_calls = []
    tool_call_metadata = []

    for request_id, api_response in indexed_results.items():
        query_info = query_id_mapping[request_id]
        message = _extract_assistant_message(api_response)
        if message is None:
            error_msg = _extract_api_error_message(api_response)
            logger.warning(
                "Request %s returned no assistant message; skipping follow-up. Error: %s",
                request_id,
                error_msg,
            )
            query_info["tool_calls_record"].append(
                {
                    "type": "api_error",
                    "stage": "generation",
                    "error": error_msg,
                }
            )
            continue
        query_info["messages"].append(message)

        for tool_call in message.get("tool_calls", []):
            func_args = tool_call["function"]["arguments"]
            if isinstance(func_args, str):
                try:
                    func_args = json.loads(func_args)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse tool-call args for request %s tool %s; skipping tool call",
                        request_id,
                        tool_call["function"]["name"],
                    )
                    query_info["tool_calls_record"].append(
                        {
                            "type": "tool_call_parse_error",
                            "tool_name": tool_call["function"]["name"],
                            "raw_arguments": func_args,
                        }
                    )
                    continue

            all_tool_calls.append({"tool_name": tool_call["function"]["name"], "arguments": func_args})
            tool_call_metadata.append(
                {
                    "request_id": request_id,
                    "call_id": tool_call["id"],
                    "func_name": tool_call["function"]["name"],
                }
            )

    if all_tool_calls:
        logger.info("Executing %d tool calls with batch execution", len(all_tool_calls))
        tool_results = tool_handler.execute_batch_tools(all_tool_calls)

        for metadata, result in zip(tool_call_metadata, tool_results):
            request_id = metadata["request_id"]
            query_info = query_id_mapping[request_id]

            query_info["tool_calls_record"].append(
                {
                    "type": "tool_call",
                    "tool_name": metadata["func_name"],
                    "output": result,
                }
            )
            query_info["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": metadata["call_id"],
                    "content": result,
                }
            )

            queries_needing_followup[request_id] = query_info
            query_info["iteration"] += 1

    return queries_needing_followup


def _accumulate_token_usage(parallel_results, totals):
    """Aggregate token usage from OpenAI parallel responses."""
    for result in parallel_results:
        if len(result) < 2:
            continue
        response = result[1]
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
        totals["completion_tokens"] += usage.get("completion_tokens", 0)
        totals["total_tokens"] += usage.get("total_tokens", 0)


def _run_forced_final_answers(
    query_id_mapping,
    args,
    max_requests_per_minute,
    max_tokens_per_minute,
    token_usage_totals,
):
    """Force one final non-tool answer turn for queries capped by max_iterations."""
    forced_api_requests = []
    forced_request_ids = set()

    for request_id, query_info in query_id_mapping.items():
        if query_info["iteration"] < args.max_iterations:
            continue

        messages = query_info["messages"]
        if not messages:
            continue

        last_message = messages[-1]
        last_role = last_message.get("role")
        # Skip queries that already ended with a non-tool assistant response.
        if last_role == "assistant" and not last_message.get("tool_calls"):
            continue

        messages.append({"role": "user", "content": BROWSECOMP_FINAL_ANSWER_PROMPT})
        forced_api_requests.append(
            create_api_request(
                messages,
                query_info["tools"],
                args,
                request_id,
                tool_choice_override="none",
            )
        )
        forced_request_ids.add(request_id)

    if not forced_api_requests:
        return 0

    logger.info(
        "Forcing final non-tool answer for %d queries that reached max iterations (%d)",
        len(forced_api_requests),
        args.max_iterations,
    )

    parallel_results = asyncio.run(
        openai_parallel_generate(
            forced_api_requests,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            request_url=requests_url_dict.get(args.model, default_request_url),
        )
    )
    _accumulate_token_usage(parallel_results, token_usage_totals)

    indexed_results = map_parallel_results(parallel_results, query_id_mapping)
    appended = 0
    for request_id, api_response in indexed_results.items():
        if request_id not in forced_request_ids:
            continue
        message = _extract_assistant_message(api_response)
        if message is None:
            error_msg = _extract_api_error_message(api_response)
            logger.warning(
                "Forced-final request %s returned no assistant message. Error: %s",
                request_id,
                error_msg,
            )
            query_id_mapping[request_id]["tool_calls_record"].append(
                {
                    "type": "api_error",
                    "stage": "forced_final_answer",
                    "error": error_msg,
                }
            )
            continue
        query_id_mapping[request_id]["messages"].append(message)
        appended += 1

    return appended


def _select_prompt_key(args):
    """Resolve prompt key from args and defaults."""
    return resolve_prompt_key(
        prompt_key=getattr(args, "prompt_key", None),
        include_get_document=bool(args.include_get_document),
    )


def main(args):
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    if not args.output_dir:
        args.output_dir = Path("runs/BrowseCompPlus/tool_observer")

    config_mode = args.config_source is not None
    config_version = None

    if config_mode:
        config_source = Path(args.config_source)
        if not config_source.exists():
            logger.error("Config file not found: %s", config_source)
            sys.exit(1)

        logger.info("Loading config from: %s", config_source)
        is_improvement, _, improvement_version = detect_improvement_context(config_source)
        if is_improvement:
            config_version = improvement_version
            logger.info("Detected improvement config version: v%s", config_version)
        else:
            config_version = 0
        args._uses_faiss = config_uses_faiss(config_source)
    elif not args.searcher_type:
        logger.error("Single-searcher mode requires --searcher-type when --config-source is not provided")
        sys.exit(1)
    else:
        args._uses_faiss = args.searcher_type == "faiss"

    if config_mode:
        from src.datasets.BrowseCompPlus.configurable_tool_handler import ConfigurableToolHandler
        from src.datasets.BrowseCompPlus.custom_searcher.bm25_domain_filtered_searcher import (
            BM25DomainFilteredSearcher,
        )
        from src.datasets.BrowseCompPlus.custom_searcher.faiss_domain_filtered_searcher import (
            FaissDomainFilteredSearcher,
        )

        shared_searchers = {}

        def create_searcher(searcher_type):
            if searcher_type not in shared_searchers:
                if searcher_type == "bm25":
                    searcher_args = MockArgs(
                        index_path=str(args.index_path),
                        url_mapping_file=str(args.url_mapping_file),
                        categories_file=str(args.categories_file),
                        filter_category=None,
                        filter_domain=None,
                        show_urls=not args.hide_urls,
                    )
                    shared_searchers[searcher_type] = BM25DomainFilteredSearcher(searcher_args)
                elif searcher_type == "faiss":
                    if "qwen3-embedding-8b" in args.faiss_index_path.lower():
                        model_name = "Qwen/Qwen3-Embedding-8B"
                    elif "qwen3-embedding-4b" in args.faiss_index_path.lower():
                        model_name = "Qwen/Qwen3-Embedding-4B"
                    else:
                        model_name = "Qwen/Qwen3-Embedding-0.6B"

                    searcher_args = MockArgs(
                        index_path=args.faiss_index_path,
                        model_name=model_name,
                        normalize=False,
                        pooling="eos",
                        torch_dtype="float16",
                        dataset_name="Tevatron/browsecomp-plus-corpus",
                        task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
                        max_length=8192,
                        max_batch_search_queries=args.faiss_max_batch_queries,
                        categories_file=str(args.categories_file),
                        filter_category=None,
                        filter_domain=None,
                        show_urls=not args.hide_urls,
                        attn_implementation="eager",
                    )
                    shared_searchers[searcher_type] = FaissDomainFilteredSearcher(searcher_args)
                else:
                    raise ValueError(f"Unknown searcher type: {searcher_type}")

                logger.info("Created shared %s searcher instance", searcher_type)

            return shared_searchers[searcher_type]

        tool_handler = ConfigurableToolHandler(
            config_path=args.config_source,
            searcher_factory=create_searcher,
            k=args.k,
            snippet_max_tokens=args.snippet_max_tokens,
            include_get_document=args.include_get_document,
        )
        config_summary = tool_handler.get_config_summary()
        logger.info(
            "Loaded config '%s' with %d tools",
            config_summary["config_name"],
            config_summary["num_tools"],
        )
    else:
        filter_info = []
        if args.filter_category:
            filter_info.append(f"category={args.filter_category}")
        if args.filter_domain:
            filter_info.append(f"domain={args.filter_domain}")
        filter_str = ", ".join(filter_info) if filter_info else "no filtering"
        logger.info("Running %s with %s", args.searcher_type, filter_str)

        if args.searcher_type == "bm25":
            from src.datasets.BrowseCompPlus.custom_searcher.bm25_domain_filtered_searcher import (
                BM25DomainFilteredSearcher,
            )

            searcher_args = MockArgs(
                index_path=str(args.index_path),
                url_mapping_file=str(args.url_mapping_file),
                categories_file=str(args.categories_file) if args.filter_category else None,
                filter_category=args.filter_category,
                filter_domain=args.filter_domain,
                show_urls=not args.hide_urls,
            )
            searcher = BM25DomainFilteredSearcher(searcher_args)
        else:
            from src.datasets.BrowseCompPlus.custom_searcher.faiss_domain_filtered_searcher import (
                FaissDomainFilteredSearcher,
            )

            if "qwen3-embedding-8b" in args.faiss_index_path.lower():
                model_name = "Qwen/Qwen3-Embedding-8B"
            elif "qwen3-embedding-4b" in args.faiss_index_path.lower():
                model_name = "Qwen/Qwen3-Embedding-4B"
            else:
                model_name = "Qwen/Qwen3-Embedding-0.6B"

            searcher_args = MockArgs(
                index_path=args.faiss_index_path,
                model_name=model_name,
                normalize=False,
                pooling="eos",
                torch_dtype="float16",
                dataset_name="Tevatron/browsecomp-plus-corpus",
                task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
                max_length=8192,
                max_batch_search_queries=args.faiss_max_batch_queries,
                categories_file=str(args.categories_file) if args.filter_category else None,
                filter_category=args.filter_category,
                filter_domain=args.filter_domain,
                show_urls=not args.hide_urls,
                attn_implementation="eager",
            )
            searcher = FaissDomainFilteredSearcher(searcher_args)

        logger.info("Searcher initialized: %s", searcher.search_type)
        tool_handler = SearchToolHandler(
            searcher,
            snippet_max_tokens=args.snippet_max_tokens,
            k=args.k,
            include_get_document=args.include_get_document,
        )

    # Resolve queries file from --split or default to all queries
    if getattr(args, "split", None):
        queries_file = Path(__file__).parent / "data" / f"{args.split}_queries.tsv"
        logger.info("Using %s split: %s", args.split, queries_file)
    else:
        queries_file = vendor_path / "topics-qrels" / "queries.tsv"
        logger.info("Using all queries: %s", queries_file)

    queries = []
    with open(queries_file, "r") as f:
        for i, line in enumerate(f):
            if args.num_queries is not None and i >= args.num_queries:
                break
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries.append({"id": parts[0], "text": parts[1]})

    logger.info("Processing %d queries", len(queries))

    _ = openai.OpenAI()
    results = []

    prompt_key = _select_prompt_key(args)
    args._resolved_prompt_key = prompt_key
    logger.info("Using prompt key: %s", prompt_key)

    api_requests, query_id_mapping = prepare_initial_requests(queries, tool_handler, args, prompt_key)

    if args.debug and api_requests:
        print("=== DEBUG: First API Request ===")
        print(json.dumps(api_requests[0], indent=2))
        print("=" * 40)

    max_requests_per_minute = get_rate_limit(args.model)
    max_tokens_per_minute = get_token_limit(args.model)
    logger.info(
        "Using rate limits - Max requests/min: %s, Max tokens/min: %s",
        max_requests_per_minute,
        max_tokens_per_minute,
    )

    overall_start_time = time.time()
    token_usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "model": args.model}

    pbar = tqdm(range(args.max_iterations), desc="Iterations", position=0)
    for iter_num in pbar:
        if not api_requests:
            break

        pbar.set_description(
            f"Iteration {iter_num + 1}/{args.max_iterations}: Processing {len(api_requests)} requests"
        )

        parallel_results = asyncio.run(
            openai_parallel_generate(
                api_requests,
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=max_tokens_per_minute,
                request_url=requests_url_dict.get(args.model, default_request_url),
            )
        )
        _accumulate_token_usage(parallel_results, token_usage_totals)

        indexed_results = map_parallel_results(parallel_results, query_id_mapping)
        queries_needing_followup = process_batch_responses(indexed_results, query_id_mapping, tool_handler)

        api_requests = []
        for request_id, query_info in queries_needing_followup.items():
            if query_info["iteration"] < args.max_iterations:
                api_requests.append(
                    create_api_request(query_info["messages"], query_info["tools"], args, request_id)
                )

    forced_final_answers = _run_forced_final_answers(
        query_id_mapping=query_id_mapping,
        args=args,
        max_requests_per_minute=max_requests_per_minute,
        max_tokens_per_minute=max_tokens_per_minute,
        token_usage_totals=token_usage_totals,
    )

    total_processing_time = time.time() - overall_start_time

    for request_id in sorted(query_id_mapping.keys()):
        query_info = query_id_mapping[request_id]
        messages = query_info["messages"]

        final_content = ""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                final_content = content
            elif content is None:
                final_content = ""
            else:
                final_content = str(content)
            break

        answer = ""
        if "Exact Answer:" in final_content:
            for line in final_content.split("\n"):
                if line.startswith("Exact Answer:"):
                    answer = line.replace("Exact Answer:", "").strip()
                    break
        else:
            answer = final_content.strip()

        retrieved_docids = extract_retrieved_docids_from_result(query_info["tool_calls_record"])

        results.append(
            {
                "query_id": query_info["query_id"],
                "query": query_info["query_text"],
                "answer": answer,
                "retrieved_docids": retrieved_docids,
                "messages": messages,
                "tool_calls": query_info["tool_calls_record"],
            }
        )

    now = datetime.utcnow()
    ts_machine = now.strftime("%Y%m%dT%H%M%S%fZ")
    ts_human = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    is_improvement = False
    save_in_improvement_layout = False
    if config_mode:
        config_name = parse_config_name(Path(args.config_source))
        is_improvement, improvement_base, improvement_version = detect_improvement_context(Path(args.config_source))

        # For iterative-improve train runs, keep writing directly into improvements/vN.
        # For direct test evaluation on an improvement config, write a normal versioned run
        # under --output-dir to avoid clobbering improvement artifacts.
        eval_improvement_on_test = is_improvement and getattr(args, "split", None) == "test"

        if is_improvement and not eval_improvement_on_test:
            save_in_improvement_layout = True
            version = improvement_version
            output_folder = improvement_base / f"v{improvement_version}"
            results_path = output_folder / "results.json"
            metadata_path = output_folder / "metadata.json"
            token_usage_path = output_folder / "token_usage_generation.json"
        else:
            output_folder = build_output_folder(args, config_name, mode="shared_tools")
            if eval_improvement_on_test:
                output_folder = output_folder / f"from_v{improvement_version}"
                logger.info(
                    "Evaluating improvement config v%s on test split; saving to %s",
                    improvement_version,
                    output_folder,
                )
            version = get_next_version(output_folder, is_improvement=False)
            results_path = output_folder / f"v{version}_results.json"
            metadata_path = output_folder / f"v{version}_metadata.json"
            token_usage_path = output_folder / f"v{version}_token_usage_generation.json"
    else:
        output_folder = build_output_folder(args, "single_searcher", mode="single_searcher")
        version = get_next_version(output_folder, is_improvement=False)
        results_path = output_folder / f"v{version}_results.json"
        metadata_path = output_folder / f"v{version}_metadata.json"
        token_usage_path = output_folder / f"v{version}_token_usage_generation.json"

    output_folder.mkdir(parents=True, exist_ok=True)

    if config_mode and save_in_improvement_layout:
        # .../{generation_hypers}/improvements/{editing_hypers}/vN
        hyperparam_dirname = output_folder.parents[2].name if len(output_folder.parents) >= 3 else None
    else:
        hyperparam_dirname = output_folder.name

    metadata = {
        "timestamp": ts_human,
        "timestamp_utc": ts_machine,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "reasoning_effort": args.reasoning_effort if args.model.startswith("o") or "gpt-5" in args.model else None,
        "max_tokens": args.max_tokens,
        "tool_choice": args.tool_choice,
        "prompt_key": prompt_key,
        "k": args.k,
        "snippet_max_tokens": args.snippet_max_tokens,
        "include_get_document": args.include_get_document,
        "show_urls": not args.hide_urls,
        "max_iterations": args.max_iterations,
        "forced_final_answers": forced_final_answers,
        "num_queries": len(results),
        "total_time_seconds": total_processing_time,
        "queries_file": str(queries_file),
        "split": getattr(args, "split", None),
        "version": version,
        "is_improvement": is_improvement,
        "hyperparam_dirname": hyperparam_dirname,
        "command": " ".join(sys.argv),
    }

    if config_mode:
        config_summary = tool_handler.get_config_summary()
        metadata.update(
            {
                "config_source": str(args.config_source),
                "config_iteration": config_version,
                "config_name": config_summary["config_name"],
                "config_description": config_summary["config_description"],
                "num_tools": config_summary["num_tools"],
                "tool_names": config_summary["tool_names"],
                "faiss_index_path": args.faiss_index_path,
            }
        )
    else:
        metadata.update(
            {
                "searcher_type": args.searcher_type,
                "searcher_name": searcher.search_type,
                "filter_category": args.filter_category if args.filter_category else "none",
                "filter_domain": args.filter_domain,
                "faiss_index_path": args.faiss_index_path if args.searcher_type == "faiss" else None,
            }
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(token_usage_path, "w", encoding="utf-8") as f:
        json.dump(token_usage_totals, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to %s", output_folder)
    logger.info("Metadata: %s", metadata_path)
    logger.info("Results: %s", results_path)
    logger.info("Token usage: %s", token_usage_path)

    hours = int(total_processing_time // 3600)
    minutes = int((total_processing_time % 3600) // 60)
    seconds = total_processing_time % 60
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds:.1f}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        time_str = f"{seconds:.1f}s"
    logger.info("Processed %d queries in %s total", len(results), time_str)


if __name__ == "__main__":
    parser = create_browsecompplus_parser(vendor_path)
    args = parser.parse_args()
    main(args)
