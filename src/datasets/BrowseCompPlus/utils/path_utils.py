"""Path management utilities for BrowseCompPlus file structure.

Directory structure:
- runs/BrowseCompPlus/{method}/shared_tools/{config_name}/{generation_hypers}/
  where {method} is "tool_observer"
  - v0_results.json, v0_metadata.json, v0_scored.json
  - improvements/{editing_hypers}/
    - v1/config.json, results.json, metadata.json, scored.json
    - v2/...
"""

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

from src.datasets.BrowseCompPlus.prompts import resolve_prompt_key


KNOWN_METHODS = {"tool_observer"}


def _shorten_model_name(model: str) -> str:
    """Shorten model name for directory naming."""
    model_clean = model.split("/")[-1]
    if model_clean == "gpt-5":
        return "gpt5"
    if model_clean == "gpt-5.1":
        return "gpt51"
    if model_clean == "gpt-5-mini":
        return "gpt5mini"
    if "-20" in model_clean and not model_clean.startswith("o"):
        return model_clean.split("-20")[0].replace("-", "").replace("_", "")
    return model_clean.replace("-", "").replace("_", "")


def _prompt_key_abbrev(prompt_key: str) -> str:
    """Map prompt keys to short stable tokens."""
    mapping = {
        "runtime_with_get_document": "rwd",
        "runtime_search_only": "rso",
        "runtime_search_only_no_citation": "rsonc",
    }
    if not prompt_key:
        return "auto"
    if prompt_key in mapping:
        return mapping[prompt_key]

    sanitized = re.sub(r"[^a-z0-9]+", "", prompt_key.lower())
    if not sanitized:
        sanitized = "custom"
    return f"pk{sanitized}"


def parse_config_name(config_path: Path) -> str:
    """Extract config name from config/result paths."""
    config_path = Path(config_path)
    parts = config_path.parts

    # runs/BrowseCompPlus/{method}/shared_tools/{config_name}/...
    if "runs" in parts and "BrowseCompPlus" in parts:
        bc_idx = parts.index("BrowseCompPlus")
        offset = 1
        if bc_idx + 1 < len(parts) and parts[bc_idx + 1] in KNOWN_METHODS:
            offset = 2
        st_idx = bc_idx + offset
        if st_idx < len(parts) and parts[st_idx] == "shared_tools" and st_idx + 1 < len(parts):
            return parts[st_idx + 1]

    # shared_tools/{method}/{config_name}/.../optimized_config.json
    if "shared_tools" in parts:
        st_idx = parts.index("shared_tools")
        if st_idx + 1 < len(parts) and parts[st_idx + 1] in KNOWN_METHODS and st_idx + 2 < len(parts):
            return parts[st_idx + 2]

    return config_path.stem


def config_uses_faiss(config_path: Path) -> bool:
    """Best-effort check whether a config contains FAISS-backed tools."""
    config_path = Path(config_path)
    if not config_path.exists() or not config_path.is_file():
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return False

    for tool in config.get("tools", []):
        searcher_type = tool.get("searcher_config", {}).get("searcher_type")
        if searcher_type == "faiss":
            return True
    return False


def _faiss_embedding_tag(faiss_index_path: Optional[str]) -> Optional[str]:
    """Extract a stable embedding tag from a FAISS index path."""
    if not faiss_index_path:
        return None

    path = str(faiss_index_path).lower()
    match = re.search(r"qwen3-embedding-([0-9]+(?:\.[0-9]+)?b)", path)
    if match:
        size = match.group(1).replace(".", "p")
        return f"emb{size}"

    # Fallback for custom/unknown embedding index naming.
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()[:6]
    return f"embx{digest}"


def _uses_faiss(args, config_name: Optional[str] = None) -> bool:
    """Determine whether the current run uses FAISS retrieval."""
    if hasattr(args, "_uses_faiss"):
        return bool(getattr(args, "_uses_faiss"))

    if getattr(args, "searcher_type", None) == "faiss":
        return True

    if config_name and "faiss" in config_name.lower():
        return True

    return False


def create_generation_dirname(args, config_name: Optional[str] = None) -> str:
    """Create directory name from generation hyperparameters."""
    model_short = _shorten_model_name(args.model)
    supports_temp = not (model_short.startswith("o") or model_short.startswith("gpt5"))

    parts = [model_short]

    if supports_temp:
        parts.append(f"t{str(args.temperature).replace('.', '')}")
        parts.append(f"p{str(args.top_p).replace('.', '')}")
    elif getattr(args, "reasoning_effort", None):
        parts.append(args.reasoning_effort)

    tool_choice_map = {"required": "req", "auto": "auto", "none": "none"}
    parts.append(tool_choice_map.get(args.tool_choice, args.tool_choice))
    parts.append(f"mx{args.max_tokens}")

    prompt_key = getattr(args, "_resolved_prompt_key", None) or getattr(args, "prompt_key", None)
    parts.append(_prompt_key_abbrev(prompt_key))

    parts.append(f"k{args.k}")
    parts.append(f"s{args.snippet_max_tokens}")
    parts.append(f"gd{1 if args.include_get_document else 0}")
    parts.append(f"url{0 if args.hide_urls else 1}")
    parts.append(f"iter{args.max_iterations}")

    if _uses_faiss(args, config_name=config_name):
        embedding_tag = _faiss_embedding_tag(getattr(args, "faiss_index_path", None))
        if embedding_tag:
            parts.append(embedding_tag)

    if getattr(args, "num_queries", None) is not None:
        parts.append(f"q{args.num_queries}")

    return "_".join(parts)


def create_editing_dirname(
    model: str,
    temperature: float,
    prompt_key: str,
    max_tokens: int,
    reasoning_effort: str = None,
    num_trajectories_batch: Optional[int] = None,
    synthesis_model: Optional[str] = None,
    synthesis_temperature: Optional[float] = None,
    synthesis_prompt_key: Optional[str] = None,
    synthesis_max_tokens: Optional[int] = None,
    synthesis_reasoning_effort: Optional[str] = None,
) -> str:
    """Create directory name from editing/improvement hyperparameters."""
    model_short = _shorten_model_name(model)
    supports_temp = not (model_short.startswith("o") or model_short.startswith("gpt5"))

    parts = [model_short]
    if supports_temp:
        parts.append(f"t{str(temperature).replace('.', '')}")
    elif reasoning_effort is not None:
        parts.append(reasoning_effort)

    parts.append(prompt_key)
    parts.append(str(max_tokens))

    if num_trajectories_batch is not None:
        parts.append(f"b{num_trajectories_batch}")

    if synthesis_model is not None:
        synthesis_short = _shorten_model_name(synthesis_model)
        synthesis_supports_temp = not (
            synthesis_short.startswith("o") or synthesis_short.startswith("gpt5")
        )

        parts.append(f"sm{synthesis_short}")
        if synthesis_supports_temp and synthesis_temperature is not None:
            parts.append(f"st{str(synthesis_temperature).replace('.', '')}")
        elif not synthesis_supports_temp and synthesis_reasoning_effort is not None:
            parts.append(f"sr{synthesis_reasoning_effort}")

        if synthesis_prompt_key is not None:
            parts.append(f"sp{synthesis_prompt_key}")
        if synthesis_max_tokens is not None:
            parts.append(f"sx{synthesis_max_tokens}")

    return "_".join(parts)


def validate_hyperparams_match(config_path: Path, generation_args: Dict, editing_args: Dict) -> None:
    """Validate CLI hyperparameters against an improvement config path.

    This mirrors BFCL's resume behavior: when resuming from an existing
    improvement config, generation and editing hyperparameters must match the
    directory names in that path.
    """
    config_path = Path(config_path)
    parts = config_path.parts

    if "improvements" not in parts:
        return

    imp_idx = parts.index("improvements")
    if imp_idx < 1 or imp_idx + 1 >= len(parts):
        return

    path_generation_dir = parts[imp_idx - 1]
    path_editing_dir = parts[imp_idx + 1]

    generation_ns = SimpleNamespace(**generation_args)
    generation_ns._resolved_prompt_key = resolve_prompt_key(
        prompt_key=generation_args.get("prompt_key"),
        include_get_document=bool(generation_args.get("include_get_document")),
    )
    generation_ns._uses_faiss = config_uses_faiss(config_path)
    expected_generation_dir = create_generation_dirname(
        generation_ns,
        config_name=parse_config_name(config_path),
    )

    expected_editing_dir = create_editing_dirname(
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

    mismatches = []
    if path_generation_dir != expected_generation_dir:
        expected_detail = f"expected '{expected_generation_dir}'"
        mismatches.append(
            "Generation hyperparameters: "
            f"path has '{path_generation_dir}', {expected_detail}"
        )
    if path_editing_dir != expected_editing_dir:
        mismatches.append(
            f"Editing hyperparameters: path has '{path_editing_dir}', got '{expected_editing_dir}'"
        )

    if mismatches:
        raise ValueError(
            "Hyperparameter mismatch when resuming from improvement config:\n"
            + "\n".join(f"  - {line}" for line in mismatches)
        )


def build_output_folder(args, config_name: str, mode: str = "shared_tools") -> Path:
    """Build output directory path for run outputs."""
    hyperparam_dirname = create_generation_dirname(args, config_name=config_name)

    if mode == "shared_tools":
        return Path(args.output_dir) / "shared_tools" / config_name / hyperparam_dirname

    if mode == "single_searcher":
        if args.filter_category:
            scope = args.filter_category
        elif args.filter_domain:
            scope = args.filter_domain.replace(".", "_").replace("/", "_")
        else:
            scope = "all"
        return Path(args.output_dir) / "single_searcher" / args.searcher_type / scope / hyperparam_dirname

    raise ValueError(f"Unknown mode: {mode}")


def get_next_version(base_path: Path, is_improvement: bool = False) -> int:
    """Get next available version number for base or improvement runs."""
    if not base_path.exists():
        return 1 if is_improvement else 0

    if is_improvement:
        versions = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("v"):
                try:
                    versions.append(int(item.name[1:]))
                except ValueError:
                    pass
        return max(versions) + 1 if versions else 1

    versions = []
    for item in base_path.iterdir():
        if item.is_file() and item.name.endswith("_results.json"):
            match = re.match(r"v(\d+)_results\.json", item.name)
            if match:
                versions.append(int(match.group(1)))
    return max(versions) + 1 if versions else 0


def detect_improvement_context(config_path: Path) -> Tuple[bool, Optional[Path], Optional[int]]:
    """Detect if a config path belongs to an improvements directory."""
    config_path = Path(config_path)
    parts = config_path.parts

    if "improvements" not in parts:
        return False, None, None

    imp_idx = parts.index("improvements")
    for i in range(imp_idx + 1, len(parts)):
        if parts[i].startswith("v") and parts[i][1:].isdigit():
            version = int(parts[i][1:])
            base_path = Path(*parts[:i])
            return True, base_path, version

    return False, None, None


def get_base_run_path(path: Path) -> Optional[Path]:
    """Get base run path for a result/config path.

    Returns runs/.../{method}/shared_tools/{config_name}/{generation_hypers} when possible.
    """
    path = Path(path)
    parts = path.parts

    if "improvements" in parts:
        imp_idx = parts.index("improvements")
        return Path(*parts[:imp_idx])

    if "runs" in parts and "BrowseCompPlus" in parts:
        bc_idx = parts.index("BrowseCompPlus")
        offset = 1
        if bc_idx + 1 < len(parts) and parts[bc_idx + 1] in KNOWN_METHODS:
            offset = 2

        st_idx = bc_idx + offset
        # .../shared_tools/{config}/{generation_hypers}/...
        if st_idx < len(parts) and parts[st_idx] == "shared_tools" and st_idx + 2 < len(parts):
            return Path(*parts[:st_idx + 3])

    return None
