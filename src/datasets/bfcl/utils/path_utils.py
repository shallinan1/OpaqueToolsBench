"""
Path management utilities for BFCL file structure.

Directory structure:
- configs/config1.json, config2.json, etc.
- runs/bfcl/{method}/{config_name}/{hyperparam_dir}/
  where {method} is "ours"
  - v0_results.json, v0_metadata.json, v0_scored.json (base runs)
  - improvements/{editing_hypers}/
    - v1/config.json, results.json, metadata.json, scored.json
    - v2/config.json, results.json, metadata.json, scored.json
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Known method subdirectories under runs/bfcl/
KNOWN_METHODS = {"ours"}


def parse_config_name(config_path: Path) -> str:
    """Extract config name from path.

    Examples:
        configs/config1.json -> config1
        src/datasets/bfcl/tool_configs/executable_multiple_function_name[all:increasing_number]_param[all:remove_all]_config.json
            -> executable_multiple_function_name[all:increasing_number]_param[all:remove_all]
        runs/bfcl/ours/config1/gpt5.../improvements/.../v1/config.json -> config1
    """
    config_path = Path(config_path)

    # If it's in a runs directory, extract from path
    parts = config_path.parts
    if "runs" in parts and "bfcl" in parts:
        runs_idx = parts.index("runs")
        bfcl_idx = parts.index("bfcl")
        # Check if this is runs/bfcl/{method}/{config_name}/... or runs/bfcl/{config_name}/...
        if bfcl_idx == runs_idx + 1 and bfcl_idx + 1 < len(parts):
            next_part = parts[bfcl_idx + 1]
            if next_part in KNOWN_METHODS and bfcl_idx + 2 < len(parts):
                return parts[bfcl_idx + 2]
            return next_part

    # For any other case (including tool_configs), use the filename stem
    stem = config_path.stem
    # Remove _config suffix if present
    if stem.endswith("_config"):
        return stem[:-7]
    return stem


def create_generation_dirname(args) -> str:
    """Create directory name from generation hyperparameters.

    Format:
    - For models with temp: {model}_{temp}_{top_p}_{tool_choice}_{max_tokens}_{prompt_key}_seed{seed}
    - For o/gpt-5 models: {model}_{reasoning_effort}_{tool_choice}_{max_tokens}_{prompt_key}_seed{seed}

    Example:
    - gpt4_t001_p10_req_8k_basic_seed0
    - gpt5_medium_req_8k_basic_seed0
    - gpt51_none_req_8k_basic_seed0
    - o3_medium_req_16k_basic_seed42
    """
    # Clean model name (keep gpt-5 and gpt-5-mini separate, remove dates like -2024-08-06)
    model = args.model.split('/')[-1] # Remove any path prefix and whitespace
    if model == 'gpt-5':
        model_short = 'gpt5'
    elif model == 'gpt-5.1':
        model_short = 'gpt51'
    elif model == 'gpt-5-mini':
        model_short = 'gpt5mini'
    elif '-20' in model and not model.startswith('o'):
        # For models like gpt-4o-2024-08-06, take everything before the date
        model_short = model.split('-20')[0].replace('-', '').replace('_', '')
    else:
        model_short = model.replace('-', '').replace('_', '')

    # Check if model supports temperature (gpt-5, gpt-5-mini, and o-series don't)
    supports_temp = not (model_short.startswith('o') or model_short.startswith('gpt5'))

    # Build parts list
    parts = [model_short]

    if supports_temp:
        # Format temperature (0.001 -> t001)
        temp = f"t{str(args.temperature).replace('.', '')}"
        parts.append(temp)

        # Format top_p (1.0 -> p10)
        top_p = f"p{str(args.top_p).replace('.', '')}" if hasattr(args, 'top_p') and args.top_p is not None else "p10"
        parts.append(top_p)

    # Add reasoning_effort for models that support it (gpt-5, o-series, and gpt-oss via Together AI)
    if not supports_temp or (hasattr(args, 'together') and args.together):
        if hasattr(args, 'reasoning_effort') and args.reasoning_effort is not None:
            parts.append(args.reasoning_effort)

    # Shorten tool_choice
    tool_choice_map = {
        "required": "req",
        "auto": "auto",
        "none": "none"
    }
    tool = tool_choice_map.get(args.tool_choice, args.tool_choice[:3])
    parts.append(tool)

    # Format max_tokens
    max_tokens = args.max_tokens if hasattr(args, 'max_tokens') and args.max_tokens else 8192
    parts.append(str(max_tokens))

    # Get prompt key
    prompt_key = args.prompt_key if hasattr(args, 'prompt_key') else "must_call_tool"
    parts.append(prompt_key)

    seed = args.seed if hasattr(args, 'seed') else 0
    parts.append(f"seed{seed}")

    return "_".join(parts)


def create_editing_dirname(model: str, temperature: float, prompt_key: str, max_tokens: int, reasoning_effort: str = None) -> str:
    """Create directory name from editing/improvement hyperparameters.

    Format:
    - For models with temp: {model}_{temp}_{prompt_key}_{max_tokens}
    - For o/gpt-5 models: {model}_{reasoning_effort}_{prompt_key}_{max_tokens}

    Example:
    - gpt4o_t07_reflective_8k
    - gpt5_medium_reflective_8k
    - o3_medium_basic_improved_16k
    """
    # Clean model name (keep gpt-5 and gpt-5-mini separate, remove dates like -2024-08-06)
    model_clean = model.split('/')[-1]  # Remove any path prefix
    if model_clean == 'gpt-5':
        model_short = 'gpt5'
    elif model_clean == 'gpt-5.1':
        model_short = 'gpt51'
    elif model_clean == 'gpt-5-mini':
        model_short = 'gpt5mini'
    elif '-20' in model_clean and not model_clean.startswith('o'):
        # For models like gpt-4o-2024-08-06, take everything before the date
        model_short = model_clean.split('-20')[0].replace('-', '').replace('_', '')
    else:
        model_short = model_clean.replace('-', '').replace('_', '')

    # Check if model supports temperature (gpt-5, gpt-5-mini, and o-series don't)
    supports_temp = not (model_short.startswith('o') or model_short.startswith('gpt5'))

    # Build parts list
    parts = [model_short]

    if supports_temp:
        # Format temperature
        temp = f"t{str(temperature).replace('.', '')}"
        parts.append(temp)

    # Add reasoning_effort for models that support it
    if not supports_temp and reasoning_effort is not None:
        parts.append(reasoning_effort)

    # Add prompt key
    parts.append(prompt_key)

    # Format max_tokens
    parts.append(str(max_tokens))

    return "_".join(parts)


def get_next_version(base_path: Path, is_improvement: bool = False) -> int:
    """Get the next version number for a given path.

    Args:
        base_path: Either the hyperparam directory (for base runs) or
                   the editing hyperparam directory (for improvements)
        is_improvement: Whether this is for an improvement version

    Returns:
        Next version number (0 for first base run, 1+ for improvements)
    """
    if not base_path.exists():
        return 1 if is_improvement else 0

    if is_improvement:
        # Look for v1/, v2/, etc. directories
        versions = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                try:
                    versions.append(int(item.name[1:]))
                except ValueError:
                    pass
        return max(versions) + 1 if versions else 1
    else:
        # Look for v0_results.json, v1_results.json, etc.
        versions = []
        for item in base_path.iterdir():
            if item.is_file() and item.name.endswith('_results.json'):
                match = re.match(r'v(\d+)_results\.json', item.name)
                if match:
                    versions.append(int(match.group(1)))
        return max(versions) + 1 if versions else 0


def detect_improvement_context(config_path: Path) -> Tuple[bool, Optional[Path], Optional[int]]:
    """Detect if a config is from an improvements directory.

    Returns:
        (is_improvement, improvement_base_path, version)

    Example:
        config_path: runs/bfcl/ours/config1/gpt5.../improvements/gpt4o.../v1/config.json
        Returns: (True, Path(".../improvements/gpt4o..."), 1)
    """
    config_path = Path(config_path)
    parts = config_path.parts

    if "improvements" in parts:
        imp_idx = parts.index("improvements")

        # Find version directory (v1, v2, etc.)
        for i in range(imp_idx + 1, len(parts)):
            if parts[i].startswith('v'):
                try:
                    version = int(parts[i][1:])
                    # Construct base path up to version directory's parent
                    base_path = Path(*parts[:i])
                    return True, base_path, version
                except ValueError:
                    pass

    return False, None, None


def get_base_run_path(config_path: Path) -> Optional[Path]:
    """Get the base run directory for a given config.

    Example:
        config_path: runs/bfcl/ours/config1/gpt5.../improvements/gpt4o.../v1/config.json
        Returns: Path("runs/bfcl/ours/config1/gpt5...")
    """
    config_path = Path(config_path)
    parts = config_path.parts

    if "improvements" in parts:
        imp_idx = parts.index("improvements")
        return Path(*parts[:imp_idx])
    elif "bfcl" in parts:
        # Already a base run path: runs/bfcl/{method}/{config}/{hypers} or runs/bfcl/{config}/{hypers}
        bfcl_idx = parts.index("bfcl")
        # Check if next part after bfcl is a known method
        offset = 3  # default: runs/bfcl/{config}/{hypers}
        if bfcl_idx + 1 < len(parts) and parts[bfcl_idx + 1] in KNOWN_METHODS:
            offset = 4  # runs/bfcl/{method}/{config}/{hypers}
        if len(parts) > bfcl_idx + offset - 1:
            return Path(*parts[:bfcl_idx + offset])

    return None


def parse_editing_dirname(dirname: str) -> Dict[str, Any]:
    """Parse editing hyperparameters from directory name.

    Example:
        gpt4o_t07_reflective_8192 -> {
            "model": "gpt4o",
            "temperature": 0.7,
            "reasoning_effort": None,
            "prompt_key": "reflective",
            "max_tokens": 8192
        }
        gpt5_medium_reflective_8192 -> {
            "model": "gpt-5",
            "temperature": None,
            "reasoning_effort": "medium",
            "prompt_key": "reflective",
            "max_tokens": 8192
        }
        gpt5_reflective_8192 -> {  # backwards compatible
            "model": "gpt-5",
            "temperature": None,
            "reasoning_effort": "medium",
            "prompt_key": "reflective",
            "max_tokens": 8192
        }
    """
    parts = dirname.split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid editing directory name: {dirname}")

    model_raw = parts[0]
    model = model_raw
    if model_raw == 'gpt5':
        model = 'gpt-5'
    elif model_raw in ('gpt51', 'gpt5.1'):
        model = 'gpt-5.1'
    elif model_raw == 'gpt5mini':
        model = 'gpt-5-mini'

    supports_temp = not (model_raw.startswith('o') or model_raw.startswith('gpt5'))
    reasoning_effort = None

    if supports_temp and len(parts) >= 4:
        # Parse temperature (t07 -> 0.7)
        temp_str = parts[1]
        if not temp_str.startswith('t'):
            raise ValueError(f"Invalid temperature format: {temp_str}")
        temp = float('0.' + temp_str[1:]) if len(temp_str[1:]) < 3 else float(temp_str[1:])

        # Parse max_tokens
        max_tokens = int(parts[-1])

        # Prompt key is everything in between
        prompt_key = '_'.join(parts[2:-1])
    else:
        # No temperature for o/gpt-5 models
        temp = None
        max_tokens = int(parts[-1])

        # Check if next part after model is a reasoning_effort value
        valid_reasoning_efforts = {"none", "minimal", "low", "medium", "high"}
        idx = 1
        if idx < len(parts) and parts[idx] in valid_reasoning_efforts:
            reasoning_effort = parts[idx]
            idx += 1

        # Prompt key is everything between current position and max_tokens
        prompt_key = '_'.join(parts[idx:-1])

    # Backwards compatibility: infer reasoning_effort default for old dirs
    if reasoning_effort is None and not supports_temp:
        if model == 'gpt-5.1':
            reasoning_effort = 'none'
        else:
            reasoning_effort = 'medium'

    return {
        "model": model,
        "temperature": temp,
        "reasoning_effort": reasoning_effort,
        "prompt_key": prompt_key,
        "max_tokens": max_tokens
    }

def parse_generation_dirname(dirname: str) -> Dict[str, Any]:
    """Parse generation hyperparameters from directory name.

    Example:
        gpt4_t001_p10_req_8192_must_call_tool_seed0 -> {
            "model": "gpt-4",
            "temperature": 0.001,
            "top_p": 1.0,
            "tool_choice": "required",
            "max_tokens": 8192,
            "prompt_key": "must_call_tool",
            "reasoning_effort": None,
            "seed": 0
        }
        gpt5_medium_req_8192_must_call_tool_seed42 -> {
            "model": "gpt-5",
            "temperature": None,
            "top_p": None,
            "reasoning_effort": "medium",
            "tool_choice": "required",
            "max_tokens": 8192,
            "prompt_key": "must_call_tool",
            "seed": 42
        }
    """
    parts = dirname.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid generation directory name: {dirname}")

    # Parse model
    model_short = parts[0]
    if model_short == 'gpt5':
        model = 'gpt-5'
    elif model_short in ('gpt51', 'gpt5.1'):
        model = 'gpt-5.1'
    elif model_short == 'gpt5mini':
        model = 'gpt-5-mini'
    else:
        # Convert back to hyphenated form
        model = model_short

    # Check if model supports temperature
    supports_temp = not (model_short.startswith('o') or model_short.startswith('gpt5'))

    idx = 1
    temperature = None
    top_p = None
    reasoning_effort = None

    valid_reasoning_efforts = {"none", "minimal", "low", "medium", "high"}

    if supports_temp:
        # Parse temperature (t001 -> 0.001)
        if idx < len(parts) and parts[idx].startswith('t'):
            temp_str = parts[idx][1:]
            temperature = float('0.' + temp_str) if len(temp_str) < 3 else float(temp_str) / 1000
            idx += 1

        # Parse top_p (p10 -> 1.0)
        if idx < len(parts) and parts[idx].startswith('p'):
            top_p_str = parts[idx][1:]
            top_p = float(top_p_str) / 10
            idx += 1

    # Parse reasoning_effort if present (for o/gpt-5 models it comes instead of temp,
    # for Together AI models like gpt-oss it comes after temp/top_p)
    if idx < len(parts) and parts[idx] in valid_reasoning_efforts:
        reasoning_effort = parts[idx]
        idx += 1

    # Parse tool_choice
    tool_choice_map = {"req": "required", "auto": "auto", "none": "none"}
    tool_choice = tool_choice_map.get(parts[idx], parts[idx])
    idx += 1

    max_tokens = int(parts[idx])
    idx += 1

    # Parse seed (last part must be seed{number})
    assert parts[-1].startswith('seed'), f"Invalid generation directory name: {dirname} (missing seed)"

    seed_str = parts[-1][4:]  # Remove 'seed' prefix
    seed = int(seed_str)

    # Prompt key is everything between max_tokens and seed
    prompt_key = '_'.join(parts[idx:-1])

    # For backwards compatibility: if reasoning_effort is None (old directory format),
    # infer the default based on model
    if reasoning_effort is None:
        if model == 'gpt-5.1':
            reasoning_effort = 'none'  # GPT-5.1 OpenAI default
        elif model_short.startswith('o') or model_short.startswith('gpt5') or 'gptoss' in model_short:
            reasoning_effort = 'medium'  # GPT-5, gpt-5-mini, o-series, gpt-oss default

    return {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "prompt_key": prompt_key,
        "seed": seed
    }


def parse_cross_model_source(source_path: Path) -> Tuple[str, str, int]:
    """Parse a cross-model source config path to extract source identity.

    Expects a path from an iterate-improve run like:
        runs/bfcl/ours/{config}/{gen_hyper}/improvements/{edit_hyper}/v{N}/config.json

    Returns:
        (source_gen_dirname, source_edit_dirname, source_version)

    Raises:
        ValueError: If the path doesn't have the expected improvement structure
    """
    source_path = Path(source_path)
    parts = source_path.parts

    if "improvements" not in parts:
        raise ValueError(
            f"Cross-model source must be from an improvements directory, got: {source_path}"
        )

    imp_idx = parts.index("improvements")

    # The generation hyper directory is right before "improvements"
    if imp_idx < 1:
        raise ValueError(f"Cannot find generation hyper directory in: {source_path}")
    source_gen_dirname = parts[imp_idx - 1]

    # The editing hyper directory is right after "improvements"
    if imp_idx + 1 >= len(parts):
        raise ValueError(f"Cannot find editing hyper directory in: {source_path}")
    source_edit_dirname = parts[imp_idx + 1]

    # Find the version directory (v1, v2, etc.)
    source_version = None
    for i in range(imp_idx + 2, len(parts)):
        if parts[i].startswith('v') and parts[i][1:].isdigit():
            source_version = int(parts[i][1:])
            break

    if source_version is None:
        raise ValueError(f"Cannot find version directory in: {source_path}")

    return source_gen_dirname, source_edit_dirname, source_version


def validate_improvement_path(result_dir: Path, version: Optional[int] = None) -> Tuple[Path, int]:
    """Validate and get the correct improvement path.

    Args:
        result_dir: Directory containing results to improve from
        version: Expected version (if provided)

    Returns:
        (results_file_path, detected_version)

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If version mismatch
    """
    result_dir = Path(result_dir)

    # Check if this is already an improvement directory (has config.json)
    if (result_dir / "config.json").exists(): # This is an improvement directory
        results_file = result_dir / "results.json"
        scored_file = result_dir / "scored.json"
    else: # This is a base run directory - look for v{N}_results.json
        results_files = list(result_dir.glob("v*_results.json"))
        scored_files = list(result_dir.glob("v*_scored.json"))

        if not scored_files:
            raise FileNotFoundError(f"No scored files found in {result_dir}")

        # Get the latest version
        latest_scored = max(scored_files, key=lambda f: int(re.match(r'v(\d+)_scored\.json', f.name).group(1)))
        detected_version = int(re.match(r'v(\d+)_scored\.json', latest_scored.name).group(1))

        if version is not None and version != detected_version:
            raise ValueError(f"Version mismatch: expected v{version}, found v{detected_version}")

        return latest_scored, detected_version

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not scored_file.exists():
        raise FileNotFoundError(f"Scored file not found: {scored_file}")

    # For improvement directories, extract version from path
    parts = result_dir.parts
    for part in parts:
        if part.startswith('v'):
            try:
                detected_version = int(part[1:])
                if version is not None and version != detected_version:
                    raise ValueError(f"Version mismatch: expected v{version}, found v{detected_version}")
                return scored_file, detected_version
            except ValueError:
                pass

    return scored_file, 0


def validate_hyperparams_match(config_path: Path, generation_args: Dict, editing_args: Dict) -> None:
    """Validate that provided hyperparameters match those in the improvement path.

    Only validates when resuming from an improvement config (has v[number] in path).

    Args:
        config_path: Path to the config file
        generation_args: Generation hyperparameters from command line
        editing_args: Editing hyperparameters from command line

    Raises:
        ValueError: If hyperparameters don't match the path structure
    """
    config_path = Path(config_path)
    parts = config_path.parts

    # Check if this is an improvement config (has v[number] in path)
    has_version = any(part.startswith('v') and part[1:].isdigit() for part in parts)
    if not has_version:
        # Not resuming from improvement, no validation needed
        return

    # Extract the generation and editing directory names from the path
    generation_dirname = None
    editing_dirname = None

    for i, part in enumerate(parts):
        if part == 'bfcl':
            # Skip known method dir if present: bfcl/{method}/{config}/{gen_dir} vs bfcl/{config}/{gen_dir}
            gen_offset = 2
            if i + 1 < len(parts) and parts[i + 1] in KNOWN_METHODS:
                gen_offset = 3
            if i + gen_offset < len(parts):
                generation_dirname = parts[i + gen_offset]
        elif part == 'improvements' and i + 1 < len(parts): # Next part is editing dirname
            editing_dirname = parts[i + 1]

    if not generation_dirname or not editing_dirname: # Can't extract hyperparams from path, skip validation
        return

    # Parse hyperparameters from directory names
    try:
        path_gen_params = parse_generation_dirname(generation_dirname)
        path_edit_params = parse_editing_dirname(editing_dirname)
    except (ValueError, IndexError) as e: # Can't parse directory names, skip validation
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not parse hyperparameters from path: {e}")
        return

    # Validate generation parameters
    mismatches = []

    # Check generation model
    if path_gen_params['model'] != generation_args['model']:
        mismatches.append(f"Generation model: path has '{path_gen_params['model']}', but got '{generation_args['model']}'")

    # Check generation temperature (if model supports it)
    if path_gen_params['temperature'] is not None:
        if abs(path_gen_params['temperature'] - generation_args['temperature']) > 0.0001:
            mismatches.append(f"Generation temperature: path has {path_gen_params['temperature']}, but got {generation_args['temperature']}")

    # Check reasoning_effort (if present in path)
    if path_gen_params.get('reasoning_effort') is not None:
        if path_gen_params['reasoning_effort'] != generation_args.get('reasoning_effort'):
            mismatches.append(f"Generation reasoning_effort: path has '{path_gen_params['reasoning_effort']}', but got '{generation_args.get('reasoning_effort')}'")

    # Check tool choice
    if path_gen_params['tool_choice'] != generation_args.get('tool_choice', 'required'):
        mismatches.append(f"Tool choice: path has '{path_gen_params['tool_choice']}', but got '{generation_args.get('tool_choice', 'required')}'")

    # Check prompt key
    if path_gen_params['prompt_key'] != generation_args['prompt_key']:
        mismatches.append(f"Generation prompt key: path has '{path_gen_params['prompt_key']}', but got '{generation_args['prompt_key']}'")

    # Check max tokens
    if path_gen_params['max_tokens'] != generation_args.get('max_tokens', 8192):
        mismatches.append(f"Generation max tokens: path has {path_gen_params['max_tokens']}, but got {generation_args.get('max_tokens', 8192)}")

    # Validate editing parameters
    if path_edit_params['model'] != editing_args['model']:
        mismatches.append(f"Editing model: path has '{path_edit_params['model']}', but got '{editing_args['model']}'")

    if path_edit_params['temperature'] is not None:
        if abs(path_edit_params['temperature'] - editing_args['temperature']) > 0.0001:
            mismatches.append(f"Editing temperature: path has {path_edit_params['temperature']}, but got {editing_args['temperature']}")

    if path_edit_params['prompt_key'] != editing_args['prompt_key']:
        mismatches.append(f"Editing prompt key: path has '{path_edit_params['prompt_key']}', but got '{editing_args['prompt_key']}'")

    if path_edit_params['max_tokens'] != editing_args.get('max_tokens', 8192):
        mismatches.append(f"Editing max tokens: path has {path_edit_params['max_tokens']}, but got {editing_args.get('max_tokens', 8192)}")

    if mismatches:
        error_msg = "Hyperparameter mismatch when resuming from improvement config:\n" + "\n".join(f"  - {m}" for m in mismatches)
        raise ValueError(error_msg)
