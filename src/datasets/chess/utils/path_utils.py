"""
Path management utilities for chess file structure.

Directory structure:
- runs/chess/{method}/shared_tools/{config_name}/{split}/{hyperparams}/{white}/vs_{black}/
  where {method} is "tool_observer"
  - v0_trajectories.json, v0_metadata.json, v0_scored.json, v0_token_usage.json
  - improvements/{editing_hypers}/
    - v1/llm_responses.json, config.json, synthesis_*.json
    - v2/...
- runs/chess/{method}/individual_tools/{config_name}/{hyperparams}/
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Known method subdirectories under runs/chess/
KNOWN_METHODS = {"tool_observer"}


def parse_config_name(config_path: Path) -> str:
    """Extract config name from path.

    Examples:
        shared_tools/elo_tools_obfuscated.json -> elo_tools_obfuscated
        runs/chess/tool_observer/shared_tools/elo_tools_obfuscated/.../v1/config.json
            -> elo_tools_obfuscated
    """
    config_path = Path(config_path)
    parts = config_path.parts

    # If it's in a runs directory, extract config_name from path
    if "runs" in parts and "chess" in parts:
        chess_idx = parts.index("chess")
        # runs/chess/{method}/shared_tools/{config_name}/... or
        # runs/chess/{method}/individual_tools/{config_name}/...
        offset = 1  # start after "chess"
        if chess_idx + 1 < len(parts) and parts[chess_idx + 1] in KNOWN_METHODS:
            offset = 2  # skip method dir
        mode_idx = chess_idx + offset
        if mode_idx < len(parts) and parts[mode_idx] in ("shared_tools", "individual_tools"):
            config_name_idx = mode_idx + 1
            if config_name_idx < len(parts):
                return parts[config_name_idx]

    # For regular config files, use the stem
    return config_path.stem


def create_generation_dirname(args) -> str:
    """Create directory name from generation hyperparameters.

    Matches run.py's inline path construction. Format:
        {model_short}[_t{temp}][_{reasoning}]_{tool_choice}_{prompt_key}_{num_traj}_{max_moves}[_mirror][_hist]

    Examples:
        gpt4o_t10_auto_basic_3_500
        gpt5_medium_required_basic_3_500_mirror
    """
    model_short = _shorten_model_name(args.model)
    is_reasoning = model_short.startswith('o') or model_short.startswith('gpt5')

    hyperparam_parts = [model_short]

    # Temperature for non-reasoning models
    if not is_reasoning:
        hyperparam_parts.append(f"t{str(args.temperature).replace('.', '')}")

    # Reasoning effort for reasoning models
    if hasattr(args, 'reasoning_effort') and args.reasoning_effort:
        if is_reasoning:
            hyperparam_parts.append(f"{args.reasoning_effort}")

    hyperparam_parts.append(f"{args.tool_choice}")
    hyperparam_parts.append(args.prompt_key)
    hyperparam_parts.append(f"{args.num_trajectories}")
    hyperparam_parts.append(f"{args.max_moves}")

    if args.mirror:
        hyperparam_parts.append("mirror")
    if args.include_history:
        hyperparam_parts.append("hist")

    return "_".join(hyperparam_parts)


def _shorten_model_name(model: str) -> str:
    """Shorten model name for directory naming, matching BFCL convention.

    Examples:
        gpt-5 -> gpt5
        gpt-5-mini -> gpt5mini
        gpt-5.1 -> gpt51
        gpt-4o-2024-08-06 -> gpt4o
        o3 -> o3
    """
    model_clean = model.split('/')[-1]  # Remove any path prefix
    if model_clean == 'gpt-5':
        return 'gpt5'
    elif model_clean == 'gpt-5.1':
        return 'gpt51'
    elif model_clean == 'gpt-5-mini':
        return 'gpt5mini'
    elif '-20' in model_clean and not model_clean.startswith('o'):
        return model_clean.split('-20')[0].replace('-', '').replace('_', '')
    else:
        return model_clean.replace('-', '').replace('_', '')


def create_editing_dirname(model: str, temperature: float, prompt_key: str,
                           max_tokens: int, show_agent_values: bool = False,
                           reasoning_effort: str = None) -> str:
    """Create directory name for editing/improvement hyperparameters.

    Format:
    - Models with temperature: {model_short}_t{temp}_{prompt_key}_{max_tokens}[_showvalues]
    - Reasoning models (o/gpt-5): {model_short}_{reasoning_effort}_{prompt_key}_{max_tokens}[_showvalues]

    Examples:
        gpt4o_t07_detailed_8192
        gpt5_medium_detailed_8192_showvalues
    """
    model_short = _shorten_model_name(model)
    is_reasoning = model_short.startswith('o') or model_short.startswith('gpt5')

    parts = [model_short]
    if is_reasoning and reasoning_effort is not None:
        parts.append(reasoning_effort)
    else:
        parts.append(f"t{str(temperature).replace('.', '')}")
    parts.append(prompt_key)
    parts.append(str(max_tokens))

    dirname = "_".join(parts)
    if show_agent_values:
        dirname += "_showvalues"
    return dirname


def build_output_folder(args, config_name: str, mode: str) -> Path:
    """Build the full output folder path for a run.

    Args:
        args: Parsed command line arguments (needs model, temperature, tool_choice,
              prompt_key, num_trajectories, max_moves, mirror, include_history,
              white_type, black_type, split, num_queries, output_dir)
        config_name: Name of the tool configuration
        mode: Either "tool_config" or "tool_set"

    Returns:
        Path to the output directory
    """
    from src.datasets.chess.probabilistic_player import format_probabilities_for_path

    hyperparam_dirname = create_generation_dirname(args)

    if mode == "tool_config":
        return Path(args.output_dir) / "individual_tools" / config_name / hyperparam_dirname
    else:
        # Build split name
        split_name = args.split if args.split else "custom"
        if args.num_queries:
            split_name = f"{split_name}_{args.num_queries}q"

        # Build player directory names
        white_desc = args.white_type
        if args.white_type == "probabilistic" and hasattr(args, '_white_tool_probs'):
            white_desc = f"prob_{format_probabilities_for_path(args._white_tool_probs)}"

        black_desc = args.black_type
        if args.black_type == "probabilistic" and hasattr(args, '_black_tool_probs'):
            black_desc = f"prob_{format_probabilities_for_path(args._black_tool_probs)}"

        return (Path(args.output_dir) / "shared_tools" / config_name / split_name /
                hyperparam_dirname / white_desc / f"vs_{black_desc}")


def get_next_version(base_path: Path, is_improvement: bool = False) -> int:
    """Get the next version number for a given path.

    Args:
        base_path: Either the run directory (for base runs) or
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
        # Look for v0_trajectories.json, v1_trajectories.json, etc.
        versions = []
        for item in base_path.iterdir():
            if item.is_file() and item.name.endswith('_trajectories.json'):
                match = re.match(r'v(\d+)_trajectories\.json', item.name)
                if match:
                    versions.append(int(match.group(1)))
        return max(versions) + 1 if versions else 0


def detect_improvement_context(config_path: Path) -> Tuple[bool, Optional[Path], Optional[int]]:
    """Detect if a config is from an improvements directory.

    Returns:
        (is_improvement, improvement_base_path, version)

    Example:
        config_path: runs/chess/tool_observer/shared_tools/.../improvements/gpt-5_detailed_0.7_8192/v1/config.json
        Returns: (True, Path(".../improvements/gpt-5_detailed_0.7_8192"), 1)
    """
    config_path = Path(config_path)
    parts = config_path.parts

    if "improvements" in parts:
        imp_idx = parts.index("improvements")

        # Find version directory (v1, v2, etc.)
        for i in range(imp_idx + 1, len(parts)):
            if parts[i].startswith('v') and parts[i][1:].isdigit():
                version = int(parts[i][1:])
                # Base path is up to (but not including) the version dir
                base_path = Path(*parts[:i])
                return True, base_path, version

    return False, None, None


def get_base_run_path(config_path: Path) -> Optional[Path]:
    """Get the base run directory for a given config.

    Example:
        config_path: runs/chess/tool_observer/shared_tools/.../improvements/.../v1/config.json
        Returns: Path("runs/chess/tool_observer/shared_tools/.../{hyperparams}/{white}/vs_{black}")
    """
    config_path = Path(config_path)
    parts = config_path.parts

    if "improvements" in parts:
        imp_idx = parts.index("improvements")
        return Path(*parts[:imp_idx])

    return None


def parse_editing_dirname(dirname: str) -> Dict[str, Any]:
    """Parse editing hyperparameters from directory name.

    Examples:
        gpt4o_t07_detailed_8192 -> {"model": "gpt4o", "temperature": 0.7, ...}
        gpt5_medium_detailed_8192_showvalues -> {"model": "gpt-5", "reasoning_effort": "medium", ...}
    """
    # Strip _showvalues suffix if present
    show_agent_values = dirname.endswith("_showvalues")
    if show_agent_values:
        dirname = dirname[: -len("_showvalues")]

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
        temp_str = parts[1]
        if not temp_str.startswith('t'):
            raise ValueError(f"Invalid temperature format: {temp_str}")
        temp = float('0.' + temp_str[1:]) if len(temp_str[1:]) < 3 else float(temp_str[1:])
        max_tokens = int(parts[-1])
        prompt_key = '_'.join(parts[2:-1])
    else:
        temp = None
        max_tokens = int(parts[-1])
        valid_reasoning_efforts = {"none", "minimal", "low", "medium", "high"}
        idx = 1
        if idx < len(parts) and parts[idx] in valid_reasoning_efforts:
            reasoning_effort = parts[idx]
            idx += 1
        prompt_key = '_'.join(parts[idx:-1])

    if reasoning_effort is None and not supports_temp:
        reasoning_effort = 'medium'

    return {
        "model": model,
        "temperature": temp,
        "reasoning_effort": reasoning_effort,
        "prompt_key": prompt_key,
        "max_tokens": max_tokens,
        "show_agent_values": show_agent_values,
    }


def parse_generation_dirname(dirname: str) -> Dict[str, Any]:
    """Parse generation hyperparameters from directory name.

    Format: {model}[_t{temp}][_{reasoning}]_{tool_choice}_{prompt_key}_{num_traj}_{max_moves}[_mirror][_hist]
    Examples:
        gpt5_low_auto_optimized_single_1_10_mirror_hist
        gpt4o_t10_auto_basic_3_500
    """
    parts = dirname.split('_')
    if len(parts) < 4:
        raise ValueError(f"Invalid generation directory name: {dirname}")

    # Strip trailing flags
    mirror = "mirror" in parts
    include_history = "hist" in parts
    if include_history:
        parts = parts[:parts.index("hist")]
    if mirror and "mirror" in parts:
        parts = parts[:parts.index("mirror")]

    model_raw = parts[0]
    model = model_raw
    if model_raw == 'gpt5':
        model = 'gpt-5'
    elif model_raw in ('gpt51', 'gpt5.1'):
        model = 'gpt-5.1'
    elif model_raw == 'gpt5mini':
        model = 'gpt-5-mini'

    supports_temp = not (model_raw.startswith('o') or model_raw.startswith('gpt5'))

    idx = 1
    temperature = None
    reasoning_effort = None

    if supports_temp and parts[idx].startswith('t'):
        temp_str = parts[idx][1:]
        temperature = float('0.' + temp_str) if len(temp_str) < 3 else float(temp_str)
        idx += 1

    if not supports_temp:
        valid_efforts = {"none", "minimal", "low", "medium", "high"}
        if idx < len(parts) and parts[idx] in valid_efforts:
            reasoning_effort = parts[idx]
            idx += 1

    # Remaining: tool_choice, prompt_key (may have underscores), num_traj, max_moves
    # Last two are integers
    max_moves = int(parts[-1])
    num_trajectories = int(parts[-2])
    tool_choice = parts[idx]
    idx += 1
    prompt_key = '_'.join(parts[idx:-2])

    return {
        "model": model,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "tool_choice": tool_choice,
        "prompt_key": prompt_key,
        "num_trajectories": num_trajectories,
        "max_moves": max_moves,
        "mirror": mirror,
        "include_history": include_history,
    }


def validate_hyperparams_match(config_path: Path, generation_args: Dict, editing_args: Dict) -> None:
    """Validate that provided hyperparameters match those in the improvement path.

    Checks both generation and editing hyperparameters against the directory structure.
    Path structure: .../{gen_dirname}/{white}/vs_{black}/improvements/{edit_dirname}/v{N}/...

    Raises:
        ValueError: If hyperparameters don't match the path structure
    """
    config_path = Path(config_path)
    parts = config_path.parts

    if "improvements" not in parts:
        return

    imp_idx = parts.index("improvements")
    mismatches = []

    # Validate editing hyperparameters
    if imp_idx + 1 < len(parts):
        editing_dirname = parts[imp_idx + 1]
        try:
            edit_params = parse_editing_dirname(editing_dirname)
        except (ValueError, IndexError):
            edit_params = None

        if edit_params:
            if edit_params['model'] != editing_args['model']:
                mismatches.append(f"Editing model: path has '{edit_params['model']}', got '{editing_args['model']}'")
            if edit_params['temperature'] is not None:
                if abs(edit_params['temperature'] - editing_args['temperature']) > 0.0001:
                    mismatches.append(f"Editing temperature: path has {edit_params['temperature']}, got {editing_args['temperature']}")
            if edit_params['prompt_key'] != editing_args['prompt_key']:
                mismatches.append(f"Editing prompt key: path has '{edit_params['prompt_key']}', got '{editing_args['prompt_key']}'")
            if edit_params['max_tokens'] != editing_args.get('max_tokens', 8192):
                mismatches.append(f"Editing max tokens: path has {edit_params['max_tokens']}, got {editing_args.get('max_tokens', 8192)}")
            if edit_params.get('reasoning_effort') is not None:
                if edit_params['reasoning_effort'] != editing_args.get('reasoning_effort'):
                    mismatches.append(f"Editing reasoning_effort: path has '{edit_params['reasoning_effort']}', got '{editing_args.get('reasoning_effort')}'")
            if edit_params.get('show_agent_values', False) != editing_args.get('show_agent_values', False):
                mismatches.append(f"show_agent_values: path has {edit_params['show_agent_values']}, got {editing_args.get('show_agent_values', False)}")

    # Validate generation hyperparameters
    # Path: .../{gen_dirname}/{white}/vs_{black}/improvements/...
    # gen_dirname is 3 levels above "improvements"
    if imp_idx >= 3:
        gen_dirname = parts[imp_idx - 3]
        try:
            gen_params = parse_generation_dirname(gen_dirname)
        except (ValueError, IndexError):
            gen_params = None

        if gen_params:
            if gen_params['model'] != generation_args['model']:
                mismatches.append(f"Generation model: path has '{gen_params['model']}', got '{generation_args['model']}'")
            if gen_params['temperature'] is not None:
                if abs(gen_params['temperature'] - generation_args.get('temperature', 1.0)) > 0.0001:
                    mismatches.append(f"Generation temperature: path has {gen_params['temperature']}, got {generation_args.get('temperature', 1.0)}")
            if gen_params.get('reasoning_effort') is not None:
                if gen_params['reasoning_effort'] != generation_args.get('reasoning_effort'):
                    mismatches.append(f"Generation reasoning_effort: path has '{gen_params['reasoning_effort']}', got '{generation_args.get('reasoning_effort')}'")
            if gen_params['tool_choice'] != generation_args.get('tool_choice', 'required'):
                mismatches.append(f"Generation tool_choice: path has '{gen_params['tool_choice']}', got '{generation_args.get('tool_choice', 'required')}'")
            if gen_params['prompt_key'] != generation_args.get('prompt_key'):
                mismatches.append(f"Generation prompt_key: path has '{gen_params['prompt_key']}', got '{generation_args.get('prompt_key')}'")

    if mismatches:
        error_msg = "Hyperparameter mismatch when resuming from improvement config:\n" + "\n".join(f"  - {m}" for m in mismatches)
        raise ValueError(error_msg)
