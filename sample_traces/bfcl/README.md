# BFCL Sample Traces

Trajectories from the paper's BFCL runs. Provided so reviewers (and future readers) can verify the headline numbers without spending API credit. Every file here was produced by running `src.datasets.bfcl.run` and `src.datasets.bfcl.evaluate` on a fresh copy of the BFCL v1 vendor data; nothing is synthetic.

## Layout

```
sample_traces/bfcl/
├── tool_observer/   # ToolObserver (Ours) — iterative description improvement
├── easytool/        # EasyTool baseline
└── play2prompt/     # Play2Prompt baseline
```

Per method, the same opacity matrix:

```
{method}/
├── executable_simple_base                                                       # transparent reference
├── executable_simple_name[all:increasing_number]_param[all:remove_all]          # opaque names + removed params
├── executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:blank_descriptions]
├── executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]   # full opacity
└── (same four for executable_multiple_function)
```

Inside each `{config}/{generation_hypers}/`:

| File | Contents |
|---|---|
| `v0_results.json` | Raw model outputs + tool calls for the base run (JSONL) |
| `v0_scored.json` | Per-test grading against ground truth |
| `v0_metadata.json` | Run hyperparameters (model, temperature, prompt key, etc.) |
| `v0_token_usage_generation.json` | Token usage summary |
| `improvements/{editing_hypers}/v{N}/...` | ToolObserver iteration N: improved config + re-run results |

## Hyperparameters used (paper canonical)

- **Generation models:** `gpt-5`, `gpt-5-mini`
- **Generation prompt:** `must_call_tool` (forces tool use; matches paper)
- **Editing model:** `gpt-5` at `medium` reasoning effort
- **Editing prompt:** `basic_improved`
- **Tool choice:** `required`
- **Max tokens:** 8192
- **Seed:** 0

Each generation hyperparameter directory is named:

```
{model_short}_{reasoning_effort}_{tool_choice}_{max_tokens}_{prompt_key}_seed{seed}
```

For example: `gpt5_medium_req_8192_must_call_tool_seed0`.

## What's not here

Open-source model rows (`gpt-oss`, `Kimi`, `Qwen`) and dev-only prompt sweeps were excluded to keep the bundle focused on the paper's reported numbers.

## How to use

The repo's `evaluate.py` can re-grade any run directory in this tree. **The level you point `--result-dir` at depends on whether the run is a base run or an iteration:**

| Run type | Point `--result-dir` at | Files at that level |
|---|---|---|
| Base run (paper "Base" or "Gold" column) | `<config>/<gen_hypers>/` | `v0_results.json`, `v0_scored.json`, `v0_metadata.json` |
| ToolObserver iteration (paper "+TO" column) | `<config>/<gen_hypers>/improvements/<edit_hypers>/v{N}/` | `config.json`, `results.json`, `scored.json`, `metadata.json` |
| EasyTool / Play2Prompt baselines | `<config>/<gen_hypers>/` (one-shot, no iteration tree) | `v0_results.json`, `v0_scored.json`, `v0_metadata.json` |

Examples:

```bash
# Base run (Gold or Base column)
python -m src.datasets.bfcl.evaluate \
  --result-dir sample_traces/bfcl/tool_observer/executable_simple_base/gpt5_medium_req_8192_must_call_tool_seed0

# ToolObserver iteration leaf (the v{N}/ matters; converged N varies per config)
python -m src.datasets.bfcl.evaluate \
  --result-dir 'sample_traces/bfcl/tool_observer/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]/gpt5mini_medium_req_8192_must_call_tool_seed0/improvements/gpt5_medium_basic_improved_8192/v6'

# EasyTool / Play2Prompt baseline (same shape as base run)
python -m src.datasets.bfcl.evaluate \
  --result-dir 'sample_traces/bfcl/easytool/executable_simple_name[all:increasing_number]_param[all:remove_all]/gpt5_medium_req_8192_must_call_tool_seed0'
```

Common mistake: pointing `--result-dir` at `improvements/<edit_hypers>/` (the parent of the v{N} dirs). That level has no results files and the script will fail — go one deeper into the specific `v{N}/`.

Cached function-call results live in `src/datasets/bfcl/function_call_cache.json` and cover every executable lookup the paper made, so re-grading does not require RapidAPI / OMDB / etc. credentials.
