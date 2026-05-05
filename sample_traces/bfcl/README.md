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

The repo's `evaluate.py` can re-grade any run directory in this tree:

```bash
python -m src.datasets.bfcl.evaluate \
  --result-dir sample_traces/bfcl/tool_observer/executable_simple_base/gpt5_medium_req_8192_must_call_tool_seed0
```

Cached function-call results live in `src/datasets/bfcl/function_call_cache.json` and cover every executable lookup the paper made, so re-grading does not require RapidAPI / OMDB / etc. credentials.
