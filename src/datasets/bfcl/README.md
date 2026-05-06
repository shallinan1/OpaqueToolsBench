# BFCL (Berkeley Function Call Leaderboard)

Pipeline for running BFCL function-calling benchmarks under controlled tool opacity, with an iterative description-improvement loop.

```
Generation → Evaluation → Description rewrite → (repeat)
```

Evaluation executes model calls against BFCL ground truth. The rewrite step conditions on failed calls and execution errors to propose better descriptions, which feed the next iteration.

> **Reproducing paper numbers without spending API credits:** the trajectories from the paper's BFCL runs are committed at [`sample_traces/bfcl/`](../../../sample_traces/bfcl/). You can re-grade any run with `evaluate.py` against the shipped function-call cache; no live API calls needed. See `sample_traces/bfcl/README.md` for the layout.

## Reproducing Table 2 (BFCL paper results)

Each cell of Table 2 corresponds to one shipped trajectory directory under `sample_traces/bfcl/`. Map paper rows to opacity configs:

| Paper row | Config dir suffix |
|---|---|
| Anon. Fn. Names Only | `_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]` |
| Anon. Fn. + Real Desc | `_name[all:increasing_number]_param[all:remove_all]` |
| Anon. Fn. + Param Names | `_name[all:increasing_number]_desc[all:blank]_param[all:blank_descriptions]` |
| Gold (transparent) | `_base` (under `tool_observer/` only) |

Map paper columns to method subtrees:

| Paper column | Subtree | Point `--result-dir` at |
|---|---|---|
| Gold | `sample_traces/bfcl/tool_observer/<config>_base/<gen_hypers>/` | base run dir (has `v0_results.json`) |
| Base | `sample_traces/bfcl/tool_observer/<opaque_config>/<gen_hypers>/` | base run dir |
| `+ TO` (ToolObserver) | `sample_traces/bfcl/tool_observer/<opaque_config>/<gen_hypers>/improvements/gpt5_medium_basic_improved_8192/v{N}/` | the **converged leaf** v{N} (per-test convergence; max iteration varies by config) |
| `+ P2P` (Play2Prompt) | `sample_traces/bfcl/play2prompt/<opaque_config>/<gen_hypers>/` | base run dir (one-shot baseline) |
| `+ ET` (EasyTool) | `sample_traces/bfcl/easytool/<opaque_config>/<gen_hypers>/` | base run dir (one-shot baseline) |

`<gen_hypers>` is `gpt5_medium_req_8192_must_call_tool_seed0` (GPT-5 row) or `gpt5mini_medium_req_8192_must_call_tool_seed0` (GPT-5-mini row).

Concrete example — Table 2, *Anon. Fn. Names Only*, GPT-5-mini, ToolObserver column:

```bash
python -m src.datasets.bfcl.evaluate \
  --result-dir 'sample_traces/bfcl/tool_observer/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]/gpt5mini_medium_req_8192_must_call_tool_seed0/improvements/gpt5_medium_basic_improved_8192/v6'
```

The shipped `scored.json` in each leaf already contains the cell value; re-running `evaluate.py` reproduces it (uses the function-call cache, no API needed).

ToolObserver convergence depth varies per config — find the highest `v{N}` directory under each `improvements/gpt5_medium_basic_improved_8192/` to get the converged result. Per-config maxima range from v5 to v11 in the shipped bundle.

## Scoring logic

The cell values in Table 2 (E / P / A columns: Execution, Parameter, AST accuracy) are produced by:

- `evaluate.py::evaluate_result()` and `evaluate_category_results()` — execution-based grading using BFCL's vendor `eval_runner`. Produces the **E** (Execution) column.
- `enhanced_metrics.py::evaluate_enhanced_metrics()` and `aggregate_enhanced_metrics()` — produces the **P** (Parameter accuracy) and **A** (AST accuracy) columns.
- `evaluate.py::main()` writes the per-leaf `scored.json` aggregating all three.

## Test categories

| Category | Description |
|---|---|
| `executable_simple` | Single function call per question |
| `executable_multiple_function` | One correct function among several choices |

## Shipped configs

`tool_configs/` bundles questions + tool definitions per opacity setting. For each category:

| Config suffix | Names | Descriptions | Parameters |
|---|---|---|---|
| `_base_config.json` | accurate | accurate | accurate |
| `_name[all:increasing_number]_param[all:remove_all]_config.json` | opaque (`function_N`) | accurate | removed |
| `_name[all:increasing_number]_desc[all:blank]_config.json` *(multi only)* | opaque | blank | accurate |
| `_name[all:increasing_number]_desc[all:blank]_param[all:blank_descriptions]_config.json` | opaque | blank | descriptions blanked |
| `_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]_config.json` | opaque | blank | removed (full opacity) |

To generate other opacity settings, use `generate_configs.py` (see Step 1).

## Workflow

### Step 1 — Generate configs (optional)

Skip if you're using a shipped config.

```bash
python -m src.datasets.bfcl.generate_configs \
  --test-category executable_simple \
  --output-dir src/datasets/bfcl/tool_configs
```

### Step 2 — Run generation

```bash
python -m src.datasets.bfcl.run \
  --config-source 'src/datasets/bfcl/tool_configs/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]_config.json' \
  --model gpt-5-mini \
  --output-dir runs/bfcl/tool_observer
```

### Step 3 — Evaluate

```bash
python -m src.datasets.bfcl.evaluate \
  --result-dir runs/bfcl/tool_observer/<config_name>/<hyperparam_dir>
```

Executed function-call results are cached in `function_call_cache.json` (keyed by
an md5 of the call string). We ship a pre-populated cache (654 entries) so paper
scores reproduce exactly — some BFCL tests hit live REST APIs (weather, stocks)
whose values drift, so rebuilding from scratch will not match. Delete the file
to force re-execution.

### Step 4 — Iterative improvement (ToolObserver)

Full loop:

```bash
python -m src.datasets.bfcl.iterative_improve \
  --config-source 'src/datasets/bfcl/tool_configs/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]_config.json' \
  --generation-model gpt-5-mini \
  --editing-model gpt-5-mini \
  --iterations 3
```

Single rewrite step:

```bash
python -m src.datasets.bfcl.generate_descriptions \
  --result-dir runs/bfcl/tool_observer/<config_name>/<hyperparam_dir> \
  --model gpt-5-mini \
  --prompt-key basic_improved
```

Description-rewrite prompt strategies: `basic_improved` (default), `reflective`, `evidence_based`.

## Outputs

Under `runs/bfcl/tool_observer/<config>/<hypers>/`:
- `v0_results.json`, `v0_scored.json` — base run
- `improvements/<edit_hypers>/v1/{config,results,scored}.json`, `v2/…` — iterations

## Dependencies

Requires vendor BFCL data at `src/vendor/gorilla_bfcl_v1/` (see top-level README for setup).

## Paper

[arXiv:2602.15197](https://arxiv.org/abs/2602.15197v1)
