# BFCL (Berkeley Function Call Leaderboard)

Pipeline for running BFCL function-calling benchmarks under controlled tool opacity, with an iterative description-improvement loop.

```
Generation → Evaluation → Description rewrite → (repeat)
```

Evaluation executes model calls against BFCL ground truth. The rewrite step conditions on failed calls and execution errors to propose better descriptions, which feed the next iteration.

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
  --model gpt-4o-2024-08-06 \
  --output-dir runs/bfcl/ours
```

### Step 3 — Evaluate

```bash
python -m src.datasets.bfcl.evaluate \
  --result-dir runs/bfcl/ours/<config_name>/<hyperparam_dir>
```

Executed function-call results are cached in `function_call_cache.json` (keyed by
an md5 of the call string). We ship a pre-populated cache (654 entries) so paper
scores reproduce exactly — some BFCL tests hit live REST APIs (weather, stocks)
whose values drift, so rebuilding from scratch will not match. Delete the file
to force re-execution.

### Step 4 — Iterative improvement

Full loop:

```bash
python -m src.datasets.bfcl.iterative_improve \
  --config-source 'src/datasets/bfcl/tool_configs/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]_config.json' \
  --generation-model gpt-4o-2024-08-06 \
  --editing-model gpt-4o-2024-08-06 \
  --iterations 3
```

Single rewrite step:

```bash
python -m src.datasets.bfcl.generate_descriptions \
  --result-dir runs/bfcl/ours/<config_name>/<hyperparam_dir> \
  --model gpt-4o-2024-08-06 \
  --prompt-key reflective
```

Description-rewrite prompt strategies: `reflective` (default), `evidence_based`, `basic_improved`.

## Outputs

Under `runs/bfcl/ours/<config>/<hypers>/`:
- `v0_results.json`, `v0_scored.json` — base run
- `improvements/<edit_hypers>/v1/{config,results,scored}.json`, `v2/…` — iterations

## Dependencies

Requires vendor BFCL data at `src/vendor/gorilla_bfcl_v1/` (see top-level README for setup).

## Paper

[arXiv:2602.15197](https://arxiv.org/abs/2602.15197v1)
