# OpaqueToolsBench

A benchmark and pipeline for studying whether LLM agents can **recover the meaning of opacified tools** — tools whose names, descriptions, and parameters have been deliberately obscured — and whether **iterative description improvement** driven by evaluation feedback can recover lost performance.

This release covers the **BFCL (function calling)** domain on the two test categories (`executable_simple`, `executable_multiple_function`) used in the paper. Additional domains (BrowseCompPlus, chess) will follow.

## What's shipped

- Opacity setup (`generate_configs.py`) with independent name / description / parameter knobs.
- Iterative description-improvement pipeline (**ToolObserver** in the paper): `v0 (opaque) → evaluate → rewrite descriptions → v1 → …`. Outputs land under `runs/bfcl/tool_observer/`. Description-rewrite prompt strategies: `basic_improved` (default), `reflective`, `evidence_based`.
- Ready-to-use configs for both paper categories — transparent base + opacified variants (full matrix in [`src/datasets/bfcl/README.md`](src/datasets/bfcl/README.md)).
- Pre-populated `function_call_cache.json` so paper scores reproduce exactly (some BFCL tests hit live REST APIs whose results drift).

## Install

Python 3.10 with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

Vendor data (BFCL test data + evaluation logic) is not committed. Clone it once:

```bash
mkdir -p src/vendor
git clone https://github.com/ShishirPatil/gorilla src/vendor/gorilla_bfcl_v1
```

Create a `.env` file at the repo root (auto-loaded by `python-dotenv` at startup):

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...   # optional, only used with --together
```

## Quickstart

Run the full iterative loop on a fully-opacified config:

```bash
python -m src.datasets.bfcl.iterative_improve \
  --config-source 'src/datasets/bfcl/tool_configs/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]_config.json' \
  --generation-model gpt-4o-2024-08-06 \
  --editing-model gpt-4o-2024-08-06 \
  --iterations 3
```

Outputs land under `runs/bfcl/tool_observer/…/v{N}/` (gitignored). Exit code 2 means all tests converged.

See [`src/datasets/bfcl/README.md`](src/datasets/bfcl/README.md) for the full workflow, config matrix, and how to regenerate configs.

## Paper

[arXiv:2602.15197](https://arxiv.org/abs/2602.15197v1)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
