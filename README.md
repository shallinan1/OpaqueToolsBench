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

Vendor data (BFCL test data + evaluation logic) is not committed. Clone it once, **pinned to `v1.3`** — our code targets the BFCL v1 layout (`berkeley-function-call-leaderboard/eval_checker/...`). Upstream HEAD has restructured under `bfcl_eval/...` and won't work.

```bash
mkdir -p src/vendor
git clone --depth 1 --branch v1.3 \
  https://github.com/ShishirPatil/gorilla src/vendor/gorilla_bfcl_v1
```

If you already have a working BFCL v1 checkout elsewhere, you can `ln -s /path/to/your/gorilla src/vendor/gorilla_bfcl_v1` instead.

### LLM API keys

Create a `.env` file at the repo root (auto-loaded by `python-dotenv` at startup):

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...   # optional, only used with --together
```

### Function-execution credentials (BFCL upstream) — required

BFCL v1's executable test categories make live REST/RapidAPI calls during evaluation (Yahoo Finance, Urban Dictionary, COVID-19, ExchangeRate-API, OMDB, Geocode). These keys go in **`src/vendor/gorilla_bfcl_v1/berkeley-function-call-leaderboard/function_credential_config.json`** — a separate file from `.env`, read by upstream BFCL code. Required keys (all have free tiers):

- `RAPID-API-KEY`
- `EXCHANGERATE-API-KEY`
- `OMDB-API-KEY`
- `GEOCODE-API-KEY`

**Why you need these even with the cache:** the shipped `function_call_cache.json` is keyed on `md5(exact_call_string)`. It hits only when your run produces byte-identical calls to ours. Any divergence — different model, different temperature, different prompt, even normal model stochasticity — produces a cache miss → BFCL hits the live API → `NoAPIKeyError` if these keys aren't set. The cache speeds up reruns; it doesn't replace the credentials.

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
