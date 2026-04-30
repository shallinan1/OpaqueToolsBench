# OpaqueToolsBench

A benchmark and pipeline for studying whether LLM agents can **recover the meaning of opacified tools** — tools whose names, descriptions, and parameters have been deliberately obscured — and whether **iterative description improvement** driven by evaluation feedback can recover lost performance.

This release covers two domains:

- **BFCL** (function calling) — the two paper categories `executable_simple` and `executable_multiple_function`.
- **BrowseCompPlus** (information retrieval) — 9 domain-specialized search tools (Wikipedia, academic, news, etc.) with opaque/transparent variants over BM25 and FAISS retrieval backends.

The chess domain will follow.

## What's shipped

- Opacity setup with independent name / description / parameter knobs (BFCL) and tool-shape opacity (BrowseCompPlus).
- Iterative description-improvement pipeline (**ToolObserver** in the paper): `v0 (opaque) → evaluate → rewrite descriptions → v1 → …`. Outputs land under `runs/{domain}/tool_observer/`.
- **BFCL:** ready-to-use opacified configs + pre-populated `function_call_cache.json` (654 entries) so paper scores reproduce exactly even though some BFCL tests hit live REST APIs.
- **BrowseCompPlus:** 12 ready-to-use shared-tool configs (transparent/opaque × BM25/FAISS × {no-doc, no-doc_search-all, no-doc_search-all-only}) + pre-built `id_to_url.json` and `base_url_counts.json`.

## Install

Python 3.10 with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

## LLM API keys

Create a `.env` file at the repo root (auto-loaded by `python-dotenv` at startup):

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...    # optional, only used with --together (BFCL)
HF_TOKEN=hf_...         # required for BrowseCompPlus (gated corpus on HuggingFace)
```

---

## BFCL setup

Vendor data is not committed. Clone Gorilla **pinned to `v1.3`** — our code targets the BFCL v1 layout. Upstream HEAD has restructured under `bfcl_eval/...` and won't work.

```bash
mkdir -p src/vendor
git clone --depth 1 --branch v1.3 \
  https://github.com/ShishirPatil/gorilla src/vendor/gorilla_bfcl_v1
```

### Function-execution credentials (required)

BFCL v1's executable test categories make live REST/RapidAPI calls during evaluation. The four keys below all must be present in **`src/vendor/gorilla_bfcl_v1/berkeley-function-call-leaderboard/function_credential_config.json`** (a separate file from `.env`), but only one typically needs payment.

| Key | Used by | Sign up | Cost |
|---|---|---|---|
| `RAPID-API-KEY` | Yahoo Finance, Urban Dictionary, COVID-19, Amazon, time-zone | [rapidapi.com](https://rapidapi.com/) | Free tier covers light/replication use |
| `EXCHANGERATE-API-KEY` | `convert_currency` | [exchangerate-api.com](https://www.exchangerate-api.com/) | Free |
| `OMDB-API-KEY` | movie rating/director | [omdbapi.com](https://www.omdbapi.com/apikey.aspx) | Free |
| `GEOCODE-API-KEY` | `get_coordinates_from_city` | [geocode.maps.co](https://geocode.maps.co/) | Free |

The shipped `function_call_cache.json` covers every call the paper made (keyed on `md5(exact_call_string)`), so cache-hit replication runs don't touch live APIs. Cache misses (different model, temperature, prompt, stochasticity) go to live APIs — free tiers handle that comfortably.

### BFCL quickstart

```bash
python -m src.datasets.bfcl.iterative_improve \
  --config-source 'src/datasets/bfcl/tool_configs/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:remove_all]_config.json' \
  --generation-model gpt-4o-2024-08-06 \
  --editing-model gpt-4o-2024-08-06 \
  --iterations 3
```

Outputs land under `runs/bfcl/tool_observer/…/v{N}/`. See [`src/datasets/bfcl/README.md`](src/datasets/bfcl/README.md) for the full workflow.

---

## BrowseCompPlus setup

```bash
# 1. Clone the upstream BrowseComp-Plus repo, pinned to a known-good commit.
#    (Upstream has no tags; this SHA matches the layout our code expects.)
mkdir -p src/vendor
git clone https://github.com/texttron/BrowseComp-Plus src/vendor/BrowseComp-Plus
(cd src/vendor/BrowseComp-Plus && git checkout 56534c8453a9efe37862f0173cf221974a99a49c)

# 2. Authenticate with Hugging Face (the corpus is gated)
huggingface-cli login   # or: export HF_TOKEN=hf_...

# 3. Download indexes + build the URL mapping
bash src/datasets/BrowseCompPlus/scripts/setup_database.sh
```

### Java 21 (required for BM25)

Pyserini needs a JDK 21 runtime:

```bash
conda install -c conda-forge openjdk=21
# or: sudo apt install -y openjdk-21-jdk
```

### BrowseCompPlus quickstart

```bash
python -m src.datasets.BrowseCompPlus.iterative_improve \
  --config-source src/datasets/BrowseCompPlus/shared_tools/fully_opaque_bm25_no-doc.json \
  --generation-model gpt-5 \
  --generation-reasoning-effort medium \
  --editing-model gpt-5 \
  --editing-reasoning-effort medium \
  --editing-prompt-type detailed_v2 \
  --synthesis-prompt-key v2 \
  --num-trajectories-batch 10 \
  --iterations 3
```

Outputs land under `runs/BrowseCompPlus/tool_observer/…`. See [`src/datasets/BrowseCompPlus/README.md`](src/datasets/BrowseCompPlus/README.md) for the full workflow and config matrix.

---

## Paper

[arXiv:2602.15197](https://arxiv.org/abs/2602.15197v1)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
