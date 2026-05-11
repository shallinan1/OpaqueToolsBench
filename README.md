# OpaqueToolsBench

A benchmark and pipeline for studying whether LLM agents can **recover the meaning of opacified tools** — tools whose names, descriptions, and parameters have been deliberately obscured — and whether **iterative description improvement** driven by evaluation feedback can recover lost performance.

This release covers three domains:

- **BFCL** (function calling) — the two paper categories `executable_simple` and `executable_multiple_function`.
- **BrowseCompPlus** (information retrieval) — 9 domain-specialized search tools (Wikipedia, academic, news, etc.) with opaque/transparent variants over BM25 and FAISS retrieval backends.
- **Chess** (strategic reasoning) — 4 phase-specialist or 3 Elo-rated stateless move-suggestion tools; the agent plays a Fairy-Stockfish opponent at a target Elo and discovers tool semantics through play.

Chess code is now in the repo; paper-canonical trajectories will ship as a separate Release asset.

## What's shipped

- Opacity setup with independent name / description / parameter knobs (BFCL) and tool-shape opacity (BrowseCompPlus).
- Iterative description-improvement pipeline (**ToolObserver** in the paper): `v0 (opaque) → evaluate → rewrite descriptions → v1 → …`. Outputs land under `runs/{domain}/tool_observer/`.
- **BFCL:** ready-to-use opacified configs + pre-populated `function_call_cache.json` (654 entries) so paper scores reproduce exactly even though some BFCL tests hit live REST APIs.
- **BrowseCompPlus:** 12 ready-to-use shared-tool configs (transparent/opaque × BM25/FAISS × {no-doc, no-doc_search-all, no-doc_search-all-only}) + pre-built `id_to_url.json` and `base_url_counts.json`.
- **Chess:** 4 paper-canonical shared-tool configs (opaque + Gold variants for both phase-specialist and Elo skill settings) + pre-sampled `train.jsonl` / `test.jsonl` positions (2000 total, ~400 KB).

## Install

Python 3.10 with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

> **Apple Silicon note:** if `uv sync` fails on torch wheels with a hint about `macosx_*_x86_64`, your `uv` / Python is Intel-emulated under Rosetta 2 (common with `~/opt/anaconda3` installations). Install ARM-native uv via Homebrew (`arch -arm64 brew install uv`) and let it resolve a fresh ARM Python.

## LLM API keys

Create a `.env` file at the repo root (auto-loaded by `python-dotenv` at startup):

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...    # optional, only used with --together (BFCL)
HF_TOKEN=hf_...         # required for BrowseCompPlus (gated corpus on HuggingFace)
```

---

## BFCL setup

Vendor data is not committed. Clone Gorilla and check out a **specific SHA** — our code targets the BFCL v1 layout (`berkeley-function-call-leaderboard/eval_checker/...`), and that layout was removed in the upstream restructure. The `v1.3` tag is *not* the right pin (it points to a commit predating the v1 eval layout).

```bash
mkdir -p src/vendor
git clone https://github.com/ShishirPatil/gorilla src/vendor/gorilla_bfcl_v1
(cd src/vendor/gorilla_bfcl_v1 && git checkout 83dfe1a97329a167a79bbe2fa67bc57d55369d1f)
```

### Function-execution credentials (required)

BFCL v1's executable test categories make live REST/RapidAPI calls during evaluation. Upstream ships an empty template at **`src/vendor/gorilla_bfcl_v1/berkeley-function-call-leaderboard/function_credential_config.json`** — open it and fill in the four keys. (This file is *separate* from `.env`; it's read by upstream BFCL code, not our code.)

After cloning the vendor at the SHA above, the file already exists with empty values:

```json
[{"RAPID-API-KEY" : ""},{"EXCHANGERATE-API-KEY" : ""},{"OMDB-API-KEY" : ""}, {"GEOCODE-API-KEY": ""}]
```

Fill each empty string with your key. Any missing key → `NoAPIKeyError` at eval time. Only one typically needs payment:

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
  --generation-model gpt-5-mini \
  --editing-model gpt-5-mini \
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

## Chess setup

Chess requires the **Fairy-Stockfish** binary for live game-play (GPLv3, not redistributed). Fairy-Stockfish, not vanilla Stockfish: only the Fairy fork exposes the `UCI_Elo` knob required for the Elo-rated tools.

```bash
# macOS (Homebrew)
brew install fairy-stockfish
# binary lands at /opt/homebrew/bin/fairy-stockfish (ARM)
# or /usr/local/bin/fairy-stockfish (Intel)

# Linux / Windows: download a release binary, chmod +x
# https://github.com/fairy-stockfish/Fairy-Stockfish/releases
```

Then point at it from your `.env`:

```
FAIRY_STOCKFISH_PATH=/abs/path/to/fairy-stockfish
```

The chess corpus (2000 pre-sampled positions, 200 train + 1800 test) is already shipped under `src/datasets/chess/data/`. No Lichess DB download required for paper reproduction.

### Chess quickstart

```bash
python -m src.datasets.chess.iterative_improve \
  --config-source src/datasets/chess/shared_tools/elo_tools_obfuscated.json \
  --generation-model gpt-5 \
  --editing-model gpt-5 \
  --editing-prompt-key detailed \
  --iterations 10 \
  --black-type elo_1800
```

Outputs land under `runs/chess/tool_observer/…`. See [`src/datasets/chess/README.md`](src/datasets/chess/README.md) for the full workflow, the two paper tool settings (phase specialists / Elo skills), and headline-metric scoring (`evaluate_tool_selection.py`, `compute_elo.py`).

---

## Paper

[arXiv:2602.15197](https://arxiv.org/abs/2602.15197v1)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
