# Setup & Reproduction Guide

End-to-end setup for OpaqueToolsBench, with exact versions, a tiered install,
expected resource usage, and instructions for both API-based and local-LLM
execution. For per-domain details see each `src/datasets/<domain>/README.md`.

---

## 1. Python & dependency tiers

**Python 3.10** (pinned: `requires-python = "==3.10.*"`). Use [`uv`](https://docs.astral.sh/uv/).

Dependencies are split into a lightweight **core** plus optional groups, so the
core installs cleanly on macOS (Intel + Apple Silicon), Linux, and Windows
without pulling a GPU/serving stack:

```bash
uv sync                        # CORE: inspect, re-grade, reproduce tables, API agent runs
uv sync --extra retrieval      # + BrowseCompPlus FAISS/BM25 retrieval
uv sync --extra local-llm      # + local model serving (GPU Linux)
uv sync --extra metrics        # + description-similarity metrics (Table 6; GPU)
uv sync --all-extras           # everything (GPU Linux recommended)
source .venv/bin/activate
```

| Tier | What it covers | Platforms | Pinned highlights |
|---|---|---|---|
| **core** (default) | offline inspection, re-grading shipped trajectories, `make_paper_tables.py`, API-based BFCL/Chess agent runs | macOS / Linux / Windows | `tree-sitter==0.21.3` (0.22 broke the vendor API), `mistralai==0.4.2` (vendor uses `mistralai.client.MistralClient`, gone in 1.0+), all four provider SDKs |
| **retrieval** | BrowseCompPlus FAISS + BM25 retrieval | Linux preferred | `faiss-cpu`, `pyserini` (needs **JDK 21**), `tevatron` |
| **local-llm** | run agents against a locally-served model | GPU Linux (torch/vLLM have no x86_64-macOS wheels) | `vllm`, `torch`, `transformers` |
| **metrics** | Table 6 semantic/NLI similarity | GPU | `sentence-transformers`, `torch` |

> **Apple Silicon note:** if a `--extra local-llm` / `--all-extras` sync fails on
> torch wheels mentioning `macosx_*_x86_64`, your `uv`/Python is Intel-emulated
> under Rosetta 2 (common with `~/opt/anaconda3`). Install ARM-native uv
> (`arch -arm64 brew install uv`) and let it resolve a fresh ARM Python. The
> **core** tier is unaffected and installs under either.

### JDK 21 (only for `--extra retrieval`)
Pyserini's BM25 needs a JDK 21 runtime:
```bash
conda install -c conda-forge openjdk=21      # or: sudo apt install -y openjdk-21-jdk
```

---

## 2. Credentials

API-based execution reads keys from a repo-root `.env` (auto-loaded by `python-dotenv`):

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...     # optional, only if running Claude models
TOGETHER_API_KEY=...      # optional, BFCL --together
HF_TOKEN=hf_...           # required for BrowseCompPlus (gated corpus)
FAIRY_STOCKFISH_PATH=/abs/path/to/fairy-stockfish   # required for Chess gameplay
```

BFCL's executable test categories additionally read four REST keys from the
**vendor** file `src/vendor/gorilla_bfcl_v1/.../function_credential_config.json`
(see [`src/datasets/bfcl/README.md`](src/datasets/bfcl/README.md)). The shipped
`function_call_cache.json` (654 entries) covers every call the paper made, so
cache-hit re-grading needs **no** REST keys.

---

## 3. Quick verification (no LLM credits, ~1 minute)

Confirm the install works end-to-end against shipped data:

```bash
# Reproduce the paper tables from committed sample traces (stdlib only)
python scripts/make_paper_tables.py --table all

# Re-grade a shipped BFCL baseline trajectory (uses function_call_cache, no API keys)
python -m src.datasets.bfcl.evaluate \
  --result-dir "sample_traces/bfcl/easytool/executable_simple_name[all:increasing_number]_desc[all:blank]_param[all:blank_descriptions]/gpt5_medium_req_8192_must_call_tool_seed0"
# Expected: 80.00% (80/100), parameter 82.33%, AST 96.40%
```

---

## 4. Expected resource usage

| Task | Compute | Wall-clock | $ (API) |
|---|---|---|---|
| Re-grade shipped trajectories | CPU, core tier | seconds–minutes | $0 (cache) |
| `make_paper_tables.py` | CPU, core tier | seconds | $0 |
| BFCL `iterative_improve` (1 config, 3 iters, gpt-5-mini) | CPU + API | ~10–30 min | a few $ |
| Chess test-set run (1 config, 1 opponent, 100 positions × 3 traj, gpt-5) | CPU + API + Fairy-Stockfish | **~4–5 h** | **~$100/experiment** |
| BrowseCompPlus index build + corpus | `--extra retrieval`, ~3 GB disk | ~30–60 min | $0 (download) |
| BrowseCompPlus agent run | retrieval + API | ~1–3 h | tens of $ |
| Local-LLM serving | `--extra local-llm`, **GPU** | model-dependent | $0 (self-hosted) |

Chess Table 3 in full (3 methods × 2 tool settings × 3 opponents × 2 models) is
the most expensive line item — budget accordingly before launching a full re-run.

---

## 5. API-based vs local-LLM execution

- **API-based (default):** set provider keys in `.env`, pass `--model gpt-5`
  (or `gpt-5-mini`, `claude-*`, etc.). This is the core tier and what all paper
  numbers were produced with.
- **Local-LLM:** `uv sync --extra local-llm` on a GPU Linux box, serve the model
  with vLLM, and point the run at the local endpoint. See
  [`src/datasets/BrowseCompPlus/README.md`](src/datasets/BrowseCompPlus/README.md)
  for the vLLM/Qwen serving path used for the open-model rows.

---

## 6. Canonical experiment settings

The exact hyperparameters and commands behind each paper table live in a single
central file: [`configs/paper_experiments.yaml`](configs/paper_experiments.yaml).
Each entry lists the model, reasoning effort, iteration count, opponent/seed, and
the precise `python -m ...` invocation, so a run can be reproduced by copying one
block rather than reconstructing flags.
