# BrowseCompPlus

Information-retrieval benchmark with 9 domain-specialized search tools (Wikipedia, academic, news, etc.) plus an optional `get_document` tool. Opacity hides which tool searches which domain — the agent must discover that through usage.

```
Generation → Evaluation (LLM-as-judge) → Description rewrite (batch + synthesis) → (repeat)
```

The rewrite step is two-phase: per-batch trajectory analysis (default 10 trajectories per minibatch) followed by cross-batch synthesis into final tool descriptions.

> **Reproducing paper numbers without spending API credits or running retrieval:** trajectories from the paper's BrowseCompPlus runs are available as a [GitHub Release asset](https://github.com/shallinan1/OpaqueToolsBench/releases/tag/v0.1-bcp-trajectories) (~720 MB compressed). Includes the train trajectory at v0–v3, the v1 and v4 test-split evaluations, the transparent-config ceiling, and the EasyTool / Play2Prompt baselines.
>
> **Verifying without API/dataset access:** every leaf's `v0_scored.json` already contains both the aggregate (`summary.accuracy`, `summary.tool_usage`, etc.) and per-query judge transcripts (`detailed_evaluations[].judge_result.reasoning`). Reading these is free. The headline numbers in the paper come straight from `summary.accuracy`.
>
> **Independently re-grading:** running `evaluate.py` on a result_dir reads the shipped `v0_results.json` and re-invokes the LLM judge. Requires an OpenAI key (~cents per query) **and** the BrowseComp-Plus held-out answers file `browsecomp_plus_decrypted.jsonl`, which is part of the gated BrowseComp-Plus dataset on HuggingFace (cloned via `setup_database.sh`).

## Reproducing Table 4 (BrowseCompPlus paper results)

After downloading and extracting the trajectories tarball, paper rows map as follows:

| Paper row (Tool Setting) | Config base name |
|---|---|
| Domain-specific (9) Search | `_no-doc` |
| Domain-specific (9) + Full Search | `_no-doc_search-all` |

| Paper column | Subtree | Point `--result-dir` at |
|---|---|---|
| Gold | `gold_baseline/shared_tools/transparent_faiss_<base>/<gen_hypers>/` | base run dir |
| Base | `gold_baseline/shared_tools/fully_opaque_faiss_<base>/<gen_hypers>/` | base run dir |
| `+ TO` | `tool_observer_test_iter4/shared_tools/fully_opaque_faiss_<base>/<gen_hypers>/from_v4/` | `from_v4` dir |
| `+ P2P` | `play2prompt/shared_tools/fully_opaque_faiss_<base>/<gen_hypers>/` | base run dir |
| `+ ET` | `easytool/shared_tools/fully_opaque_faiss_<base>/<gen_hypers>/` | base run dir |

`<gen_hypers>` is `gpt5_minimal_auto_mx20000_pkbesttoolfirst_k5_s512_gd0_url1_iter50_emb0p6b` (GPT-5) or `gpt5mini_minimal_auto_mx20000_pkbesttoolfirst_k5_s512_gd0_url1_iter50_emb0p6b` (GPT-5-mini).

Pre-computed scored numbers live in each leaf's `v0_scored.json` (the Acc and #TC columns). To re-grade, point `evaluate.py` at the leaf `--result-dir`.

## Scoring logic

Table 4 cell values come from:

- `evaluate.py::main()` — orchestrates LLM-as-judge grading per query.
- `evaluation_utils.py::create_judge_prompt()` and `parse_judge_response()` — judge prompt construction and answer extraction.
- `evaluation_utils.py::compute_citation_metrics()` — derives the per-query correctness flag.

Aggregation across queries (mean accuracy, mean tool-call count) happens at the end of `main()` and is written to the `scored.json` summary in the result dir.

## Setup

Vendor data is **not committed**. You need both the upstream BrowseComp-Plus repo (for the `search_agent` library and pre-built indexes) and a Hugging Face token (the corpus dataset is gated).

```bash
# 1. Clone the upstream BrowseComp-Plus repo into src/vendor/, pinned to a
#    known-good commit (upstream has no tags; this SHA matches our layout).
mkdir -p src/vendor
git clone https://github.com/texttron/BrowseComp-Plus src/vendor/BrowseComp-Plus
(cd src/vendor/BrowseComp-Plus && git checkout 56534c8453a9efe37862f0173cf221974a99a49c)

# 2. Authenticate with Hugging Face (the corpus is a gated dataset)
huggingface-cli login
# or: export HF_TOKEN=hf_...

# 3. Download indexes + build the URL mapping
bash src/datasets/BrowseCompPlus/scripts/setup_database.sh
```

The setup script downloads BM25 + FAISS indexes via the upstream's
`scripts_build_index/download_indexes.sh`, then runs `extract_base_urls.py` to
populate `outputs/id_to_url.json` and `outputs/base_url_counts.json`. (Both are
also shipped pre-built; the setup step refreshes them against your local
corpus.)

### Java 21 (required for BM25)

Pyserini needs a JDK 21 runtime:

```bash
conda install -c conda-forge openjdk=21
# or: sudo apt install -y openjdk-21-jdk
```

## Shipped configs

`shared_tools/` ships 12 configs. Each bundles an OpenAI tool definition + a `searcher_config` (which backend, which category filter):

|  | `_no-doc` | `_no-doc_search-all` | `_no-doc_search-all-only` |
|---|---|---|---|
| `transparent_bm25_*` | accurate names + descs | + a "search all domains" tool | only the search-all tool |
| `fully_opaque_bm25_*` | `tool_1`/`tool_2`/... + generic descs | + opaque search-all | only opaque search-all |
| `transparent_faiss_*` | (FAISS variants) | | |
| `fully_opaque_faiss_*` | | | |

Paper evaluations use `*_no-doc.json` (the simple variant, no `get_document` tool, no search-all). The `_search-all` and `_search-all-only` variants are for ablations.

## Workflow

### Step 1 — Run

```bash
python -m src.datasets.BrowseCompPlus.run \
  --config-source src/datasets/BrowseCompPlus/shared_tools/fully_opaque_bm25_no-doc.json \
  --model gpt-5 \
  --num-queries 100 \
  --output-dir runs/BrowseCompPlus/tool_observer
```

For FAISS configs, the path includes an embedding tag (e.g. `emb0p6b`, `emb4b`, `emb8b`) derived from `--faiss-index-path`.

### Step 2 — Evaluate (LLM-as-judge)

```bash
python -m src.datasets.BrowseCompPlus.evaluate \
  --result-dir runs/BrowseCompPlus/tool_observer/shared_tools/fully_opaque_bm25_no-doc/<gen_hypers> \
  --judge-model gpt-5
```

### Step 3 — Generate improved descriptions (single iteration)

```bash
python -m src.datasets.BrowseCompPlus.generate_improved_descriptions \
  --result-dir runs/BrowseCompPlus/tool_observer/shared_tools/fully_opaque_bm25_no-doc/<gen_hypers> \
  --prompt-type detailed_v2 \
  --synthesis-prompt-key v2 \
  --num-trajectories-batch 10 \
  --model gpt-5
```

Writes:
- `.../improvements/{editing_hypers}/v1/config.json` — improved config
- `.../improvements/{editing_hypers}/v1/llm_responses.json` — per-batch analyses
- `.../improvements/{editing_hypers}/v1/synthesis_response.json` — cross-batch synthesis

### Step 4 — Iterative improve (ToolObserver)

Full loop:

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

Description-rewrite prompt strategies: `detailed_v2` (paper default), `detailed`. Synthesis strategies: `v2` (paper default), `v1`.

### Iteration semantics

`--iterations N`:
- From a base config: run `v0` plus `N` improvement rounds.
- From `.../improvements/.../vK/config.json`: run `N` additional rounds starting at `vK+1`.

## Outputs

Base runs:
```
runs/BrowseCompPlus/tool_observer/shared_tools/<config>/<gen_hypers>/
├── v0_results.json
├── v0_metadata.json
├── v0_scored.json
└── v0_token_usage_generation.json
```

Improvements:
```
…/<gen_hypers>/improvements/<editing_hypers>/
├── v1/{config,results,scored,metadata,llm_responses,synthesis_response,token_usage_generation}.json
└── v2/…
```

## Legacy single-searcher mode

For debugging without a config file:

```bash
python -m src.datasets.BrowseCompPlus.run \
  --searcher-type bm25 \
  --filter-category wikipedia \
  --model gpt-5 \
  --num-queries 50 \
  --output-dir runs/BrowseCompPlus/tool_observer
```

## Paper

[arXiv:2602.15197](https://arxiv.org/abs/2602.15197v1)
