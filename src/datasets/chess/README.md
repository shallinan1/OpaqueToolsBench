# Chess

Strategic-reasoning benchmark using stateless move-suggestion tools. The agent plays a chess game against a fixed Stockfish opponent at a target Elo; tools either return moves from a phase-specialist or from a target-Elo engine. Opacity hides which tool maps to which strength or phase: the agent must discover the mapping through gameplay.

```
Trajectory collection → Stockfish scoring → Description rewrite (batch + synthesis) → (repeat)
```

The rewrite step is two-phase: per-batch trajectory analysis (`generate_descriptions.py`) followed by cross-batch synthesis (`synthesize_descriptions.py`) into a final config.

## Setup

### Fairy-Stockfish (required)

Live game-play requires the [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) binary (GPLv3, not shipped here). Use this rather than vanilla Stockfish: it exposes the `UCI_Elo` knob for strength-limited play (calibrated 500–2850 at 120s+1s, CCRL 40/4).

```bash
# macOS (Homebrew formula, easiest)
brew install fairy-stockfish

# Linux / Windows: download a release binary + chmod +x
# https://github.com/fairy-stockfish/Fairy-Stockfish/releases
```

Then point at it from your `.env`:

```
FAIRY_STOCKFISH_PATH=/abs/path/to/fairy-stockfish
```

Quick sanity check that the binary works (should print a UCI banner and exit on `quit`):

```bash
echo -e "uci\nquit" | "$FAIRY_STOCKFISH_PATH"
```

### Data (shipped, no download needed)

Pre-sampled chess positions are already in `data/`:
- `data/train.jsonl` — 200 positions used for description-iteration
- `data/test.jsonl` — 1800 positions used for headline numbers

These were produced by `process_data.py` from the full Lichess evaluation database. **You don't need to re-run `process_data.py`** unless you want to re-sample with different proportions; `process_data.py` is an internal data-prep utility and is kept here only for documentation.

## Test categories

The paper (Table 3) evaluates on two tool settings:

| Setting | Config | Tool family |
|---|---|---|
| Phase specialists | `shared_tools/all_specialists_obfuscated.json` | 4 tools: opening / middlegame / endgame / late-endgame specialists |
| Skill / Elo | `shared_tools/elo_tools_obfuscated.json` | 3 tools: beginner / intermediate / advanced (Elo 1200 / 1800 / 2400) |

The corresponding `_accurate.json` configs are the transparent ("Gold") versions with descriptive names and real descriptions.

## Workflow

### Step 1 — Run trajectory collection

```bash
python -m src.datasets.chess.run \
  --shared-tools src/datasets/chess/shared_tools/elo_tools_obfuscated.json \
  --model gpt-5 \
  --black-type elo_1800 \
  --max-moves 120 \
  --output-dir runs/chess/tool_observer
```

`--black-type` controls the opponent's strength. `elo_1800` is the paper's default.

### Step 2 — Score trajectories with Stockfish

```bash
python -m src.datasets.chess.evaluate \
  --result-dir runs/chess/tool_observer/shared_tools/elo_tools_obfuscated/<split>/<gen_hypers>/agent/vs_elo_1800
```

Adds Stockfish-eval annotations to each move in `v*_trajectories.json`, written alongside as `v*_scored.json`. Used as feedback for the description-rewrite step.

### Step 3 — Description rewrite (single iteration)

```bash
# Batch analysis (per-trajectory LLM critique)
python -m src.datasets.chess.generate_descriptions \
  --result-dir runs/chess/tool_observer/.../v0/ \
  --model gpt-5 \
  --prompt-key detailed

# Cross-batch synthesis (combine batches into a new config.json)
python -m src.datasets.chess.synthesize_descriptions \
  --result-dir runs/chess/tool_observer/.../v0/ \
  --model gpt-5 \
  --prompt-key v1
```

Writes:
- `improvements/{editing_hypers}/v1/llm_responses.json` — per-batch analyses
- `improvements/{editing_hypers}/v1/synthesis_*.json` — cross-batch synthesis intermediates
- `improvements/{editing_hypers}/v1/config.json` — the improved tool descriptions

### Step 4 — Iterative improve (ToolObserver)

Full loop:

```bash
python -m src.datasets.chess.iterative_improve \
  --config-source src/datasets/chess/shared_tools/elo_tools_obfuscated.json \
  --generation-model gpt-5 \
  --editing-model gpt-5 \
  --editing-prompt-key detailed \
  --iterations 10 \
  --black-type elo_1800
```

Description-rewrite prompt strategies: `detailed` (batch analysis, paper canonical), `v1` (synthesis, paper canonical).

### Final headline numbers

Aggregating the per-config / per-iteration results into Table 3 cells:

```bash
# % optimal tool selection (paper's "Acc" column)
python -m src.datasets.chess.evaluate_tool_selection \
  --result-dir runs/chess/tool_observer/.../v{N}/

# Streaming Elo from actual win/loss/draw outcomes (paper's "ELO" column)
python -m src.datasets.chess.compute_elo \
  --result-dir runs/chess/tool_observer/.../v{N}/
```

## Reproducing Table 3

Trajectories from the paper's chess runs ship as a GitHub Release asset: **[v0.2-chess-trajectories](https://github.com/shallinan1/OpaqueToolsBench/releases/tag/v0.2-chess-trajectories)** (~17 MB compressed, ~505 MB extracted).

Pre-computed summaries for the cells reproducible from the shipped data are committed at `sample_traces/chess/`, so `python scripts/make_paper_tables.py --table 3` produces a partial Table 3 directly from a fresh clone without downloading the tarball.

### Known scope gaps (please read before reproducing)

The shipped bundle is a **partial** reproduction of Table 3. When packaging this release we discovered that our preserved artifact storage did not contain the full set of chess runs we expected:

1. **Only one of three Stockfish opponents was preserved.** The paper's Table 3 pools games against `vs_elo_1200`, `vs_elo_1800`, and `vs_elo_2400`; the released bundle contains only `vs_elo_1800`. Single-opponent ELO estimates run **systematically higher than the paper number** (~+200–380 ELO across the four baseline cells we can reproduce) because the streaming-Elo estimator needs multiple anchors to converge. Method *ordering* is expected to hold.
2. **No `+TO` test-set gameplay was preserved.** The shipped bundle contains the description-iteration training data on `train_10q` (`ours/.../improvements/v1..v11/`) and the *converged ToolObserver descriptions* at `ours/.../v11/config.json`, but not a test-set evaluation of the agent running with those descriptions. To produce the `+TO` row of Table 3 a reviewer would need to run `src.datasets.chess.run` with `--shared-tools <v11/config.json> --split test --black-type elo_1800` (and ideally the other two opponents too).
3. **No GPT-5-mini chess data.** The paper has a GPT-5-mini row for both tool settings; the shipped bundle is GPT-5 only.
4. **The `optimized_config.json` files referenced by baseline trajectory metadata are not in the bundle.** Baseline EasyTool and Play2Prompt runs reference per-method config files at `src/datasets/chess/shared_tools/easytool/.../optimized_config.json` that were generated by the baseline pipelines. Those derived configs were not preserved; exact baseline regeneration would require re-running the baseline pipelines first.

### Cell-producing scripts

- `evaluate_tool_selection.py` — per-position "best tool chosen?" percentage, aggregated to the **Acc** column.
- `compute_elo.py` — streaming Elo from game outcomes, aggregated to the **ELO** column.
- Both read the `v*_scored.json` outputs from Step 2 (or the trajectory JSON files directly for tool selection).

## Scoring logic

Cell values in Table 3 come from these functions:

- `evaluate.py::score_trajectory_file()` — Stockfish position evaluation per move; writes `v*_scored.json` from `v*_trajectories.json`.
- `evaluate_tool_selection.py::evaluate_tool_selection()` (entry: `main()`) — "best tool chosen?" metric per position. Produces the **Acc** column.
- `compute_elo.py::compute_streaming_elo()` and `compute_bootstrap_elo()` (entry: `main()`) — streaming Elo rating with bootstrap confidence intervals. Produces the **ELO** column.

## Outputs

```
runs/chess/tool_observer/shared_tools/<config>/<split>/<gen_hypers>/agent/vs_<opponent>/
├── v0_trajectories.json
├── v0_metadata.json
├── v0_scored.json
└── improvements/<edit_hypers>/
    ├── v1/{config,llm_responses,synthesis_*}.json
    ├── v2/...
    └── ...
```

## Paper

[arXiv:2602.15197](https://arxiv.org/abs/2602.15197v1)
