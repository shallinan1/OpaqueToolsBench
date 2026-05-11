# Chess

Strategic-reasoning benchmark using stateless move-suggestion tools. The agent plays a chess game against a fixed Stockfish opponent at a target Elo; tools either return moves from a phase-specialist or from a target-Elo engine. Opacity hides which tool maps to which strength or phase: the agent must discover the mapping through gameplay.

```
Trajectory collection → Stockfish scoring → Description rewrite (batch + synthesis) → (repeat)
```

The rewrite step is two-phase: per-batch trajectory analysis (`generate_descriptions.py`) followed by cross-batch synthesis (`synthesize_descriptions.py`) into a final config.

## Setup

### Fairy-Stockfish (required)

Live game-play requires the [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) binary (GPLv3, not shipped here). Use this rather than vanilla Stockfish: it exposes the `UCI_Elo` knob for strength-limited play (calibrated 500–2850 at 120s+1s, CCRL 40/4).

Download a release binary, make it executable, and point `FAIRY_STOCKFISH_PATH` at it in your `.env`:

```
FAIRY_STOCKFISH_PATH=/abs/path/to/fairy-stockfish-binary
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

Trajectories from the paper's chess runs will ship in a follow-up release; the iterative loop above produces them when run live. Headline metrics are produced by:

- `evaluate_tool_selection.py` — per-position "best tool chosen?" percentage, aggregated to the "Acc" column.
- `compute_elo.py` — streaming Elo from game outcomes, aggregated to the "ELO" column.
- Both read the `v*_scored.json` outputs from Step 2.

## Scoring logic

Cell values in Table 3 come from these functions (verify these names with the actual code; they're the entry-point aggregators):

- `evaluate.py::main()` — Stockfish position evaluation per move.
- `evaluate_tool_selection.py` — "best tool chosen?" metric per position.
- `compute_elo.py` — Elo rating computation from W/L/D records.

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
