# Chess Shared Tools

Tool configurations the agent sees during play. Each JSON bundles tool names + descriptions + parameters; the actual move generation lives in `chess_tools.py`.

## Shipped configs (paper canonical)

| File | Tool family | Tools | Opacity |
|---|---|---|---|
| `all_specialists_obfuscated.json` | Phase specialists | `function_1`–`function_4` (opening / middlegame / endgame / late-endgame, generic descs) | Opaque |
| `all_specialists_accurate.json` | Phase specialists | `opening_specialist`, `middlegame_specialist`, `endgame_specialist`, `late_endgame_specialist` (real descs) | Transparent (Gold) |
| `elo_tools_obfuscated.json` | Skill / Elo | `function_1`–`function_3` (1200 / 1800 / 2400 Elo, generic descs) | Opaque |
| `elo_tools_accurate.json` | Skill / Elo | `beginner`, `intermediate`, `advanced` (real descs) | Transparent (Gold) |

These map directly to the two tool settings reported in the paper's Table 3.

## Opacity definition

In every `*_obfuscated.json`, the agent sees:
- `function_N` style names
- An identical, non-informative description on every tool
- The tool parameters (a board position, depth, etc.) are accurate

In every `*_accurate.json`, the agent sees:
- Descriptive names that match the underlying tool
- Real descriptions explaining what each tool does

A `function_mapping` field in the obfuscated configs records which `function_N` corresponds to which real tool (for evaluation, not for the agent).

## Usage

```bash
# Run with the obfuscated config (the interesting case)
python -m src.datasets.chess.run \
  --shared-tools src/datasets/chess/shared_tools/elo_tools_obfuscated.json \
  --model gpt-5 \
  --black-type elo_1800

# Or with the transparent Gold config for the ceiling reference
python -m src.datasets.chess.run \
  --shared-tools src/datasets/chess/shared_tools/elo_tools_accurate.json \
  --model gpt-5 \
  --black-type elo_1800
```

See `../README.md` for the full ToolObserver loop.
