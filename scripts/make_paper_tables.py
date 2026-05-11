#!/usr/bin/env python3
"""
Reproduce the paper's Tables 2, 3, and 4 directly from the shipped trajectories
and Release tarball, without re-running any LLM calls. Markdown by default; pass
--csv for a flat CSV.

Examples:

    # Table 2 (BFCL) — reads from sample_traces/bfcl/ committed in the repo
    python -m scripts.make_paper_tables --table 2

    # Table 4 (BCP) — needs the trajectory tarball extracted somewhere
    python -m scripts.make_paper_tables --table 4 \\
        --bcp-dir /path/to/extracted/BrowseCompPlus

    # CSV
    python -m scripts.make_paper_tables --table 2 --csv

    # All tables at once
    python -m scripts.make_paper_tables --table all --bcp-dir /path/to/BrowseCompPlus

Output reflects what the shipped trajectories actually scored. For most
cells this matches the paper print to two decimals; a few cells differ
slightly because of how the public bundle was filtered or which
iteration's snapshot the paper printed. Specifically:

  Table 2 (BFCL): `+ P2P` for the two `*_param[remove_all]` opacity rows
  aggregates over only `executable_multiple_function` (n=50). The source
  play2prompt runs for `executable_simple` at those two opacity settings
  used only open-source models (gpt-oss, Kimi, Qwen), which were
  filtered out of the public bundle per the OpenAI-only release policy.
  The third opacity row aggregates over both categories.
"""

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----- Table 2 (BFCL) -----------------------------------------------------------

BFCL_ROWS = [
    # (display_label, config_suffix)
    ("Anon. Fn. Names Only",   "name[all:increasing_number]_desc[all:blank]_param[all:remove_all]"),
    ("Anon. Fn. + Real Desc",  "name[all:increasing_number]_param[all:remove_all]"),
    ("Anon. Fn. + Param Names","name[all:increasing_number]_desc[all:blank]_param[all:blank_descriptions]"),
]

BFCL_MODELS = [
    # (display, gen_hypers_dir)
    ("GPT-5",      "gpt5_medium_req_8192_must_call_tool_seed0"),
    ("GPT-5-mini", "gpt5mini_medium_req_8192_must_call_tool_seed0"),
]

BFCL_CATEGORIES = ["executable_simple", "executable_multiple_function"]


def _load_summary(scored_path: Path) -> Optional[Dict]:
    if not scored_path.exists():
        return None
    with scored_path.open() as f:
        return json.load(f).get("summary", {})


def _bfcl_cell_for_category(bfcl_dir: Path, method: str, config_name: str,
                            gen_hypers: str, edit_hypers: Optional[str]) -> Optional[Dict]:
    """Return summary dict for a single (method, config, model, [edit]) leaf, or None."""
    cfg_dir = bfcl_dir / method / config_name / gen_hypers
    if edit_hypers is None:
        # Base / one-shot baseline: use v0_scored.json
        return _load_summary(cfg_dir / "v0_scored.json")
    # ToolObserver iteration: walk improvements/<edit_hypers>/v{N}/ and take the
    # highest N with a scored.json present (converged depth).
    imp_dir = cfg_dir / "improvements" / edit_hypers
    if not imp_dir.exists():
        return None
    versions = []
    for v in imp_dir.iterdir():
        m = re.match(r"^v(\d+)$", v.name)
        if m and (v / "scored.json").exists():
            versions.append((int(m.group(1)), v / "scored.json"))
    if not versions:
        return None
    versions.sort()
    return _load_summary(versions[-1][1])


def _aggregate_bfcl(per_category: List[Optional[Dict]]) -> Optional[Tuple[float, float, float]]:
    """Combine per-category summary dicts into (E, P, A) totals weighted by test count."""
    by_cat = [s for s in per_category if s]
    if not by_cat:
        return None
    total = sum(s.get("total", 0) for s in by_cat) or 1
    correct = sum(s.get("correct", 0) for s in by_cat)
    e = correct / total
    p = sum(s.get("parameter_accuracy_avg", 0.0) * s.get("total", 0) for s in by_cat) / total
    a = sum(s.get("ast_format_score_avg", 0.0) * s.get("total", 0) for s in by_cat) / total
    return (e, p, a)


def collect_table_2(bfcl_dir: Path) -> List[Dict]:
    """One row per (paper_row, model, method); each row has E, P, A."""
    EDIT_HYPERS = "gpt5_medium_basic_improved_8192"
    out: List[Dict] = []

    for row_label, suffix in BFCL_ROWS:
        for model_label, gen_h in BFCL_MODELS:
            row = {"opacity": row_label, "model": model_label}
            for method_label, method_dir, edit_h, use_base_cfg in [
                ("Gold",  "tool_observer", None, True),   # transparent base config
                ("Base",  "tool_observer", None, False),  # opacified, v0
                ("+ TO",  "tool_observer", EDIT_HYPERS, False),
                ("+ P2P", "play2prompt",   None, False),
                ("+ ET",  "easytool",      None, False),
            ]:
                e_acc, p_acc, a_acc = None, None, None
                per_cat: List[Optional[Dict]] = []
                for cat in BFCL_CATEGORIES:
                    cfg_name = f"{cat}_base" if use_base_cfg else f"{cat}_{suffix}"
                    per_cat.append(
                        _bfcl_cell_for_category(bfcl_dir, method_dir, cfg_name, gen_h, edit_h)
                    )
                agg = _aggregate_bfcl(per_cat)
                if agg is not None:
                    e_acc, p_acc, a_acc = agg
                row[f"{method_label}_E"] = e_acc
                row[f"{method_label}_P"] = p_acc
                row[f"{method_label}_A"] = a_acc
            out.append(row)
    return out


def render_table_2_markdown(rows: List[Dict]) -> str:
    lines = []
    lines.append("## Table 2 — BFCL\n")
    header = ["Documentation", "Model"]
    methods = ["Gold", "Base", "+ TO", "+ P2P", "+ ET"]
    for m in methods:
        header += [f"{m} E", f"{m} P", f"{m} A"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        cells = [r["opacity"], r["model"]]
        for m in methods:
            for x in ("E", "P", "A"):
                v = r.get(f"{m}_{x}")
                cells.append(f"{v:.2f}" if v is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----- Table 3 (Chess) ----------------------------------------------------------

def render_table_3_stub() -> str:
    return (
        "## Table 3 — Chess\n\n"
        "Chess trajectories are not yet shipped in this artifact bundle. The chess\n"
        "pipeline (run.py, evaluate.py, evaluate_tool_selection.py, compute_elo.py) is\n"
        "in the repo at `src/datasets/chess/`; trajectories will follow as a separate\n"
        "GitHub Release. Re-run this script with `--table 3 --chess-dir <path>` once\n"
        "they're available.\n"
    )


# ----- Table 4 (BrowseCompPlus) -------------------------------------------------

BCP_ROWS = [
    # (display_label, config_base)
    ("Domain-specific (9) Search",            "no-doc"),
    ("Domain-specific (9) + Full Search",     "no-doc_search-all"),
]

BCP_MODELS = [
    ("GPT-5",      "gpt5_minimal_auto_mx20000_pkbesttoolfirst_k5_s512_gd0_url1_iter50_emb0p6b"),
    ("GPT-5-mini", "gpt5mini_minimal_auto_mx20000_pkbesttoolfirst_k5_s512_gd0_url1_iter50_emb0p6b"),
]


def _bcp_cell(bcp_dir: Path, subtree: str, config_dir: str, gen_h: str,
              leaf_suffix: str = "") -> Tuple[Optional[float], Optional[float]]:
    """Return (accuracy, avg_calls_per_query) or (None, None) if missing."""
    p = bcp_dir / subtree / "shared_tools" / config_dir / gen_h
    if leaf_suffix:
        p = p / leaf_suffix
    scored = p / "v0_scored.json"
    if not scored.exists():
        return (None, None)
    summary = json.load(scored.open()).get("summary", {})
    return (summary.get("accuracy"), summary.get("tool_usage", {}).get("avg_calls_per_query"))


def collect_table_4(bcp_dir: Path) -> List[Dict]:
    out: List[Dict] = []
    for row_label, base in BCP_ROWS:
        for model_label, gen_h in BCP_MODELS:
            row = {"setting": row_label, "model": model_label}
            cells = [
                ("Gold",  "gold_baseline",            f"transparent_faiss_{base}", ""),
                ("Base",  "gold_baseline",            f"fully_opaque_faiss_{base}", ""),
                ("+ TO",  "tool_observer_test_iter4", f"fully_opaque_faiss_{base}", "from_v4"),
                ("+ P2P", "play2prompt",              f"fully_opaque_faiss_{base}", ""),
                ("+ ET",  "easytool",                 f"fully_opaque_faiss_{base}", ""),
            ]
            for method_label, subtree, config_dir, leaf in cells:
                acc, tc = _bcp_cell(bcp_dir, subtree, config_dir, gen_h, leaf)
                row[f"{method_label}_Acc"] = acc
                row[f"{method_label}_#TC"] = tc
            out.append(row)
    return out


def render_table_4_markdown(rows: List[Dict]) -> str:
    lines = []
    lines.append("## Table 4 — BrowseCompPlus\n")
    methods = ["Gold", "Base", "+ TO", "+ P2P", "+ ET"]
    header = ["Setting", "Model"]
    for m in methods:
        header += [f"{m} Acc", f"{m} #TC"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        cells = [r["setting"], r["model"]]
        for m in methods:
            acc = r.get(f"{m}_Acc")
            tc = r.get(f"{m}_#TC")
            cells.append(f"{acc * 100:.1f}" if acc is not None else "—")
            cells.append(f"{tc:.1f}" if tc is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ----- CSV helpers --------------------------------------------------------------

def to_csv(rows: List[Dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r[k]) for k in keys})
    return buf.getvalue()


# ----- CLI ----------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", choices=["2", "3", "4", "all"], required=True)
    p.add_argument("--bfcl-dir", type=Path, default=Path("sample_traces/bfcl"),
                   help="Path to sample_traces/bfcl (default: repo-relative).")
    p.add_argument("--bcp-dir",  type=Path, default=None,
                   help="Path to extracted BrowseCompPlus trajectory dir (from the Release tarball).")
    p.add_argument("--chess-dir", type=Path, default=None,
                   help="Path to chess trajectories (not yet released; reserved).")
    p.add_argument("--csv", action="store_true", help="Output CSV instead of markdown.")
    args = p.parse_args()

    out: List[str] = []

    def emit(table_name: str, rows: List[Dict], md_renderer):
        if args.csv:
            out.append(f"# {table_name}\n")
            out.append(to_csv(rows))
        else:
            out.append(md_renderer(rows))

    if args.table in ("2", "all"):
        if not args.bfcl_dir.exists():
            print(f"BFCL trace dir not found: {args.bfcl_dir}", file=sys.stderr)
            return 2
        rows = collect_table_2(args.bfcl_dir)
        emit("Table 2 — BFCL", rows, render_table_2_markdown)

    if args.table in ("3", "all"):
        out.append(render_table_3_stub())

    if args.table in ("4", "all"):
        if args.bcp_dir is None:
            out.append(
                "## Table 4 — BrowseCompPlus\n\n"
                "Pass `--bcp-dir <path-to-extracted-tarball-BrowseCompPlus-root>` to populate.\n"
                "Tarball download: https://github.com/shallinan1/OpaqueToolsBench/releases/tag/v0.1-bcp-trajectories\n"
            )
        elif not args.bcp_dir.exists():
            print(f"BCP trace dir not found: {args.bcp_dir}", file=sys.stderr)
            return 2
        else:
            rows = collect_table_4(args.bcp_dir)
            emit("Table 4 — BrowseCompPlus", rows, render_table_4_markdown)

    print("\n".join(out), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
