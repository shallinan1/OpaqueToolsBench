"""BrowseCompPlus-specific argument extensions for the shared dataset runner."""

import argparse
from pathlib import Path
from src.datasets.BrowseCompPlus.prompts import BROWSECOMP_AGENT_PROMPTS

def add_browsecompplus_args(parser: argparse.ArgumentParser, vendor_path: Path) -> argparse.ArgumentParser:
    """Add BrowseCompPlus-specific args to a parser."""
    parser.set_defaults(model="gpt-5", temperature=1.0, top_p=1.0, output_dir="runs/BrowseCompPlus/tool_observer")

    # Search configuration (legacy single-searcher mode)
    parser.add_argument("--searcher-type", choices=["bm25", "faiss"], default=None, help="Search backend for single-searcher mode.")
    parser.add_argument("--filter-category", default=None, help="Category filter for single-searcher mode.")
    parser.add_argument("--filter-domain", default=None, help="Domain filter for single-searcher mode.")

    # Tool behavior
    parser.add_argument("--k", type=int, default=5, help="Top-k results per search tool call.")
    parser.add_argument("--snippet-max-tokens", type=int, default=512, help="Max tokens per result snippet.")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max completion tokens per model response (default: 10000; matches BrowseComp-Plus).")
    parser.add_argument("--include-get-document", action="store_true", help="Include get_document tool.")
    parser.add_argument("--hide-urls", action="store_true", help="Hide URLs from search results.")
    parser.add_argument("--max-iterations", type=int, default=50, help="Max conversation rounds.")
    parser.add_argument("--prompt-key", type=str, default=None, choices=sorted(BROWSECOMP_AGENT_PROMPTS.keys()), help="Prompt template key from src/datasets/BrowseCompPlus/prompts.py.")

    parser.add_argument("--reasoning-effort", type=str, default="medium", choices=["minimal", "low", "medium", "high"], help="Reasoning effort for reasoning models (gpt-5/o-series).")
    parser.add_argument("--split", type=str, choices=["train", "test"], default=None, help="Load queries from train or test split. If omitted, uses all queries.")

    # File paths
    parser.add_argument("--categories-file", type=Path, default=Path(__file__).parent / "category_mappings" / "simple_categories.json", help="Path to category mapping JSON.")
    parser.add_argument("--url-mapping-file", type=Path, default=Path(__file__).parent / "outputs" / "id_to_url.json", help="Path to docid->URL mapping JSON.")
    parser.add_argument("--index-path", type=Path, default=vendor_path / "indexes" / "bm25", help="BM25 index path.")
    parser.add_argument("--faiss-index-path", type=str, default=str(vendor_path / "indexes" / "qwen3-embedding-0.6b" / "*.pkl"), help="FAISS index glob path.")
    parser.add_argument(
        "--faiss-max-batch-queries",
        type=int,
        default=32,
        help="Max queries per FAISS batch-search chunk to control GPU memory (clamped to 1..256).",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug printing.")
    return parser


def create_browsecompplus_parser(vendor_path: Path) -> argparse.ArgumentParser:
    """Create BrowseCompPlus parser from shared parser + dataset args."""
    from src.datasets.run_args import parser as shared_parser

    parser = argparse.ArgumentParser(
        parents=[shared_parser],
        description="Run BrowseCompPlus evaluation",
    )
    return add_browsecompplus_args(parser, vendor_path)
