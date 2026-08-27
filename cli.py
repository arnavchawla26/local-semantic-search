#!/usr/bin/env python3
"""CLI for the local-first semantic search tool.

    python cli.py index ./my_notes --out .semsearch_index.pkl
    python cli.py search .semsearch_index.pkl "what did I write about backpropagation"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semsearch.index import SemanticIndex


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.directory).expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    idx = SemanticIndex(n_components=args.components, chunk_size=args.chunk_size, overlap=args.overlap)
    n_chunks = idx.build(root)
    idx.save(Path(args.out))
    print(f"Indexed {n_chunks} chunk(s) from {root} -> {args.out}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Index file not found: {index_path} (run 'index' first)", file=sys.stderr)
        return 1

    idx = SemanticIndex.load(index_path)
    results = idx.search(args.query, top_k=args.top)

    if not results:
        print("No results.")
        return 0

    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [{r.score:.3f}] {r.doc_path} (chunk {r.chunk_index})")
        print(f"   {r.snippet}\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build a search index from a directory of notes/PDFs")
    p_index.add_argument("directory", help="Directory to index (.txt, .md, .pdf files)")
    p_index.add_argument("--out", default=".semsearch_index.pkl", help="Where to save the index")
    p_index.add_argument("--components", type=int, default=128, help="Latent dimensions (SVD components)")
    p_index.add_argument("--chunk-size", type=int, default=800, help="Approx characters per chunk")
    p_index.add_argument("--overlap", type=int, default=150, help="Approx character overlap between chunks")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search a previously built index")
    p_search.add_argument("index", help="Path to the index file")
    p_search.add_argument("query", help="Natural-language search query")
    p_search.add_argument("--top", type=int, default=5, help="Number of results to show")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
