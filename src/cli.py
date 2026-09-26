"""WeaveMem command line: build typed memory graphs and retrieve from them."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Modules under src/ import each other by bare name, so src/ must be on sys.path when
# this file is run as a script rather than imported as part of a package.
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config
import retrieval
from enriched_embeddings import build_enriched_mid_embeddings
from ingest import ingest_corpus

logger = logging.getLogger("weave-mem")


def _build(args: argparse.Namespace) -> int:
    total = ingest_corpus(
        args.input,
        sample=args.sample,
        session=args.session,
        restart=args.restart,
    )
    logger.info("data_dir=%s", config.DATA_DIR)
    logger.info("TOTAL: %s", dict(total))
    return 0


def _embed(args: argparse.Namespace) -> int:
    stats = build_enriched_mid_embeddings(
        conversation_id=args.conversation,
        restart=args.restart,
        prune_orphans=args.conversation is None,
    )
    logger.info("enriched mid embeddings: %s", stats)
    return 0


def _search(args: argparse.Namespace) -> int:
    result = retrieval.search_memory_result(
        args.question,
        conversation_id=args.conversation,
        user_id=args.user,
        top_k=args.top_k,
    )
    if args.trace:
        print(json.dumps(result.trace, ensure_ascii=False, indent=2))
        return 0
    if not result.memories:
        print("no memories retrieved")
        return 0
    for i, memory in enumerate(result.memories, 1):
        print(f"[{i}] {memory.get('type') or 'mid'}  id={memory.get('id')}")
        print(f"    {retrieval.mid_text(memory).strip()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(prog="weave-mem", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build memories from a conversation corpus")
    build.add_argument("--input", required=True, help="path to the corpus JSON file")
    build.add_argument(
        "--sample",
        default=None,
        help="sample_id to process, e.g. conv-26; default: every conversation",
    )
    build.add_argument(
        "--session",
        type=int,
        default=None,
        help="session number within the sample, e.g. 1 (requires --sample)",
    )
    build.add_argument(
        "--restart",
        action="store_true",
        help="rebuild sessions already recorded as done (ignore saved progress)",
    )
    build.set_defaults(func=_build)

    embed = sub.add_parser(
        "embed", help="rebuild the enriched mid-embedding sidecar graph retrieval needs"
    )
    embed.add_argument(
        "--conversation", default=None, help="limit to one conversation_id"
    )
    embed.add_argument(
        "--restart",
        action="store_true",
        help="re-embed every mid instead of reusing unchanged vectors",
    )
    embed.set_defaults(func=_embed)

    search = sub.add_parser("search", help="retrieve memories for a question")
    search.add_argument("--question", required=True, help="the question to retrieve for")
    search.add_argument(
        "--conversation", default=None, help="restrict retrieval to one conversation_id"
    )
    search.add_argument("--user", default=None, help="restrict retrieval to one user_id")
    search.add_argument(
        "--top-k", type=int, default=None, help="cap on the memories returned"
    )
    search.add_argument(
        "--trace",
        action="store_true",
        help="print the full retrieval trace as JSON instead of the memories",
    )
    search.set_defaults(func=_search)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
