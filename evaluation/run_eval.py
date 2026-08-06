"""Score retrieval against the golden set. Requires live Atlas + Voyage.

    python -m evaluation.run_eval                 # score every question
    python -m evaluation.run_eval --verbose       # plus a per-question line
    python -m evaluation.run_eval --json out.json # machine-readable, for diffing runs

Run it before and after any retrieval change (threshold, reranker, filter, hybrid)
and compare the three headline numbers. This is the only thing standing between a
retrieval "improvement" and a silent regression.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import key_param
import rag
from evaluation.metrics import QuestionResult, score_question, summarize

GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    """Read the golden questions, ignoring the leading _comment key."""
    return json.loads(path.read_text())["questions"]


MISSING_PAGE = -1


def retrieved_pages(docs) -> list[int]:
    """Origin page of each retrieved chunk, in rank order - that is the ground truth key.

    A chunk with no page becomes MISSING_PAGE rather than being dropped. Dropping it
    would shift every later chunk up a rank and silently inflate MRR - and a
    measurement tool that lies is worse than no measurement at all. -1 can never
    match a real expected page, so it scores as the miss it is.
    """
    return [doc.metadata.get("page", MISSING_PAGE) for doc in docs]


def run(
    vector_store, reranker, golden: list[dict[str, Any]], verbose: bool = False
) -> list[QuestionResult]:
    results = []
    for entry in golden:
        # rag.retrieve retries its own network calls, so a 429 cannot void the run
        docs = rag.retrieve(vector_store, entry["question"], reranker)
        result = score_question(entry["question"], entry["expected_pages"], retrieved_pages(docs))
        results.append(result)
        if verbose:
            print(_format_line(result))
    return results


def _format_line(result: QuestionResult) -> str:
    if result.is_control:
        mark = "ok  " if result.abstained else "LEAK"
        detail = "abstained" if result.abstained else f"returned {list(result.retrieved_pages)}"
    else:
        mark = "hit " if result.hit else "MISS"
        detail = f"rr={result.reciprocal_rank:.2f} got={list(result.retrieved_pages)} want={list(result.expected_pages)}"
    return f"  [{mark}] {result.question[:58]:58} {detail}"


def format_summary(summary: dict[str, float | int]) -> str:
    return (
        f"\n  questions       {summary['questions']} "
        f"({summary['answerable']} answerable, {summary['controls']} controls)\n"
        f"  hit_rate        {summary['hit_rate']:.3f}   (answer found in top-k)\n"
        f"  mrr             {summary['mrr']:.3f}   (how near rank 1 it landed)\n"
        f"  abstention_rate {summary['abstention_rate']:.3f}   (controls correctly refused)\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score retrieval against the golden set.")
    parser.add_argument("--verbose", action="store_true", help="print a line per question")
    parser.add_argument("--json", type=Path, help="write the full run to this path")
    args = parser.parse_args()

    golden = load_golden()
    print(f"Scoring {len(golden)} golden questions against {rag.DB_NAME}.{rag.COLLECTION_NAME}...")

    client: MongoClient[dict[str, Any]]
    with MongoClient(key_param.MONGODB_URI) as client:
        results = run(
            rag.build_vector_store(client), rag.make_reranker(), golden, verbose=args.verbose
        )

    summary = summarize(results)
    print(format_summary(summary))

    if args.json:
        args.json.write_text(
            json.dumps(
                {"summary": summary, "results": [vars(r) for r in results]}, indent=2, default=list
            )
        )
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
