"""Find where to put the rerank thresholds, by measuring the score gap.

    python -m evaluation.calibrate

Prints the top reranker score for every golden question, split into answerable and
control. A usable threshold sits in the gap between the lowest answerable score and
the highest control score. If those two overlap, no threshold separates them and the
fix is better retrieval or a better golden set - not a number nudged until it looks
right.

This exists because RERANK_THRESHOLD and RELAXED_RERANK_THRESHOLD are the two numbers
most likely to be tuned by vibes, and vibes are what put a `hasCode` filter in the
retrieval path for months.
"""

import sys
from pathlib import Path
from typing import Any

from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import key_param
import rag
from evaluation.run_eval import load_golden


def top_score(vector_store, query: str, reranker) -> float:
    """Best reranker score among the candidates - 0.0 when nothing was retrieved.

    Goes through rag.score_candidates so calibration measures the same scoring path
    retrieval uses. A calibration that scored differently from production would
    produce thresholds that are precisely wrong.

    score_candidates brings its own backoff; the Atlas query is spelled out here
    rather than reused from rag.retrieve because calibration wants top_k=1, so it
    borrows rag's retry policy to get the same protection.
    """
    candidates = rag.retryer()(rag.build_candidate_retriever(vector_store).invoke, query)
    scored = rag.score_candidates(query, candidates, reranker, top_k=1)
    return max((score for _, score in scored), default=0.0)


def report(answerable: list[tuple[str, float]], controls: list[tuple[str, float]]) -> str:
    lines = ["\nAnswerable (want these ABOVE the threshold):"]
    lines += [f"  {score:.4f}  {q[:60]}" for q, score in sorted(answerable, key=lambda x: x[1])]
    lines += ["\nControls (want these BELOW the threshold):"]
    lines += [f"  {score:.4f}  {q[:60]}" for q, score in sorted(controls, key=lambda x: -x[1])]

    lowest_answerable = min((s for _, s in answerable), default=0.0)
    highest_control = max((s for _, s in controls), default=0.0)
    lines.append(f"\n  lowest answerable  {lowest_answerable:.4f}")
    lines.append(f"  highest control    {highest_control:.4f}")

    if highest_control < lowest_answerable:
        midpoint = (highest_control + lowest_answerable) / 2
        lines.append(f"  -> clean gap; a threshold near {midpoint:.2f} separates them")
    else:
        lines.append(
            "  -> NO GAP: some control outscores some answerable question. "
            "No threshold separates them; fix retrieval or the golden set instead."
        )
    return "\n".join(lines)


def main() -> None:
    golden = load_golden()
    answerable: list[tuple[str, float]] = []
    controls: list[tuple[str, float]] = []

    client: MongoClient[dict[str, Any]]
    with MongoClient(key_param.MONGODB_URI) as client:
        vector_store = rag.build_vector_store(client)
        reranker = rag.make_reranker()
        for i, entry in enumerate(golden, 1):
            score = top_score(vector_store, entry["question"], reranker)
            bucket = answerable if entry["expected_pages"] else controls
            bucket.append((entry["question"], score))
            print(f"  {i:2}/{len(golden)} {score:.4f}  {entry['question'][:55]}", flush=True)

    print(report(answerable, controls))


if __name__ == "__main__":
    main()
