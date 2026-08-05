"""Retrieval metrics - pure functions, no I/O, so they can be unit tested.

Ground truth is expressed as the set of source PDF pages that actually answer a
question. A retrieved chunk carries its origin page in `metadata["page"]`, so a
retrieval is "relevant at rank i" when that chunk came from an expected page.

An expected-page set that is empty means the question is an off-topic control:
the corpus does not answer it, and the correct behaviour is to retrieve nothing.
Those cases are scored separately (`abstention`) rather than folded into
hit-rate, because averaging a refusal into a recall number hides both.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionResult:
    """Scored outcome for one golden question."""

    question: str
    expected_pages: tuple[int, ...]
    retrieved_pages: tuple[int, ...]
    hit: bool
    reciprocal_rank: float
    is_control: bool

    @property
    def abstained(self) -> bool:
        """A control question passes only when retrieval returned nothing at all."""
        return not self.retrieved_pages


def first_relevant_rank(
    retrieved_pages: Sequence[int], expected_pages: Sequence[int]
) -> int | None:
    """1-indexed rank of the first retrieved chunk from an expected page, else None."""
    expected = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return rank
    return None


def reciprocal_rank(retrieved_pages: Sequence[int], expected_pages: Sequence[int]) -> float:
    """1/rank of the first relevant chunk - 0.0 when none of the top-k is relevant.

    Rewards putting the answer first, not merely somewhere in the window, which is
    what matters once only the top few chunks reach the prompt.
    """
    rank = first_relevant_rank(retrieved_pages, expected_pages)
    return 0.0 if rank is None else 1.0 / rank


def score_question(
    question: str, expected_pages: Sequence[int], retrieved_pages: Sequence[int]
) -> QuestionResult:
    """Score one question. Controls (no expected pages) are flagged, not hit-scored."""
    return QuestionResult(
        question=question,
        expected_pages=tuple(expected_pages),
        retrieved_pages=tuple(retrieved_pages),
        hit=first_relevant_rank(retrieved_pages, expected_pages) is not None,
        reciprocal_rank=reciprocal_rank(retrieved_pages, expected_pages),
        is_control=not expected_pages,
    )


def summarize(results: Sequence[QuestionResult]) -> dict[str, float | int]:
    """Aggregate scored questions into the numbers a retrieval change is judged on.

    hit_rate and mrr cover answerable questions only; abstention_rate covers the
    controls. A change that lifts hit_rate while dropping abstention_rate has
    bought recall with noise - that trade is the whole reason both are reported.
    """
    answerable = [r for r in results if not r.is_control]
    controls = [r for r in results if r.is_control]

    return {
        "questions": len(results),
        "answerable": len(answerable),
        "controls": len(controls),
        "hit_rate": _mean(1.0 if r.hit else 0.0 for r in answerable),
        "mrr": _mean(r.reciprocal_rank for r in answerable),
        "abstention_rate": _mean(1.0 if r.abstained else 0.0 for r in controls),
    }


def _mean(values) -> float:
    """Mean of an iterable, 0.0 when empty - an empty slice scores 0, never crashes."""
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
