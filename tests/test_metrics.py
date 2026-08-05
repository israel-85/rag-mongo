import json
from pathlib import Path

from evaluation.metrics import (
    first_relevant_rank,
    reciprocal_rank,
    score_question,
    summarize,
)

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "golden.json"


class TestFirstRelevantRank:
    def test_returns_one_indexed_rank_of_first_expected_page(self):
        assert first_relevant_rank([9, 24, 3], [24]) == 2

    def test_returns_none_when_nothing_matches(self):
        assert first_relevant_rank([1, 2, 3], [24]) is None

    def test_returns_none_for_an_empty_retrieval(self):
        assert first_relevant_rank([], [24]) is None

    def test_a_control_question_never_matches(self):
        """No expected pages means nothing can be relevant - not everything."""
        assert first_relevant_rank([1, 2, 3], []) is None

    def test_stops_at_the_first_match_not_the_best(self):
        """Rank is about position, so a later expected page must not win."""
        assert first_relevant_rank([12, 13], [12, 13]) == 1


class TestReciprocalRank:
    def test_rank_one_scores_full_credit(self):
        assert reciprocal_rank([24, 1, 2], [24]) == 1.0

    def test_rank_three_scores_one_third(self):
        assert reciprocal_rank([1, 2, 24], [24]) == 1 / 3

    def test_a_miss_scores_zero(self):
        assert reciprocal_rank([1, 2, 3], [24]) == 0.0

    def test_position_matters_more_than_presence(self):
        """The reranker's whole job is moving a hit from rank 3 to rank 1."""
        assert reciprocal_rank([24, 1], [24]) > reciprocal_rank([1, 24], [24])


class TestScoreQuestion:
    def test_scores_an_answerable_hit(self):
        result = score_question("q", [24], [1, 24])

        assert result.hit is True
        assert result.reciprocal_rank == 0.5
        assert result.is_control is False
        assert result.retrieved_pages == (1, 24)

    def test_scores_an_answerable_miss(self):
        result = score_question("q", [24], [1, 2])

        assert result.hit is False
        assert result.reciprocal_rank == 0.0
        assert result.is_control is False

    def test_marks_a_question_with_no_expected_pages_as_a_control(self):
        result = score_question("q", [], [])

        assert result.is_control is True
        assert result.abstained is True

    def test_a_control_that_retrieved_something_did_not_abstain(self):
        """Returning chunks for a question the corpus cannot answer is the failure."""
        result = score_question("q", [], [7])

        assert result.is_control is True
        assert result.abstained is False


class TestSummarize:
    def test_reports_hit_rate_over_answerable_questions_only(self):
        results = [
            score_question("a", [1], [1]),
            score_question("b", [2], [9]),
            score_question("control", [], []),
        ]

        summary = summarize(results)

        assert summary["answerable"] == 2
        assert summary["controls"] == 1
        assert summary["hit_rate"] == 0.5

    def test_abstention_rate_covers_controls_only(self):
        results = [
            score_question("a", [1], [1]),
            score_question("c1", [], []),
            score_question("c2", [], [4]),
        ]

        assert summarize(results)["abstention_rate"] == 0.5

    def test_mrr_averages_reciprocal_ranks(self):
        results = [score_question("a", [1], [1]), score_question("b", [2], [9, 2])]

        assert summarize(results)["mrr"] == (1.0 + 0.5) / 2

    def test_a_control_only_run_reports_zero_hit_rate_without_dividing_by_zero(self):
        """An empty answerable slice must score 0, not raise."""
        summary = summarize([score_question("c", [], [])])

        assert summary["hit_rate"] == 0.0
        assert summary["mrr"] == 0.0
        assert summary["abstention_rate"] == 1.0

    def test_an_empty_run_summarizes_to_zeros(self):
        assert summarize([]) == {
            "questions": 0,
            "answerable": 0,
            "controls": 0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "abstention_rate": 0.0,
        }


class TestGoldenDataset:
    """The dataset is ground truth - a typo in it silently corrupts every metric."""

    def test_every_question_is_unique(self):
        questions = [q["question"] for q in _load_golden()]

        assert len(questions) == len(set(questions))

    def test_every_entry_has_a_question_and_expected_pages(self):
        for entry in _load_golden():
            assert entry["question"].strip()
            assert isinstance(entry["expected_pages"], list)

    def test_expected_pages_are_in_range_for_the_source_pdf(self):
        """Pages are 0-indexed and the book has 36 - an out-of-range page can never hit."""
        for entry in _load_golden():
            for page in entry["expected_pages"]:
                assert isinstance(page, int)
                assert 0 <= page < 36, entry["question"]

    def test_the_set_has_both_answerable_questions_and_controls(self):
        """Without controls, lowering the threshold always looks like an improvement."""
        entries = _load_golden()
        controls = [e for e in entries if not e["expected_pages"]]

        assert len(controls) >= 3
        assert len(entries) - len(controls) >= 10


def _load_golden() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text())["questions"]
