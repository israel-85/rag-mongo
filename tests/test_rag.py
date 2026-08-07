import importlib
import itertools
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymongo
import pymongo.errors
import pytest
import voyageai.error
from langchain_core.documents import Document
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever

import config
import load_data
import rag

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_opens_no_mongo_connection():
    """Importing rag must not construct a MongoClient - all I/O lives in main()."""
    try:
        with patch.object(pymongo, "MongoClient") as mongo_client:
            importlib.reload(rag)
            # the patch must actually reach rag's namespace, or the assertion below is vacuous
            assert rag.MongoClient is mongo_client

        mongo_client.assert_not_called()
    finally:
        importlib.reload(rag)

    # the reload must leave no mock behind for later tests
    assert rag.MongoClient is pymongo.MongoClient


def test_import_stays_out_of_the_ingest_stack():
    """The query path must not drag in the PDF/ingest dependencies."""
    probe = "import sys, rag; print('langchain_community' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=REPO_ROOT
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_embeddings_match_ingest_model():
    """Query and ingest must share one embedding model - drift silently breaks retrieval."""
    embeddings = rag.make_embeddings()

    assert embeddings.model == config.EMBED_MODEL
    assert load_data.EMBED_MODEL is config.EMBED_MODEL
    assert (rag.DB_NAME, rag.COLLECTION_NAME) == (config.DB_NAME, config.COLLECTION_NAME)


def test_retriever_config_casts_a_wide_net_for_the_reranker():
    """This stage owns recall; a score_threshold here would cap what reranking can see."""
    search_type, search_kwargs = rag.retriever_config()

    assert search_type == "similarity"
    assert search_kwargs["k"] == rag.CANDIDATE_K
    assert search_kwargs["k"] > rag.TOP_K
    assert "score_threshold" not in search_kwargs


def test_retriever_config_does_not_filter_out_code_pages():
    """The old hasCode pre_filter made every code-bearing page permanently unreachable."""
    _, search_kwargs = rag.retriever_config()

    assert "pre_filter" not in search_kwargs


class FakeRerankResult:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


class FakeRerankResponse:
    def __init__(self, results: list[FakeRerankResult]):
        self.results = results


class FakeReranker:
    """Records the call and returns scores the test dictates, in the order given."""

    def __init__(self, scored: list[tuple[int, float]]):
        self.scored = scored
        self.calls: list[dict] = []

    def rerank(self, query, documents, model, top_k):
        self.calls.append(
            {"query": query, "documents": documents, "model": model, "top_k": top_k}
        )
        return FakeRerankResponse([FakeRerankResult(i, s) for i, s in self.scored])


def _docs(*texts: str) -> list[Document]:
    return [Document(page_content=t, metadata={"page": i}) for i, t in enumerate(texts)]


class TestScoreCandidates:
    def test_returns_pairs_in_the_reranker_order(self):
        """The reranker's job is moving the real answer to rank 1."""
        docs = _docs("a", "b", "c")
        reranker = FakeReranker([(2, 0.9), (0, 0.8), (1, 0.7)])

        scored = rag.score_candidates("q", docs, reranker)

        assert [doc.page_content for doc, _ in scored] == ["c", "a", "b"]
        assert [score for _, score in scored] == [0.9, 0.8, 0.7]

    def test_skips_the_api_call_when_there_are_no_candidates(self):
        """No candidates means nothing to score - do not spend a request finding out."""
        reranker = FakeReranker([])

        assert rag.score_candidates("q", [], reranker) == []
        assert reranker.calls == []

    def test_sends_only_chunk_text_and_asks_for_top_k(self):
        docs = _docs("a", "b")
        reranker = FakeReranker([(0, 0.9)])

        rag.score_candidates("how do indexes work?", docs, reranker, top_k=3)

        call = reranker.calls[0]
        assert call["documents"] == ["a", "b"]
        assert call["query"] == "how do indexes work?"
        assert call["model"] == rag.RERANK_MODEL
        assert call["top_k"] == 3

    def test_preserves_metadata_needed_for_citations(self):
        """Returning Documents, not bare scores, is what keeps source/page available."""
        reranker = FakeReranker([(1, 0.9)])

        scored = rag.score_candidates("q", _docs("a", "b"), reranker)

        assert scored[0][0].metadata["page"] == 1


class TestKeepAbove:
    def test_drops_documents_below_the_threshold(self):
        scored = [(_docs("a")[0], 0.9), (_docs("b")[0], 0.2)]

        assert [d.page_content for d in rag.keep_above(scored, 0.5)] == ["a"]

    def test_keeps_a_document_exactly_at_the_threshold(self):
        """The floor is inclusive - a score equal to it is not a near miss."""
        assert rag.keep_above([(_docs("a")[0], 0.5)], 0.5)

    def test_returns_nothing_when_all_scores_are_weak(self):
        """Empty here means the same as retrieving nothing: main() refuses to answer."""
        scored = [(_docs("a")[0], 0.1), (_docs("b")[0], 0.05)]

        assert rag.keep_above(scored, 0.5) == []

    def test_preserves_the_reranker_order(self):
        """Filtering must not resort - rank order is the reranker's output, not ours."""
        docs = _docs("a", "b", "c")
        scored = [(docs[2], 0.9), (docs[0], 0.8), (docs[1], 0.7)]

        assert [d.page_content for d in rag.keep_above(scored, 0.0)] == ["c", "a", "b"]

    def test_handles_no_scores_at_all(self):
        assert rag.keep_above([], 0.5) == []


_namespace_counter = itertools.count()


def _store_with_indexes(*names: str) -> MagicMock:
    """A store on its own namespace, so the per-namespace index cache cannot leak."""
    store = MagicMock()
    store.collection.database.name = "db"
    store.collection.name = f"coll{next(_namespace_counter)}"
    store.collection.list_search_indexes.return_value = [{"name": n} for n in names]
    return store


class TestBuildCandidateRetriever:
    def test_uses_hybrid_search_when_the_fulltext_index_exists(self):
        """Lexical search is what catches quoted tokens like $unwind that vectors blur."""
        store = _store_with_indexes(rag.INDEX_NAME, rag.FULLTEXT_INDEX_NAME)

        with patch.object(rag, "MongoDBAtlasHybridSearchRetriever") as hybrid:
            rag.build_candidate_retriever(store)

        store.as_retriever.assert_not_called()
        assert hybrid.call_args.kwargs == {
            "vectorstore": store,
            "search_index_name": rag.FULLTEXT_INDEX_NAME,
            "k": rag.CANDIDATE_K,
        }

    def test_the_real_retriever_accepts_those_arguments(self):
        """Guard the patched test above: the kwargs must match the real signature."""
        fields = MongoDBAtlasHybridSearchRetriever.model_fields

        assert {"vectorstore", "search_index_name", "k"} <= set(fields)

    def test_falls_back_to_vector_only_when_the_fulltext_index_is_missing(self):
        """A collection ingested before the index existed must still answer."""
        store = _store_with_indexes(rag.INDEX_NAME)

        rag.build_candidate_retriever(store)

        store.as_retriever.assert_called_once()
        assert store.as_retriever.call_args.kwargs["search_type"] == "similarity"

    def test_falls_back_when_listing_indexes_fails(self):
        """A permissions or connectivity hiccup listing indexes must not sink the query."""
        store = _store_with_indexes()
        store.collection.list_search_indexes.side_effect = RuntimeError("no permission")

        rag.build_candidate_retriever(store)

        store.as_retriever.assert_called_once()

    def test_checks_for_the_index_only_once_per_namespace(self):
        """Index presence cannot change mid-run; asking per query is a wasted round-trip."""
        store = _store_with_indexes(rag.INDEX_NAME)

        rag.build_candidate_retriever(store)
        rag.build_candidate_retriever(store)

        assert store.collection.list_search_indexes.call_count == 1


class TestRetrieve:
    def _vector_store(self, docs):
        store = _store_with_indexes(rag.INDEX_NAME)  # vector-only path, no live hybrid
        store.as_retriever.return_value.invoke.return_value = docs
        return store

    def test_returns_the_reranked_survivors(self):
        store = self._vector_store(_docs("a", "b", "c"))
        reranker = FakeReranker([(1, 0.9)])

        result = rag.retrieve(store, "q", reranker)

        assert [d.page_content for d in result] == ["b"]

    def test_scores_once_even_when_the_relaxed_pass_runs(self):
        """Rerank scores are deterministic - re-scoring to lower a floor pays twice
        for an identical answer, on exactly the queries that are already slowest."""
        store = self._vector_store(_docs("a"))
        reranker = FakeReranker([(0, 0.4)])

        rag.retrieve(store, "q", reranker)

        assert store.as_retriever.return_value.invoke.call_count == 1
        assert len(reranker.calls) == 1

    def test_applies_both_thresholds_to_the_same_scores(self):
        """The relaxed pass must reuse the scores, not re-derive them."""
        store = self._vector_store(_docs("a", "b"))
        reranker = FakeReranker([(0, 0.48), (1, 0.2)])

        result = rag.retrieve(store, "q", reranker)

        assert [d.page_content for d in result] == ["a"]
        assert len(reranker.calls) == 1

    def test_the_relaxed_pass_rescues_a_near_miss(self):
        store = self._vector_store(_docs("a"))
        reranker = FakeReranker([(0, rag.RELAXED_RERANK_THRESHOLD)])

        assert len(rag.retrieve(store, "q", reranker)) == 1

    def test_returns_nothing_when_even_the_relaxed_pass_fails(self):
        store = self._vector_store(_docs("a"))
        reranker = FakeReranker([(0, 0.01)])

        assert rag.retrieve(store, "q", reranker) == []

    def test_treats_whitespace_only_survivors_as_nothing(self):
        """A blank chunk clearing the floor must not count as usable context."""
        store = self._vector_store([Document(page_content="   ", metadata={"page": 1})])
        reranker = FakeReranker([(0, 0.99)])

        assert rag.retrieve(store, "q", reranker) == []


class FlakyReranker(FakeReranker):
    """Raises `error` on the first `failures` calls, then behaves like FakeReranker."""

    def __init__(self, scored, failures: int, error: Exception | None = None):
        super().__init__(scored)
        self.failures = failures
        self.error = error or voyageai.error.RateLimitError("rate limit exceeded")

    def rerank(self, query, documents, model, top_k):
        self.calls.append({"query": query, "documents": documents})
        if len(self.calls) <= self.failures:
            raise self.error
        return FakeRerankResponse([FakeRerankResult(i, s) for i, s in self.scored])


class TestRetryPolicy:
    """A 429 from Voyage's 3-requests-per-minute tier is routine, not exceptional.

    Every test zeroes the wait multiplier - otherwise the suite would sit through a
    real exponential backoff, and a slow test suite stops being run.
    """

    @pytest.fixture(autouse=True)
    def _no_sleeping(self, monkeypatch):
        monkeypatch.setattr(rag, "RETRIEVE_RETRY_WAIT_MULTIPLIER", 0)

    def _vector_store(self, docs):
        store = _store_with_indexes(rag.INDEX_NAME)
        store.as_retriever.return_value.invoke.return_value = docs
        return store

    def test_retries_a_rate_limited_rerank_and_succeeds(self):
        """One 429 must not end a query the user is waiting on."""
        reranker = FlakyReranker([(0, 0.9)], failures=1)

        scored = rag.score_candidates("q", _docs("a"), reranker)

        assert [score for _, score in scored] == [0.9]
        assert len(reranker.calls) == 2

    def test_retries_a_failing_atlas_query_and_succeeds(self):
        docs = _docs("a", "b")
        store = self._vector_store(docs)
        store.as_retriever.return_value.invoke.side_effect = [
            pymongo.errors.ConnectionFailure("connection reset"),
            docs,
        ]
        reranker = FakeReranker([(1, 0.9)])

        result = rag.retrieve(store, "q", reranker)

        assert [d.page_content for d in result] == ["b"]
        assert store.as_retriever.return_value.invoke.call_count == 2

    def test_reraises_a_permanently_failing_rerank(self):
        """Exhausted retries must raise, never return [].

        Returning [] would make "the reranker is down" indistinguishable from "the
        corpus cannot answer this", and main() would print NO_CONTEXT_MESSAGE over a
        broken pipeline.
        """
        reranker = FlakyReranker([(0, 0.9)], failures=rag.RETRIEVE_MAX_ATTEMPTS)

        with pytest.raises(voyageai.error.RateLimitError, match="rate limit"):
            rag.score_candidates("q", _docs("a"), reranker)

        assert len(reranker.calls) == rag.RETRIEVE_MAX_ATTEMPTS

    def test_reraises_a_permanently_failing_atlas_query(self):
        store = self._vector_store(_docs("a"))
        store.as_retriever.return_value.invoke.side_effect = pymongo.errors.ServerSelectionTimeoutError(
            "no primary"
        )

        with pytest.raises(pymongo.errors.ServerSelectionTimeoutError, match="no primary"):
            rag.retrieve(store, "q", FakeReranker([(0, 0.9)]))

        assert store.as_retriever.return_value.invoke.call_count == rag.RETRIEVE_MAX_ATTEMPTS

    def test_spends_no_retry_budget_when_there_are_no_candidates(self):
        """The empty guard sits above the retry - a call that never happens cannot fail."""
        reranker = FlakyReranker([], failures=rag.RETRIEVE_MAX_ATTEMPTS)

        assert rag.score_candidates("q", [], reranker) == []
        assert reranker.calls == []

    def test_a_healthy_query_calls_each_service_once(self):
        """Backoff must be invisible on the happy path, not an extra round-trip."""
        store = self._vector_store(_docs("a"))
        reranker = FakeReranker([(0, 0.9)])

        rag.retrieve(store, "q", reranker)

        assert store.as_retriever.return_value.invoke.call_count == 1
        assert len(reranker.calls) == 1

    def test_does_not_retry_a_non_transient_error(self):
        """A bad API key must fail in one call, not be retried for ~4.5 minutes.

        Without a type filter this looked identical to a rate limit from the outside
        - eight attempts of the same auth failure, printed as if it might resolve.
        """
        reranker = FlakyReranker(
            [(0, 0.9)], failures=rag.RETRIEVE_MAX_ATTEMPTS, error=voyageai.error.AuthenticationError("bad key")
        )

        with pytest.raises(voyageai.error.AuthenticationError):
            rag.score_candidates("q", _docs("a"), reranker)

        assert len(reranker.calls) == 1

    def test_does_not_retry_a_non_transient_atlas_error(self):
        store = self._vector_store(_docs("a"))
        store.as_retriever.return_value.invoke.side_effect = pymongo.errors.OperationFailure(
            "bad filter"
        )

        with pytest.raises(pymongo.errors.OperationFailure):
            rag.retrieve(store, "q", FakeReranker([(0, 0.9)]))

        assert store.as_retriever.return_value.invoke.call_count == 1


def test_resolve_query_prefers_cli_argument():
    assert rag.resolve_query(["rag.py", "how do indexes work?"]) == "how do indexes work?"


def test_resolve_query_falls_back_to_default():
    assert rag.resolve_query(["rag.py"]) == rag.DEFAULT_QUERY


def test_resolve_query_ignores_empty_argument():
    """An empty arg is not a question - embedding it retrieves noise."""
    assert rag.resolve_query(["rag.py", ""]) == rag.DEFAULT_QUERY


def test_resolve_query_ignores_blank_argument():
    """Same for whitespace-only - shell quoting makes this easy to hit by accident."""
    assert rag.resolve_query(["rag.py", "   "]) == rag.DEFAULT_QUERY


def test_resolve_query_strips_surrounding_whitespace():
    """A padded query is still a query; the padding is not part of it."""
    assert rag.resolve_query(["rag.py", "  how do indexes work?  "]) == "how do indexes work?"


def _cited(text: str, page: int = 0, source: str = "./sample_files/mongodb.pdf") -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


class TestCitationLabel:
    def test_uses_the_file_name_and_a_one_based_page(self):
        """Metadata pages are 0-indexed; a citation the reader can check is not."""
        assert rag.citation_label(_cited("x", page=23)) == "mongodb.pdf p.24"

    def test_falls_back_to_the_source_alone_when_there_is_no_page(self):
        doc = Document(page_content="x", metadata={"source": "a.pdf"})

        assert rag.citation_label(doc) == "a.pdf"

    def test_survives_a_chunk_with_no_metadata(self):
        assert rag.citation_label(Document(page_content="x")) == "unknown source"


def test_format_context_numbers_and_labels_each_chunk():
    """A claim the reader cannot trace to a page is not verifiable."""
    docs = [_cited("first", page=0), _cited("second", page=4)]

    assert rag.format_context(docs) == (
        "[1] (mongodb.pdf p.1)\nfirst\n\n[2] (mongodb.pdf p.5)\nsecond"
    )


def test_format_context_handles_a_single_chunk():
    """One chunk means no separator at all - no leading or trailing blank line."""
    assert rag.format_context([_cited("only")]) == "[1] (mongodb.pdf p.1)\nonly"


def test_format_context_handles_no_chunks():
    """Retrieval can return nothing; main() refuses rather than prompting on empty context."""
    assert rag.format_context([]) == ""


def test_format_context_drops_blank_chunks():
    """A whitespace-only chunk carries no information - it must not look like context."""
    assert rag.format_context([_cited("   \n ")]) == ""


def test_format_context_renumbers_after_dropping_a_blank_chunk():
    """Numbering must be gapless, or a cited [3] points at a chunk that is not there."""
    docs = [_cited("first", page=0), _cited("  ", page=1), _cited("second", page=2)]

    assert rag.format_context(docs) == (
        "[1] (mongodb.pdf p.1)\nfirst\n\n[2] (mongodb.pdf p.3)\nsecond"
    )


def test_format_context_preserves_indentation_inside_a_chunk():
    """Only fully blank chunks are dropped - leading whitespace can be meaningful."""
    assert rag.format_context([_cited("  indented")]).endswith("\n  indented")


def test_format_context_keeps_retrieval_tags_out_of_the_prompt():
    """Source and page are citation data; title/keywords/hasCode would read as content."""
    doc = Document(
        page_content="body",
        metadata={"source": "a.pdf", "page": 0, "title": "leak", "keywords": ["k"], "hasCode": True},
    )

    context = rag.format_context([doc])

    assert "leak" not in context
    assert "hasCode" not in context
    assert context == "[1] (a.pdf p.1)\nbody"


def test_format_sources_numbering_matches_format_context():
    """If these two drift, [2] in the answer cites a different page than [2] on screen."""
    docs = [_cited("first", page=0), _cited("  ", page=1), _cited("second", page=2)]

    assert rag.format_sources(docs) == "  [1] mongodb.pdf p.1\n  [2] mongodb.pdf p.3"


def test_format_sources_never_prints_chunk_text():
    """Only the query and the answer reach the console - not the retrieved passages."""
    assert "first" not in rag.format_sources([_cited("first")])


class RecordingStream:
    """Captures write/flush order so tests can prove output is not buffered."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def write(self, text: str) -> int:
        self.calls.append(("write", text))
        return len(text)

    def flush(self) -> None:
        self.calls.append(("flush", ""))


def test_stream_answer_writes_each_token_separately():
    """Tokens reach the terminal one at a time - no waiting for the full answer."""
    out = RecordingStream()

    rag.stream_answer(iter(["Mongo", "DB ", "4.0"]), out)

    writes = [text for kind, text in out.calls if kind == "write"]
    assert writes[:3] == ["Mongo", "DB ", "4.0"]


def test_stream_answer_flushes_after_every_token():
    """Without a flush per token, stdout buffers and streaming is invisible."""
    out = RecordingStream()

    rag.stream_answer(iter(["a", "b"]), out)

    assert out.calls[:4] == [("write", "a"), ("flush", ""), ("write", "b"), ("flush", "")]


def test_stream_answer_terminates_the_line():
    """The streamed answer has no trailing newline of its own; add one."""
    out = RecordingStream()

    rag.stream_answer(iter(["done"]), out)

    assert out.calls[-1] == ("flush", "")
    assert [text for kind, text in out.calls if kind == "write"][-1] == "\n"


def test_stream_answer_handles_empty_stream():
    """An LLM that returns nothing must not crash the run."""
    out = RecordingStream()

    rag.stream_answer(iter([]), out)

    assert [text for kind, text in out.calls if kind == "write"] == ["\n"]
