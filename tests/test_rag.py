import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pymongo
from langchain_core.documents import Document

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


def test_retriever_config_activates_the_score_threshold():
    """search_type must match search_kwargs, or score_threshold is silently ignored."""
    search_type, search_kwargs = rag.retriever_config()

    assert search_type == "similarity_score_threshold"
    assert search_kwargs["score_threshold"] == 0.75
    assert search_kwargs["k"] == rag.TOP_K
    assert search_kwargs["pre_filter"] == {"hasCode": {"$eq": False}}


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


def test_format_context_joins_chunks_with_a_blank_line():
    """Chunks are separate passages - a blank line keeps them from reading as one."""
    docs = [Document(page_content="first"), Document(page_content="second")]

    assert rag.format_context(docs) == "first\n\nsecond"


def test_format_context_handles_a_single_chunk():
    """One chunk means no separator at all - no leading or trailing blank line."""
    assert rag.format_context([Document(page_content="only")]) == "only"


def test_format_context_handles_no_chunks():
    """Retrieval can return nothing; main() refuses rather than prompting on empty context."""
    assert rag.format_context([]) == ""


def test_format_context_drops_blank_chunks():
    """A whitespace-only chunk carries no information - it must not look like context."""
    assert rag.format_context([Document(page_content="   \n ")]) == ""


def test_format_context_skips_blank_chunks_between_real_ones():
    """Dropping a blank chunk must not leave a widened gap in the joined context."""
    docs = [
        Document(page_content="first"),
        Document(page_content="  "),
        Document(page_content="second"),
    ]

    assert rag.format_context(docs) == "first\n\nsecond"


def test_format_context_preserves_indentation_inside_a_chunk():
    """Only fully blank chunks are dropped - leading whitespace can be meaningful."""
    assert rag.format_context([Document(page_content="  indented")]) == "  indented"


def test_format_context_keeps_chunk_text_verbatim():
    """Only page_content reaches the prompt - metadata must not leak in."""
    docs = [Document(page_content="body", metadata={"title": "leak", "hasCode": True})]

    assert rag.format_context(docs) == "body"


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
