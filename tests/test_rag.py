import importlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pymongo

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


def test_format_results_signals_zero_hits():
    """An empty result set must say so, not print a blank line."""
    assert "no matching" in rag.format_results([]).lower()


def test_resolve_query_prefers_cli_argument():
    assert rag.resolve_query(["rag.py", "how do indexes work?"]) == "how do indexes work?"


def test_resolve_query_falls_back_to_default():
    assert rag.resolve_query(["rag.py"]) == rag.DEFAULT_QUERY


def test_format_results_prints_source_and_snippet():
    """Results render as page-numbered snippets, not raw Document repr."""
    from langchain_core.documents import Document

    docs = [Document(page_content="x" * 400, metadata={"page": 7, "title": "Txns"})]

    lines = rag.format_results(docs)

    assert "page 7" in lines
    assert "Txns" in lines
    assert len(lines) < 400
