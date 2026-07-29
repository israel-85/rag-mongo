import importlib
from unittest.mock import patch

import pymongo

import loda_data
import rag


def test_import_opens_no_mongo_connection():
    """Importing rag must not construct a MongoClient - all I/O lives in main()."""
    with patch.object(pymongo, "MongoClient") as mongo_client:
        importlib.reload(rag)
        # the patch must actually reach rag's namespace, or the assertion below is vacuous
        assert rag.MongoClient is mongo_client

    mongo_client.assert_not_called()


def test_embeddings_match_ingest_model():
    """Query and ingest must share one embedding model - drift silently breaks retrieval."""
    embeddings = rag.make_embeddings()

    assert embeddings.model == loda_data.EMBED_MODEL
    assert rag.DB_NAME == loda_data.DB_NAME
    assert rag.COLLECTION_NAME == loda_data.COLLECTION_NAME


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
