import rag


def test_import_triggers_no_side_effects():
    """Importing rag must not query Mongo/Voyage - all I/O lives in main()."""
    assert callable(rag.main)


def test_embeddings_match_ingest_model():
    """Query embeddings must use the same model as ingestion (voyage-3.5-lite, 1024-dim)."""
    import loda_data

    embeddings = rag.make_embeddings()

    assert embeddings.model == "voyage-3.5-lite"
    assert rag.DB_NAME == loda_data.DB_NAME
    assert rag.COLLECTION_NAME == loda_data.COLLECTION_NAME


def test_format_results_prints_source_and_snippet():
    """Results render as page-numbered snippets, not raw Document repr."""
    from langchain_core.documents import Document

    docs = [Document(page_content="x" * 400, metadata={"page": 7, "title": "Txns"})]

    lines = rag.format_results(docs)

    assert "page 7" in lines
    assert "Txns" in lines
    assert len(lines) < 400
