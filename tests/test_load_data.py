from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import load_data
from load_data import (
    SCHEMA,
    chunk_id,
    filter_pages,
    make_batches,
    merge_tags,
    parse_args,
    store_batch,
    tag_page,
)


class TestFilterPages:
    def test_drops_pages_at_or_below_20_words(self, make_page):
        pages = [make_page(20), make_page(21)]
        assert filter_pages(pages) == [pages[1]]

    def test_keeps_pages_above_20_words_with_extra_whitespace(self):
        page = Document(page_content="word " * 25 + "\n\t  ")
        assert filter_pages([page]) == [page]

    def test_empty_input_returns_empty_list(self):
        assert filter_pages([]) == []


class TestMergeTags:
    def test_merges_recognized_schema_keys_into_metadata(self, make_page):
        page = make_page(25, metadata={"source": "doc.pdf"})
        tags = {"title": "T", "keywords": ["a"], "hasCode": False, "unrelated": "x"}

        result = merge_tags(page, tags, SCHEMA)

        assert result.metadata == {
            "source": "doc.pdf",
            "title": "T",
            "keywords": ["a"],
            "hasCode": False,
        }
        assert result.page_content == page.page_content

    def test_ignores_missing_tag_keys(self, make_page):
        page = make_page(25)
        result = merge_tags(page, {"title": "T"}, SCHEMA)
        assert result.metadata == {"title": "T"}


class TestTagPage:
    def test_returns_page_with_merged_tags_on_success(self, make_page, mock_tagger):
        page = make_page(25)
        result = tag_page(page, mock_tagger)
        assert result.metadata["title"] == "Test Title"
        assert result.metadata["hasCode"] is True

    def test_returns_untagged_page_on_tagger_failure(self, make_page, mock_tagger):
        page = make_page(25)
        mock_tagger.invoke.side_effect = RuntimeError("LM Studio down")
        result = tag_page(page, mock_tagger)
        assert result is page


class TestMakeBatches:
    def test_splits_into_exact_multiples(self):
        items = list(range(10))
        assert list(make_batches(items, 5)) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def test_handles_remainder(self):
        items = list(range(7))
        assert list(make_batches(items, 3)) == [[0, 1, 2], [3, 4, 5], [6]]

    def test_empty_input_yields_no_batches(self):
        assert list(make_batches([], 5)) == []


def _chunk(text: str = "body", source: str = "a.pdf", page: int = 3) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


class TestChunkId:
    """Ids are what make a re-ingest an upsert instead of a duplicate corpus."""

    def test_is_stable_across_calls(self):
        assert chunk_id(_chunk()) == chunk_id(_chunk())

    def test_differs_when_the_text_differs(self):
        assert chunk_id(_chunk(text="one")) != chunk_id(_chunk(text="two"))

    def test_differs_when_the_page_differs(self):
        """Identical boilerplate on two pages must stay two documents, not overwrite."""
        assert chunk_id(_chunk(page=1)) != chunk_id(_chunk(page=2))

    def test_differs_when_the_source_differs(self):
        assert chunk_id(_chunk(source="a.pdf")) != chunk_id(_chunk(source="b.pdf"))

    def test_ignores_unrelated_metadata(self):
        """LLM tags are enrichment - a re-tag must not orphan the chunk it describes."""
        tagged = Document(
            page_content="body",
            metadata={"source": "a.pdf", "page": 3, "title": "T", "hasCode": True},
        )

        assert chunk_id(tagged) == chunk_id(_chunk())

    def test_survives_missing_metadata(self):
        """A loader that supplies no source/page must still produce an id, not crash."""
        assert chunk_id(Document(page_content="body"))

    def test_is_not_a_valid_object_id(self):
        """A 24-char hex id would be coerced to an ObjectId; 64 chars stays a string _id."""
        assert len(chunk_id(_chunk())) == 64


class TestStoreBatch:
    def test_passes_deterministic_ids_to_the_vector_store(self):
        vector_store = MagicMock()
        vector_store.add_documents.return_value = ["x"]
        batch = [_chunk(text="one"), _chunk(text="two")]

        store_batch(vector_store, batch)

        _, kwargs = vector_store.add_documents.call_args
        assert kwargs["ids"] == [chunk_id(batch[0]), chunk_id(batch[1])]

    def test_returns_the_ids_the_store_reports(self):
        vector_store = MagicMock()
        vector_store.add_documents.return_value = ["new-id"]

        assert store_batch(vector_store, [_chunk()]) == ["new-id"]

    def test_retries_a_failing_batch_and_succeeds(self, monkeypatch):
        """A rate-limited batch must not abort a run that already paid for prior batches."""
        monkeypatch.setattr(load_data, "EMBED_RETRY_WAIT_MULTIPLIER", 0)
        vector_store = MagicMock()
        vector_store.add_documents.side_effect = [RuntimeError("429"), ["new-id"]]

        assert store_batch(vector_store, [_chunk()]) == ["new-id"]
        assert vector_store.add_documents.call_count == 2

    def test_replays_the_same_ids_on_retry(self, monkeypatch):
        """Retrying is only safe if the second attempt targets the same _ids."""
        monkeypatch.setattr(load_data, "EMBED_RETRY_WAIT_MULTIPLIER", 0)
        vector_store = MagicMock()
        vector_store.add_documents.side_effect = [RuntimeError("429"), []]
        batch = [_chunk()]

        store_batch(vector_store, batch)

        first, second = vector_store.add_documents.call_args_list
        assert first.kwargs["ids"] == second.kwargs["ids"] == [chunk_id(batch[0])]

    def test_gives_up_after_the_attempt_cap_and_reraises(self, monkeypatch):
        """A permanently broken batch must surface, not be swallowed as a silent gap."""
        monkeypatch.setattr(load_data, "EMBED_RETRY_WAIT_MULTIPLIER", 0)
        vector_store = MagicMock()
        vector_store.add_documents.side_effect = RuntimeError("down")

        with pytest.raises(RuntimeError, match="down"):
            store_batch(vector_store, [_chunk()])

        assert vector_store.add_documents.call_count == load_data.EMBED_MAX_ATTEMPTS


class TestEnsureFulltextIndex:
    def test_creates_the_index_when_it_is_absent(self, monkeypatch):
        created = {}
        monkeypatch.setattr(
            load_data,
            "create_fulltext_search_index",
            lambda collection, index_name, field: created.update(name=index_name, field=field),
        )
        collection = MagicMock()
        collection.list_search_indexes.return_value = [{"name": "vector_index"}]

        assert load_data.ensure_fulltext_index(collection) is True
        assert created == {"name": load_data.FULLTEXT_INDEX_NAME, "field": load_data.TEXT_KEY}

    def test_is_a_no_op_when_the_index_already_exists(self, monkeypatch):
        """Re-running the ingest must not thrash an index that is already built."""
        monkeypatch.setattr(
            load_data,
            "create_fulltext_search_index",
            lambda *a, **k: pytest.fail("must not recreate an existing index"),
        )
        collection = MagicMock()
        collection.list_search_indexes.return_value = [
            {"name": "vector_index"},
            {"name": load_data.FULLTEXT_INDEX_NAME},
        ]

        assert load_data.ensure_fulltext_index(collection) is False

    def test_indexes_the_field_the_vector_store_writes_text_to(self):
        """A full-text index on the wrong field matches nothing, silently."""
        import config

        assert load_data.TEXT_KEY == config.TEXT_KEY


class TestParseArgs:
    def test_defaults_to_the_sample_pdf_and_no_wipe(self):
        args = parse_args([])

        assert args.pdf == load_data.DEFAULT_PDF_PATH
        assert args.fresh is False

    def test_accepts_an_alternate_pdf_path(self):
        assert parse_args(["other.pdf"]).pdf == "other.pdf"

    def test_fresh_flag_is_opt_in(self):
        """Wiping the collection must never be the accidental default."""
        assert parse_args(["--fresh"]).fresh is True
