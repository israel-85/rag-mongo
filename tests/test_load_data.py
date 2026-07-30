from langchain_core.documents import Document

from load_data import SCHEMA, filter_pages, make_batches, merge_tags, tag_page


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
