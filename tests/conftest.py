import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.modules.setdefault("key_param", MagicMock())


@pytest.fixture
def make_page():
    def _make(word_count: int, metadata: dict | None = None) -> Document:
        content = " ".join(f"word{i}" for i in range(word_count))
        return Document(page_content=content, metadata=metadata or {})

    return _make


@pytest.fixture
def mock_tagger():
    tagger = MagicMock()
    tagger.invoke.return_value = {
        "title": "Test Title",
        "keywords": ["mongo", "vector"],
        "hasCode": True,
    }
    return tagger
