"""Settings shared by ingestion (load_data.py) and retrieval (rag.py).

Kept in its own module so the query path does not import the PDF/ingest stack.
Query and stored vectors must come from the same embedding model, or vector
search silently returns nothing.
"""

from collections.abc import Sequence
from typing import Any

DB_NAME = "book_mongodb_chunks"
COLLECTION_NAME = "chunked_data"
EMBED_MODEL = "voyage-3.5-lite"

# Atlas index names and the document field the chunk text is stored under. The
# full-text index must map exactly this field, or hybrid search's lexical half
# matches nothing and silently degrades to vector-only.
VECTOR_INDEX_NAME = "vector_index"
FULLTEXT_INDEX_NAME = "text_index"
TEXT_KEY = "text"

# Every name key_param.py must define. Both halves need all five: ingestion tags
# pages with the LLM, retrieval generates the answer with it.
REQUIRED_SECRETS = (
    "MONGODB_URI",
    "VOYAGE_API_KEY",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
)


def missing_secrets(module: Any, names: Sequence[str] = REQUIRED_SECRETS) -> list[str]:
    """Names that are absent, None, or blank. Pure - the caller decides what to do."""
    return [
        name
        for name in names
        if _is_unset(getattr(module, name, None))
    ]


def _is_unset(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def require_secrets(module: Any, names: Sequence[str] = REQUIRED_SECRETS) -> None:
    """Fail at startup with a readable message instead of a driver traceback later.

    Without this, a missing MONGODB_URI surfaces 30 seconds in as a 40-line
    ServerSelectionTimeoutError, and a blank VOYAGE_API_KEY surfaces only after the
    ingest has already spent minutes tagging pages. Checking at the boundary turns
    both into one line, before any work or any spend.
    """
    missing = missing_secrets(module, names)
    if missing:
        raise SystemExit(
            "key_param.py is missing or has blank values for: "
            + ", ".join(missing)
            + "\nCopy key_param.example.py to key_param.py and fill in every field."
        )
