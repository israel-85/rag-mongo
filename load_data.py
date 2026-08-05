import argparse
import hashlib
import time
from collections.abc import Iterator
from typing import Any, cast

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.index import (
    # public helper, but the module declares no __all__
    create_fulltext_search_index,  # pyright: ignore[reportPrivateImportUsage]
)
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr
from pymongo import MongoClient
from tenacity import Retrying, stop_after_attempt, wait_exponential

import key_param
from config import (
    COLLECTION_NAME,
    DB_NAME,
    EMBED_MODEL,
    FULLTEXT_INDEX_NAME,
    TEXT_KEY,
    require_secrets,
)

DEFAULT_PDF_PATH = "./sample_files/mongodb.pdf"

# Voyage free tier allows 3 requests/min and 10K tokens/min, so embeddings are
# pushed in throttled batches instead of one big from_documents() call.
EMBED_BATCH_SIZE = 50
EMBED_BATCH_SLEEP_SECONDS = 25

# A rate-limited batch is the expected failure on the free tier, not an exceptional
# one. Backoff starts above the 20s refill window and caps well under it going long.
EMBED_MAX_ATTEMPTS = 5
EMBED_RETRY_WAIT_MULTIPLIER = 15
EMBED_RETRY_MAX_WAIT_SECONDS = 120

SCHEMA = {
    "title": "DocumentMetadata",
    "description": "Metadata describing a page of documentation.",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

TAG_PROMPT = (
    "Extract metadata describing the following documentation page. "
    "Set hasCode to true only if the page contains code or shell commands.\n\n"
)

def filter_pages(pages: list[Document]) -> list[Document]:
    """Drop pages with 20 or fewer words (front matter/noise)."""
    return [page for page in pages if len(page.page_content.split()) > 20]


def merge_tags(page: Document, tags: dict[str, Any], schema: dict[str, Any]) -> Document:
    """Return a copy of the page with recognized schema keys merged into metadata."""
    return Document(
        page_content=page.page_content,
        metadata={**page.metadata, **{k: tags[k] for k in schema["properties"] if k in tags}},
    )


def tag_page(page: Document, tagger: Any, schema: dict[str, Any] = SCHEMA) -> Document:
    """Return a copy of the page with LLM-extracted metadata merged in."""
    try:
        tags = cast(dict[str, Any], tagger.invoke(TAG_PROMPT + page.page_content))
    except Exception as error:  # tagging is enrichment - never fail the ingest
        print(f"  metadata tagging failed ({type(error).__name__}: {error}); skipping")
        return page
    return merge_tags(page, tags, schema)


def make_batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    """Yield successive batch_size-sized slices of items."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def chunk_id(doc: Document) -> str:
    """Stable id for a chunk - same text from the same page always hashes the same.

    Handing these to add_documents turns every insert into an upsert (the store
    issues ReplaceOne(_id, upsert=True)), so re-running ingestion rewrites the same
    documents instead of appending a second copy of the whole corpus. Duplicates are
    worse than they look: identical chunks crowd each other out of a top-k window,
    so a silent re-ingest degrades retrieval rather than merely wasting space.

    Content is part of the hash, not just source+page, so an edited page produces a
    new id rather than quietly overwriting a chunk it no longer matches.
    """
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    return hashlib.sha256(f"{source}|{page}|{doc.page_content}".encode()).hexdigest()


def _report_retry(state: Any) -> None:
    sleep = getattr(state.next_action, "sleep", 0)
    error = state.outcome.exception()
    print(
        f"  batch failed ({type(error).__name__}: {error}); "
        f"attempt {state.attempt_number}/{EMBED_MAX_ATTEMPTS}, retrying in {sleep:.0f}s"
    )


def store_batch(vector_store: MongoDBAtlasVectorSearch, batch: list[Document]) -> list[str]:
    """Upsert one batch of chunks, retrying with exponential backoff.

    Retrying is only safe because chunk_id is deterministic: a batch that failed
    halfway replays onto the same _ids, so the retry repairs the run instead of
    doubling it. Without that, a mid-run 429 left a half-loaded collection and no
    way to continue except wiping it.
    """
    retryer = Retrying(
        stop=stop_after_attempt(EMBED_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=EMBED_RETRY_WAIT_MULTIPLIER, max=EMBED_RETRY_MAX_WAIT_SECONDS
        ),
        before_sleep=_report_retry,
        reraise=True,
    )
    return retryer(vector_store.add_documents, batch, ids=[chunk_id(doc) for doc in batch])


def ensure_fulltext_index(collection: Any) -> bool:
    """Create the full-text index hybrid retrieval needs, if it is not already there.

    Ingestion owns this because it owns the collection's shape. Creating it lazily at
    query time would mean the first query after a fresh load either blocks on an index
    build or silently returns lexical nothing while it catches up.

    Returns True when it created the index, False when one already existed.
    """
    existing = {index["name"] for index in collection.list_search_indexes()}
    if FULLTEXT_INDEX_NAME in existing:
        return False

    create_fulltext_search_index(collection, index_name=FULLTEXT_INDEX_NAME, field=TEXT_KEY)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI surface for the ingest. --fresh exists because ids only dedupe like-for-like.

    Deterministic ids make a re-ingest idempotent while the chunking settings hold.
    Change chunk_size or the source PDF and the new chunks hash differently, leaving
    the old ones orphaned in the collection - --fresh is how you clear them.
    """
    parser = argparse.ArgumentParser(description="Ingest a PDF into Atlas Vector Search.")
    parser.add_argument("pdf", nargs="?", default=DEFAULT_PDF_PATH, help="path to the source PDF")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete existing chunks before loading (use after changing chunking settings)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    require_secrets(key_param)

    client: MongoClient[dict[str, Any]] = MongoClient(key_param.MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    if args.fresh:
        removed = collection.delete_many({}).deleted_count
        print(f"--fresh: removed {removed} existing documents")

    loader = PyPDFLoader(args.pdf)
    pages = loader.load()
    cleaned_pages = filter_pages(pages)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

    # LM Studio exposes an OpenAI-compatible server; it supports structured output via
    # response_format=json_schema but not the legacy function/tool calling that
    # langchain_community's create_metadata_tagger relies on.
    llm = ChatOpenAI(
        api_key=SecretStr(key_param.LLM_API_KEY),
        base_url=key_param.LLM_BASE_URL,
        temperature=0,
        model=key_param.LLM_MODEL,
    )
    tagger = llm.with_structured_output(SCHEMA, method="json_schema")

    print(f"Tagging {len(cleaned_pages)} pages with {key_param.LLM_MODEL}...")
    # sequential: LM Studio serves one request at a time locally, concurrency wouldn't help
    docs = [tag_page(page, tagger) for page in cleaned_pages]

    split_docs = text_splitter.split_documents(docs)

    embeddings = VoyageAIEmbeddings(
        api_key=SecretStr(key_param.VOYAGE_API_KEY), model=EMBED_MODEL
    )

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection, embedding=embeddings, text_key=TEXT_KEY
    )

    if ensure_fulltext_index(collection):
        print(f"Created full-text index {FULLTEXT_INDEX_NAME!r} on '{TEXT_KEY}' (builds async)")

    batches = list(make_batches(split_docs, EMBED_BATCH_SIZE))
    print(f"Embedding {len(split_docs)} chunks in batches of {EMBED_BATCH_SIZE}...")
    stored_so_far = 0
    for i, batch in enumerate(batches):
        # only newly created _ids come back; on a re-ingest this is 0 and that is the point
        new_ids = store_batch(vector_store, batch)
        stored_so_far += len(batch)
        print(f"  upserted {len(batch)} chunks, {len(new_ids)} new ({stored_so_far}/{len(split_docs)})")
        if i + 1 < len(batches):
            time.sleep(EMBED_BATCH_SLEEP_SECONDS)

    print(f"Done. {collection.count_documents({})} documents in {DB_NAME}.{COLLECTION_NAME}")
    client.close()

if __name__ == "__main__":
    main()
