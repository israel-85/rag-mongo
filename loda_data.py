from collections.abc import Iterator
from typing import Any, cast
import time

from pydantic import SecretStr
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import key_param

# Voyage free tier allows 3 requests/min and 10K tokens/min, so embeddings are
# pushed in throttled batches instead of one big from_documents() call.
EMBED_BATCH_SIZE = 50
EMBED_BATCH_SLEEP_SECONDS = 25

DB_NAME = "book_mongodb_chunks"
COLLECTION_NAME = "chunked_data"

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


def main() -> None:
    client: MongoClient[dict[str, Any]] = MongoClient(key_param.MONGODB_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    loader = PyPDFLoader("./sample_files/mongodb.pdf")
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
        api_key=SecretStr(key_param.VOYAGE_API_KEY), model="voyage-3.5-lite"
    )

    vectorStore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings)

    batches = list(make_batches(split_docs, EMBED_BATCH_SIZE))
    print(f"Embedding {len(split_docs)} chunks in batches of {EMBED_BATCH_SIZE}...")
    for i, batch in enumerate(batches):
        inserted_ids = vectorStore.add_documents(batch)
        stored_so_far = sum(len(b) for b in batches[: i + 1])
        print(f"  stored {len(inserted_ids)} ids ({stored_so_far}/{len(split_docs)})")
        if i + 1 < len(batches):
            time.sleep(EMBED_BATCH_SLEEP_SECONDS)

    print(f"Done. {collection.count_documents({})} documents in {DB_NAME}.{COLLECTION_NAME}")
    client.close()


if __name__ == "__main__":
    main()
