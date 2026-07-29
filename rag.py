from typing import Any

from pydantic import SecretStr
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_voyageai import VoyageAIEmbeddings

import key_param

DB_NAME = "book_mongodb_chunks"
COLLECTION_NAME = "chunked_data"
INDEX_NAME = "vector_index"
TOP_K = 3
SNIPPET_CHARS = 300

QUERY = "When did MongoDB begin supporting multi-document transactions?"


def make_embeddings() -> VoyageAIEmbeddings:
    """Same model as ingestion - query and stored vectors must share dimensions."""
    return VoyageAIEmbeddings(
        api_key=SecretStr(key_param.VOYAGE_API_KEY), model="voyage-3.5-lite"
    )


def format_results(docs: list[Document]) -> str:
    """Render retrieved chunks as page-numbered snippets."""
    return "\n\n".join(
        f"[{i}] page {doc.metadata.get('page', '?')} - {doc.metadata.get('title', 'untitled')}\n"
        f"{doc.page_content[:SNIPPET_CHARS].strip()}"
        for i, doc in enumerate(docs, 1)
    )


def main() -> None:
    client: MongoClient[dict[str, Any]] = MongoClient(key_param.MONGODB_URI)
    vector_store = MongoDBAtlasVectorSearch(
        collection=client[DB_NAME][COLLECTION_NAME],
        embedding=make_embeddings(),
        index_name=INDEX_NAME,
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    print(f"Query: {QUERY}\n")
    print(format_results(retriever.invoke(QUERY)))
    client.close()


if __name__ == "__main__":
    main()
