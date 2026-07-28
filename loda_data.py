import time
from typing import Any, cast

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

# Set the MongoDB URI, DB, Collection Names

client: MongoClient[dict[str, Any]] = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]

loader = PyPDFLoader("./sample_files/mongodb.pdf")
pages = loader.load()
cleaned_pages: list[Document] = []

for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

schema = {
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

# LM Studio exposes an OpenAI-compatible server; it supports structured output via
# response_format=json_schema but not the legacy function/tool calling that
# langchain_community's create_metadata_tagger relies on.
llm = ChatOpenAI(
    api_key=SecretStr(key_param.LLM_API_KEY),
    base_url=key_param.LLM_BASE_URL,
    temperature=0,
    model=key_param.LLM_MODEL,
)
tagger = llm.with_structured_output(schema, method="json_schema")

TAG_PROMPT = (
    "Extract metadata describing the following documentation page. "
    "Set hasCode to true only if the page contains code or shell commands.\n\n"
)


def tag_page(page: Document) -> Document:
    """Return a copy of the page with LLM-extracted metadata merged in."""
    try:
        tags = cast(dict[str, Any], tagger.invoke(TAG_PROMPT + page.page_content))
    except Exception as error:  # tagging is enrichment - never fail the ingest
        print(f"  metadata tagging failed ({type(error).__name__}: {error}); skipping")
        return page
    return Document(
        page_content=page.page_content,
        metadata={**page.metadata, **{k: tags[k] for k in schema["properties"] if k in tags}},
    )


print(f"Tagging {len(cleaned_pages)} pages with {key_param.LLM_MODEL}...")
docs = [tag_page(page) for page in cleaned_pages]

split_docs = text_splitter.split_documents(docs)

embeddings = VoyageAIEmbeddings(
    api_key=SecretStr(key_param.VOYAGE_API_KEY), model="voyage-3.5-lite"
)

vectorStore = MongoDBAtlasVectorSearch(collection=collection, embedding=embeddings)

print(f"Embedding {len(split_docs)} chunks in batches of {EMBED_BATCH_SIZE}...")
for start in range(0, len(split_docs), EMBED_BATCH_SIZE):
    batch = split_docs[start : start + EMBED_BATCH_SIZE]
    inserted_ids = vectorStore.add_documents(batch)
    print(f"  stored {len(inserted_ids)} ids (batch {start + 1}-{start + len(batch)})/{len(split_docs)}")
    if start + EMBED_BATCH_SIZE < len(split_docs):
        time.sleep(EMBED_BATCH_SLEEP_SECONDS)

print(f"Done. {collection.count_documents({})} documents in {dbName}.{collectionName}")
