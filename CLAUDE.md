# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Single-script RAG ingestion pipeline: PDF → cleaned/filtered pages → LLM metadata tagging →
chunking → Voyage embeddings → MongoDB Atlas Vector Search.

## Setup & running

No `requirements.txt` — deps live only in `.venv` (langchain, langchain-openai,
langchain-community, langchain-mongodb, langchain-voyageai, langchain-text-splitters, pymongo,
pydantic, voyageai, pytest). `pyproject.toml` holds only pytest config, not packaging metadata.

```bash
cp key_param.example.py key_param.py   # fill in MONGODB_URI, VOYAGE_API_KEY, LLM_* — gitignored
python loda_data.py                    # run the ingestion pipeline
pytest tests/ -v                       # run unit tests (no external services needed)
```

`key_param.py` expects an LM Studio (or other OpenAI-compatible) local server for `LLM_BASE_URL`
running the model named in `LLM_MODEL`.

## Architecture (`loda_data.py`)

Pure logic is extracted into top-level functions for unit testing; all side effects (Mongo/PDF/LLM/
Voyage) live in `main()`, guarded by `if __name__ == "__main__":` so importing the module (e.g. from
tests) triggers no network calls:

1. `filter_pages(pages)` — drop pages with ≤20 words (front matter/noise).
2. `tag_page(page, tagger)` tags a page with LLM-extracted metadata (`title`, `keywords`, `hasCode`)
   via `ChatOpenAI.with_structured_output(schema, method="json_schema")` — required because the local
   LM Studio server supports `response_format=json_schema` but not legacy function/tool calling.
   Tagging failures are swallowed (catches and returns the untagged page) since enrichment must
   never abort the ingest. Merging is delegated to `merge_tags(page, tags, schema)`.
3. `main()` splits tagged docs with `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=150).
4. `main()` embeds with Voyage AI (`voyage-3.5-lite`) and upserts into `MongoDBAtlasVectorSearch`
   (db `book_mongodb_chunks`, collection `chunked_data`), closing the `MongoClient` when done.
5. `make_batches(items, batch_size)` slices chunks for the throttled embed loop
   (`EMBED_BATCH_SIZE=50`, `EMBED_BATCH_SLEEP_SECONDS=25`) to respect Voyage's free-tier rate limit
   (3 req/min, 10K tokens/min) — do not remove the sleep without checking the current Voyage tier.

`tests/test_loda_data.py` covers `filter_pages`, `merge_tags`, `tag_page`, `make_batches` with mocked
LLM/`Document` fixtures (`tests/conftest.py`) — no real Mongo/Voyage/LM Studio calls.

## Config

`key_param.py` (gitignored, see `key_param.example.py` for the template) holds all secrets/config:
`MONGODB_URI`, `VOYAGE_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`. Never commit this file.
