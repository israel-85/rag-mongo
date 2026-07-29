# Onboarding Guide: rag-mongo

## Overview

Single-script RAG ingestion pipeline. PDF → filter noise pages → LLM tag
metadata → chunk → Voyage embed → MongoDB Atlas Vector Search.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| Orchestration | LangChain (openai, community, mongodb, voyageai, text-splitters) |
| Vector DB | MongoDB Atlas Vector Search |
| Embeddings | Voyage AI (`voyage-3.5-lite`) |
| LLM tagging | OpenAI-compatible local server (LM Studio) via `ChatOpenAI` |
| Testing | pytest |

No `requirements.txt` — deps live only in `.venv`. `pyproject.toml` holds
pytest config only, not packaging metadata.

## Architecture

`loda_data.py` (117 lines) is the whole pipeline. Pure logic is extracted
into top-level functions for unit testing; all side effects (Mongo/PDF/LLM/
Voyage) live in `main()`, guarded by `if __name__ == "__main__":` so
importing the module (e.g. from tests) triggers no network calls.

## Key Entry Points

- **Pipeline script**: `loda_data.py` — run via `python loda_data.py`
- **Config**: `key_param.py` (gitignored) — `MONGODB_URI`, `VOYAGE_API_KEY`,
  `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- **Config template**: `key_param.example.py`
- **Sample input**: `sample_files/mongodb.pdf`

## Directory Map

```
loda_data.py            → pipeline: pure fns top-level, side effects in main()
tests/conftest.py       → mocked LLM/Document fixtures
tests/test_loda_data.py → unit tests, no real Mongo/Voyage/LM Studio calls
key_param.py             → gitignored secrets
key_param.example.py     → secrets template
sample_files/             → sample PDF input
```

## Request/Data Lifecycle

1. `filter_pages(pages)` — drop pages with ≤20 words (front matter/noise).
2. `tag_page(page, tagger)` — LLM-tags a page (`title`, `keywords`,
   `hasCode`) via `ChatOpenAI.with_structured_output(schema,
   method="json_schema")`; local LM Studio server supports
   `response_format=json_schema` but not legacy function/tool calling.
   Tagging failures are swallowed (returns untagged page) — enrichment must
   never abort the ingest. Merging delegated to `merge_tags(page, tags,
   schema)`.
3. `main()` splits tagged docs with `RecursiveCharacterTextSplitter`
   (chunk_size=500, overlap=150).
4. `main()` embeds with Voyage AI and upserts into
   `MongoDBAtlasVectorSearch` (db `book_mongodb_chunks`, collection
   `chunked_data`), closing the `MongoClient` when done.
5. `make_batches(items, batch_size)` slices chunks for the throttled embed
   loop (`EMBED_BATCH_SIZE=50`, `EMBED_BATCH_SLEEP_SECONDS=25`) to respect
   Voyage's free-tier rate limit (3 req/min, 10K tokens/min) — do not remove
   the sleep without checking the current Voyage tier.

## Conventions

- Pure logic top-level, testable without network; all I/O confined to
  `main()`.
- Tests use mocked LLM/`Document` fixtures (`tests/conftest.py`) — never hit
  real Mongo/Voyage/LM Studio.

## Common Tasks

```bash
cp key_param.example.py key_param.py   # fill in secrets, gitignored
python loda_data.py                    # run the ingestion pipeline
pytest tests/ -v                       # run unit tests
```

## Where to Look

| I want to... | Look at... |
|--------------|-----------|
| Change page filtering | `filter_pages` in `loda_data.py` |
| Change LLM tagging schema/logic | `tag_page` / `merge_tags` in `loda_data.py` |
| Change chunking/embedding/upsert | `main()` in `loda_data.py` |
| Change batching/rate-limit behavior | `make_batches`, `EMBED_BATCH_SIZE`, `EMBED_BATCH_SLEEP_SECONDS` |
| Add a test | `tests/test_loda_data.py`, fixtures in `tests/conftest.py` |
| Update secrets/config shape | `key_param.example.py` |
