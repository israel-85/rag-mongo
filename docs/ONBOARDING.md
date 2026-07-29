# Onboarding Guide: rag-mongo

## Overview

Two-script RAG demo over MongoDB Atlas Vector Search.

- **Ingest** (`load_data.py`): PDF → filter noise pages → LLM tag metadata →
  chunk → Voyage embed → Atlas.
- **Retrieve** (`rag.py`): query → Voyage embed → Atlas vector search →
  printed snippets.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| Orchestration | LangChain (openai, community, mongodb, voyageai, text-splitters) |
| Vector DB | MongoDB Atlas Vector Search |
| Embeddings | Voyage AI (`voyage-3.5-lite`) |
| LLM tagging | OpenAI-compatible local server (LM Studio) via `ChatOpenAI` |
| Testing | pytest |
| Dev tooling | ruff, vulture, pyright — run ephemerally via `uvx` / `npx` |

No `requirements.txt` — deps live only in `.venv`. `pyproject.toml` holds
pytest config only, not packaging metadata.

## Architecture

Three modules, ~180 lines total. Both scripts follow the same convention:
pure logic extracted into top-level functions for unit testing, all side
effects (Mongo/PDF/LLM/Voyage) confined to `main()` behind
`if __name__ == "__main__":`, so importing a module triggers no network calls.

`config.py` holds the three settings both halves must agree on — `DB_NAME`,
`COLLECTION_NAME`, `EMBED_MODEL`. It is a separate module rather than living in
`load_data.py` so that the query path does not import the PDF/ingest stack
(that cost 251 ms of `rag`'s 581 ms import for code it never calls).

## Key Entry Points

- **Ingest script**: `load_data.py` — run via `python load_data.py`
- **Query script**: `rag.py` — run via `python rag.py "your question"`
- **Shared settings**: `config.py` — `DB_NAME`, `COLLECTION_NAME`, `EMBED_MODEL`
- **Secrets**: `key_param.py` (gitignored) — `MONGODB_URI`, `VOYAGE_API_KEY`,
  `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- **Secrets template**: `key_param.example.py`
- **Sample input**: `sample_files/mongodb.pdf`

## Directory Map

```
config.py               → DB_NAME, COLLECTION_NAME, EMBED_MODEL (shared by both scripts)
load_data.py            → ingest: pure fns top-level, side effects in main()
rag.py                  → retrieval: pure fns top-level, side effects in main()
tests/conftest.py       → mocked LLM/Document fixtures
tests/test_load_data.py → ingest unit tests
tests/test_rag.py       → retrieval unit tests + import-hygiene guards
key_param.py            → gitignored secrets
key_param.example.py    → secrets template
sample_files/           → sample PDF input
docs/testing/           → TDD evidence reports
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

## Retrieval Lifecycle (`rag.py`)

1. `resolve_query(argv)` — first CLI argument, else `DEFAULT_QUERY`.
2. `make_embeddings()` — Voyage client on the shared `EMBED_MODEL`.
3. `main()` builds `MongoDBAtlasVectorSearch` against `vector_index` and
   retrieves `TOP_K=3` chunks by similarity.
4. `format_results(docs)` — page-numbered snippets truncated to 300 chars, or
   `"No matching chunks found."` for zero hits.

## Gotcha: embedding dimensions

Query and stored vectors must come from the same model. The Atlas
`vector_index` must declare `numDimensions: 1024` to match `voyage-3.5-lite`.
A mismatch does not raise — search just returns nothing useful. This is why
`EMBED_MODEL` is single-sourced in `config.py`, and why the test suite asserts
both scripts agree on it.

## Conventions

- Pure logic top-level, testable without network; all I/O confined to
  `main()`.
- Tests use mocked LLM/`Document` fixtures (`tests/conftest.py`) — never hit
  real Mongo/Voyage/LM Studio.

## Common Tasks

```bash
cp key_param.example.py key_param.py     # fill in secrets, gitignored
python load_data.py                      # run the ingestion pipeline
python rag.py                            # query with the demo question
python rag.py "how does sharding work?"  # query with your own
pytest tests/ -v                         # run unit tests
uvx ruff check .                         # lint without touching .venv
npx pyright                              # type check
```

## Where to Look

| I want to... | Look at... |
|--------------|-----------|
| Change page filtering | `filter_pages` in `load_data.py` |
| Change LLM tagging schema/logic | `tag_page` / `merge_tags` in `load_data.py` |
| Change chunking/embedding/upsert | `main()` in `load_data.py` |
| Change batching/rate-limit behavior | `make_batches`, `EMBED_BATCH_SIZE`, `EMBED_BATCH_SLEEP_SECONDS` |
| Change what a query returns | `format_results` / `TOP_K` in `rag.py` |
| Change the query source | `resolve_query` in `rag.py` |
| Switch embedding model | `EMBED_MODEL` in `config.py` — **and** the Atlas index dimensions |
| Add a test | `tests/test_load_data.py` or `tests/test_rag.py`, fixtures in `tests/conftest.py` |
| Update secrets shape | `key_param.example.py` |
