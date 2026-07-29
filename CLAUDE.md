# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two-script RAG demo over MongoDB Atlas Vector Search:

- **Ingest** (`loda_data.py`): PDF → cleaned/filtered pages → LLM metadata tagging → chunking →
  Voyage embeddings → Atlas.
- **Retrieve** (`rag.py`): query → Voyage embedding → Atlas vector search → printed snippets.

`config.py` holds the three settings both halves must agree on.

## Setup & running

No `requirements.txt` — deps live only in `.venv` (langchain, langchain-openai,
langchain-community, langchain-mongodb, langchain-voyageai, langchain-text-splitters, pymongo,
pydantic, voyageai, pytest). `pyproject.toml` holds only pytest config, not packaging metadata.

```bash
cp key_param.example.py key_param.py   # fill in MONGODB_URI, VOYAGE_API_KEY, LLM_* — gitignored
python loda_data.py                    # run the ingestion pipeline
python rag.py                          # query with the built-in demo question
python rag.py "how does sharding work?"  # or pass your own query
pytest tests/ -v                       # run unit tests (no external services needed)
```

`key_param.py` expects an LM Studio (or other OpenAI-compatible) local server for `LLM_BASE_URL`
running the model named in `LLM_MODEL`. Only ingestion uses it — retrieval never calls the LLM.

Dev tooling is run ephemerally so the `.venv` stays as-is: `uvx ruff check`, `uvx vulture`,
`npx pyright` (config in `pyrightconfig.json`).

## Shared config (`config.py`)

`DB_NAME`, `COLLECTION_NAME`, `EMBED_MODEL` live here and are imported by both scripts. They are
in their own module for two reasons:

1. **Correctness** — query and stored vectors must come from the same embedding model. A mismatch
   is silent: Atlas returns nothing or nonsense rather than erroring.
2. **Import cost** — `rag.py` previously read these from `loda_data`, which pulled `PyPDFLoader`
   and the sunset `langchain_community` into the query path (251 ms of a 581 ms import, for code
   retrieval never calls).

## Atlas vector index

The `vector_index` search index on `book_mongodb_chunks.chunked_data` must declare
`numDimensions: 1024` to match `voyage-3.5-lite`. If you switch `EMBED_MODEL`, update the index
too — a dimension mismatch fails silently. Update and poll to `READY` with:

```python
collection.update_search_index("vector_index", {"fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
    {"type": "filter", "path": "hasCode"}]})
```

## Ingestion architecture (`loda_data.py`)

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
4. `main()` embeds with Voyage AI (`EMBED_MODEL`) and upserts into `MongoDBAtlasVectorSearch`
   (db `DB_NAME`, collection `COLLECTION_NAME`), closing the `MongoClient` when done.
5. `make_batches(items, batch_size)` slices chunks for the throttled embed loop
   (`EMBED_BATCH_SIZE=50`, `EMBED_BATCH_SLEEP_SECONDS=25`) to respect Voyage's free-tier rate limit
   (3 req/min, 10K tokens/min) — do not remove the sleep without checking the current Voyage tier.

## Retrieval architecture (`rag.py`)

Same convention: pure functions at top level, all I/O in `main()` behind `if __name__`.

1. `resolve_query(argv)` — first CLI argument, else `DEFAULT_QUERY`.
2. `make_embeddings()` — Voyage client on `EMBED_MODEL`. Must stay in sync with ingestion; that is
   what `config.EMBED_MODEL` is for.
3. `main()` builds `MongoDBAtlasVectorSearch` against `INDEX_NAME` and retrieves `TOP_K=3` chunks
   by similarity.
4. `format_results(docs)` — renders page-numbered snippets truncated to `SNIPPET_CHARS=300`, or
   `"No matching chunks found."` for zero hits.

## Tests

- `tests/test_loda_data.py` covers `filter_pages`, `merge_tags`, `tag_page`, `make_batches`.
- `tests/test_rag.py` covers `resolve_query`, `format_results`, embedding/namespace agreement with
  `config`, and two structural guards: importing `rag` must construct no `MongoClient`, and must
  not load `langchain_community` (checked in a subprocess, since the ingest module is already in
  `sys.modules` inside the pytest session).

All tests use mocked LLM/`Document` fixtures (`tests/conftest.py`) — no real Mongo/Voyage/LM Studio
calls. `main()` in both scripts is I/O-only and verified by running the scripts, not by tests.

## Secrets

`key_param.py` (gitignored, see `key_param.example.py` for the template) holds all secrets:

| Variable | Required | Used by | Description |
|----------|----------|---------|-------------|
| `MONGODB_URI` | Yes | both | Atlas connection string (`mongodb+srv://user:pass@cluster.mongodb.net`) |
| `VOYAGE_API_KEY` | Yes | both | Voyage AI key for embeddings |
| `LLM_API_KEY` | Yes | ingest | Ignored by LM Studio, but the client requires a value |
| `LLM_BASE_URL` | Yes | ingest | OpenAI-compatible endpoint, e.g. `http://127.0.0.1:1234/v1` |
| `LLM_MODEL` | Yes | ingest | Model name exactly as shown in LM Studio |

Never commit this file. Non-secret settings belong in `config.py`, not here.
