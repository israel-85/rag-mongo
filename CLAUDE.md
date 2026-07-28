# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Single-script RAG ingestion pipeline: PDF → cleaned/filtered pages → LLM metadata tagging →
chunking → Voyage embeddings → MongoDB Atlas Vector Search.

## Setup & running

No `requirements.txt`/`pyproject.toml` — deps live only in `.venv` (langchain, langchain-openai,
langchain-community, langchain-mongodb, langchain-voyageai, langchain-text-splitters, pymongo,
pydantic, voyageai).

```bash
cp key_param.example.py key_param.py   # fill in MONGODB_URI, VOYAGE_API_KEY, LLM_* — gitignored
python loda_data.py                    # run the ingestion pipeline
```

`key_param.py` expects an LM Studio (or other OpenAI-compatible) local server for `LLM_BASE_URL`
running the model named in `LLM_MODEL`.

## Architecture (`loda_data.py`)

Linear pipeline, no functions/classes except `tag_page`:

1. Load `sample_files/mongodb.pdf` via `PyPDFLoader`; drop pages with ≤20 words (front matter/noise).
2. Tag each page with LLM-extracted metadata (`title`, `keywords`, `hasCode`) via
   `ChatOpenAI.with_structured_output(schema, method="json_schema")` — required because the local
   LM Studio server supports `response_format=json_schema` but not legacy function/tool calling.
   Tagging failures are swallowed (`tag_page` catches and returns the untagged page) since
   enrichment must never abort the ingest.
3. Split tagged docs with `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=150).
4. Embed with Voyage AI (`voyage-3.5-lite`) and upsert into `MongoDBAtlasVectorSearch`
   (db `book_mongodb_chunks`, collection `chunked_data`).
5. Embedding batches are throttled (`EMBED_BATCH_SIZE=50`, `EMBED_BATCH_SLEEP_SECONDS=25`) to
   respect Voyage's free-tier rate limit (3 req/min, 10K tokens/min) — do not remove the sleep
   without checking the current Voyage tier.

## Config

`key_param.py` (gitignored, see `key_param.example.py` for the template) holds all secrets/config:
`MONGODB_URI`, `VOYAGE_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`. Never commit this file.
