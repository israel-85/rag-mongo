# Onboarding Guide: rag-mongo

## Overview

Two-script RAG demo over MongoDB Atlas Vector Search.

- **Ingest** (`load_data.py`): PDF → filter noise pages → LLM tag metadata →
  chunk → Voyage embed → Atlas.
- **Retrieve** (`rag.py`): query → Voyage embed → Atlas vector search →
  streamed LLM answer. Retrieved chunk text is never printed.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python |
| Orchestration | LangChain (openai, community, mongodb, voyageai, text-splitters) |
| Vector DB | MongoDB Atlas Vector Search |
| Embeddings | Voyage AI (`voyage-3.5-lite`) |
| LLM tagging | OpenAI-compatible local server (LM Studio) via `ChatOpenAI` |
| Testing | pytest |
| Dev tooling | ruff, vulture, pyright — run ephemerally via `uvx` so `.venv` stays as-is |

No `requirements.txt` — deps live only in `.venv`. `pyproject.toml` holds
pytest config only, not packaging metadata.

## Architecture

Three modules, ~230 lines total. Both scripts follow the same convention:
pure logic extracted into top-level functions for unit testing, all side
effects (Mongo/PDF/LLM/Voyage) confined to `main()` behind
`if __name__ == "__main__":`, so importing a module triggers no network calls.

`config.py` holds the three settings both halves must agree on — `DB_NAME`,
`COLLECTION_NAME`, `EMBED_MODEL`. It is a separate module rather than living in
`load_data.py` so that the query path does not import the PDF/ingest stack
(that cost 251 ms of `rag`'s 581 ms import for code it never calls).

## Key Entry Points

- **Ingest script**: `load_data.py` — run via `python load_data.py`
  (`--fresh` to wipe first, or pass a different PDF path)
- **Query script**: `rag.py` — run via `python rag.py "your question"`
- **Eval harness**: `evaluation/run_eval.py` — `python -m evaluation.run_eval`
- **Threshold calibration**: `evaluation/calibrate.py` — `python -m evaluation.calibrate`
- **Shared settings**: `config.py` — `DB_NAME`, `COLLECTION_NAME`, `EMBED_MODEL`,
  `VECTOR_INDEX_NAME`, `FULLTEXT_INDEX_NAME`, `TEXT_KEY`, `require_secrets()`
- **Secrets**: `key_param.py` (gitignored) — `MONGODB_URI`, `VOYAGE_API_KEY`,
  `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
- **Secrets template**: `key_param.example.py`
- **Sample input**: `sample_files/mongodb.pdf`

## Directory Map

```
config.py               → shared names, index names, TEXT_KEY, require_secrets()
load_data.py            → ingest: pure fns top-level, side effects in main()
rag.py                  → retrieval: pure fns top-level, side effects in main()
evaluation/metrics.py   → hit_rate / mrr / abstention_rate (pure, no I/O)
evaluation/golden.json  → 22 golden questions + ground-truth pages
evaluation/run_eval.py  → live retrieval scoring (needs Atlas + Voyage)
evaluation/calibrate.py → rerank score distribution, for setting thresholds
tests/conftest.py       → mocked LLM/Document fixtures
tests/test_load_data.py → ingest unit tests
tests/test_rag.py       → retrieval unit tests + import-hygiene guards
tests/test_config.py    → secret validation + docs/template drift guards
tests/test_metrics.py   → metric scoring + golden.json validation
key_param.py            → gitignored secrets
key_param.example.py    → secrets template
sample_files/           → sample PDF input
docs/ONBOARDING.md      → this file
docs/CONTRIBUTING.md    → setup, commands, quality gates, PR checklist
docs/RUNBOOK.md         → Atlas index state, procedures, troubleshooting
docs/CODE_TOUR.md       → ingestion walkthrough (load_data.py)
docs/CODE_TOUR_RAG.md   → retrieval walkthrough (rag.py)
docs/testing/           → TDD evidence reports (historical — record past state)
.tours/                 → CodeTour JSON of the ingestion walkthrough
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

1. `resolve_query(argv)` — first CLI argument, stripped; blank or absent falls
   back to `DEFAULT_QUERY`, so `python rag.py ""` does not embed an empty
   string.
2. `make_embeddings()` — Voyage client on the shared `EMBED_MODEL`.
3. `build_candidate_retriever()` casts a wide net — `CANDIDATE_K` (20)
   candidates from `MongoDBAtlasHybridSearchRetriever`, which runs vector search
   and full-text search and fuses the two ranked lists by Reciprocal Rank Fusion.
   Falls back to vector-only if the `text_index` full-text index is missing.
   No `score_threshold` and no `pre_filter` here on purpose: this stage owns
   recall, and gating it would cap what the reranker can ever see.
4. `score_candidates()` scores those 20 with Voyage `rerank-2.5` in one call;
   `keep_above()` keeps the `TOP_K` (3) clearing `RERANK_THRESHOLD` (0.55). An
   empty result triggers one retry at `RELAXED_RERANK_THRESHOLD` (0.45) —
   re-filtering the *same* scores, not re-querying and not re-scoring. Both
   numbers come from `python -m evaluation.calibrate`, which measured a clean
   gap: highest off-topic control 0.4023, lowest answerable question 0.6875.
5. `format_context(docs)` — numbers each chunk and labels it `mongodb.pdf p.24`
   so the answer is traceable, dropping fully blank chunks so `""` always means
   "nothing to answer from". Only `source`/`page` ride along; the LLM-extracted
   `title`/`keywords`/`hasCode` tags never reach the prompt.
6. `main()` refuses on empty context: prints `NO_CONTEXT_MESSAGE` and returns
   without building the LLM. Structural, because a small local model may
   ignore the prompt's "do not answer without context" instruction.
7. `stream_answer(chunks)` — echoes the LLM's tokens as they arrive, flushing
   per token, then `format_sources(docs)` prints the citation key. Retrieved
   chunk *text* is never printed; only the query, the answer, and the labels.

**Before changing anything in this path, measure it.** Retrieval fails silently
— a wrong chunk produces a fluent, confident, wrong answer with no exception and
no failing test. `python -m evaluation.run_eval --verbose` scores 22 golden
questions. Watch `hit_rate` and `abstention_rate` together; they trade against
each other and either alone will flatter a regression:

| | hit_rate | mrr | abstention_rate |
|---|---|---|---|
| vector threshold 0.75/0.71 + `hasCode` filter, k=3 | 0.389 | 0.389 | 1.000 |
| hybrid + rerank, uncalibrated floor (0.5) | 1.000 | 1.000 | 0.250 |
| hybrid + rerank, calibrated floor (0.55) | 1.000 | 1.000 | 1.000 |

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
pytest tests/ --cov=rag --cov=load_data --cov-report=term-missing   # coverage
uvx ruff check .                         # lint without touching .venv
uvx ruff check --preview --select E301,E302,E303,E305 .   # blank-line spacing
uvx pyright                              # type check (uvx, not npx)
uvx vulture . --exclude .venv            # dead-code scan
```

## Where to Look

| I want to... | Look at... |
|--------------|-----------|
| Change page filtering | `filter_pages` in `load_data.py` |
| Change LLM tagging schema/logic | `tag_page` / `merge_tags` in `load_data.py` |
| Change chunking/embedding/upsert | `main()` in `load_data.py` |
| Change batching/rate-limit behavior | `make_batches`, `EMBED_BATCH_SIZE`, `EMBED_BATCH_SLEEP_SECONDS` |
| Change what a query returns | `TOP_K` / retriever `search_kwargs` in `rag.py` |
| Change the query source | `resolve_query` in `rag.py` |
| Change what the LLM sees as context | `format_context` in `rag.py` |
| Change the no-results behavior | `NO_CONTEXT_MESSAGE` + the guard in `main()`, `rag.py` |
| Switch embedding model | `EMBED_MODEL` in `config.py` — **and** the Atlas index dimensions |
| Add a test | `tests/test_load_data.py` or `tests/test_rag.py`, fixtures in `tests/conftest.py` |
| Update secrets shape | `key_param.example.py` |
