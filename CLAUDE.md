# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two-script RAG demo over MongoDB Atlas Vector Search:

- **Ingest** (`load_data.py`): PDF → cleaned/filtered pages → LLM metadata tagging → chunking →
  Voyage embeddings → Atlas.
- **Retrieve** (`rag.py`): query → Voyage embedding → Atlas vector search → printed snippets.

`config.py` holds the three settings both halves must agree on.

## Setup & running

No `requirements.txt` — deps live only in `.venv` (langchain, langchain-openai,
langchain-community, langchain-mongodb, langchain-voyageai, langchain-text-splitters, pymongo,
pydantic, voyageai, pytest, pytest-cov). `pyproject.toml` holds only pytest config, not packaging
metadata.

```bash
cp key_param.example.py key_param.py   # fill in MONGODB_URI, VOYAGE_API_KEY, LLM_* — gitignored
python load_data.py                    # run the ingestion pipeline
python load_data.py --fresh            # wipe existing chunks first (after changing chunking)
python load_data.py other.pdf          # ingest a different source
python rag.py                          # query with the built-in demo question
python rag.py "how does sharding work?"  # or pass your own query
pytest tests/ -v                       # run unit tests (no external services needed)
pytest tests/ --cov=rag --cov=load_data --cov-report=term-missing   # coverage
python -m evaluation.run_eval --verbose  # score retrieval against the golden set (needs Atlas)
```

`key_param.py` expects an LM Studio (or other OpenAI-compatible) local server for `LLM_BASE_URL`
running the model named in `LLM_MODEL`. Both halves use it: ingestion to tag pages with metadata,
retrieval to generate the answer. Only `evaluation/run_eval.py` skips it — it scores retrieval,
which stops before the LLM.

Dev tooling is run ephemerally so the `.venv` stays as-is: `uvx ruff check`, `uvx vulture`,
`uvx pyright` (config in `pyrightconfig.json`). Use `uvx`, not `npx` — `npx pyright` resolves to
something else on at least one dev machine and fails with `Unknown command: "pyright"`.

Operational procedures (Atlas index state, re-ingest, troubleshooting, rollback) live in
`docs/RUNBOOK.md`. Setup, quality gates and the PR checklist live in `docs/CONTRIBUTING.md`.

## Shared config (`config.py`)

`DB_NAME`, `COLLECTION_NAME`, `EMBED_MODEL`, `VECTOR_INDEX_NAME`, `FULLTEXT_INDEX_NAME` and
`TEXT_KEY` live here and are imported by both scripts. `TEXT_KEY` matters as much as `EMBED_MODEL`:
the full-text index must map exactly the field the vector store writes chunk text to, or hybrid
search's lexical half matches nothing and quietly degrades to vector-only.

`require_secrets(key_param)` also lives here and is called first thing in both `main()`s. Without
it a missing `MONGODB_URI` surfaced 30 seconds later as a 40-line `ServerSelectionTimeoutError`,
and a blank `VOYAGE_API_KEY` surfaced only after ingestion had already spent minutes tagging
pages. All five secrets are required by *both* halves — retrieval uses the `LLM_*` settings to
generate the answer, not just ingestion to tag.

The settings are in their own module for two reasons:

1. **Correctness** — query and stored vectors must come from the same embedding model. A mismatch
   is silent: Atlas returns nothing or nonsense rather than erroring.
2. **Import cost** — `rag.py` previously read these from `load_data`, which pulled `PyPDFLoader`
   and the sunset `langchain_community` into the query path (251 ms of a 581 ms import, for code
   retrieval never calls).

## Atlas search indexes

Two are needed: `vector_index` (vector) and `text_index` (full-text, for hybrid's lexical half).
`load_data.py` creates the full-text one automatically via `ensure_fulltext_index`; the vector one
is still created by hand. **Atlas M0 caps the whole cluster at 3 search indexes** — if creation
fails with `The maximum number of FTS indexes has been reached for this instance size.`, drop an
unused one (the `sample_mflix` sample-data indexes are the usual candidates) or upgrade the tier.

### Vector index

The `vector_index` search index on `book_mongodb_chunks.chunked_data` must declare
`numDimensions: 1024` to match `voyage-3.5-lite`. If you switch `EMBED_MODEL`, update the index
too — a dimension mismatch fails silently. Update and poll to `READY` with:

```python
collection.update_search_index("vector_index", {"fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
    {"type": "filter", "path": "hasCode"}]})
```

## Ingestion architecture (`load_data.py`)

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
6. `chunk_id(doc)` hashes `source|page|page_content` into a deterministic `_id`. The store issues
   `ReplaceOne(_id, upsert=True)`, so re-running ingestion **rewrites** the same documents instead
   of appending a second copy of the corpus. This is not just about wasted space: duplicate chunks
   crowd each other out of a `TOP_K` window, so an accidental re-ingest degrades retrieval.
   Content is in the hash, not just source+page, so an edited page yields a new id rather than
   silently overwriting a chunk it no longer matches.
7. `store_batch(vector_store, batch)` wraps the upsert in exponential backoff
   (`EMBED_MAX_ATTEMPTS=5`). A 429 partway through used to abort the run and leave a half-loaded
   collection with no way to continue. Retrying is only safe *because* `chunk_id` is deterministic —
   a batch that failed halfway replays onto the same `_id`s.
8. `--fresh` deletes existing chunks first. Deterministic ids only dedupe like-for-like: change
   `chunk_size` or the source PDF and the new chunks hash differently, orphaning the old ones.
9. `ensure_fulltext_index(collection)` creates `FULLTEXT_INDEX_NAME` on `TEXT_KEY` if absent —
   ingestion owns it because ingestion owns the collection's shape. Note Atlas M0 caps the cluster
   at **3 search indexes total**; hitting the cap fails with
   `The maximum number of FTS indexes has been reached for this instance size.`

## Retrieval architecture (`rag.py`)

Same convention: pure functions at top level, all I/O in `main()` behind `if __name__`.

1. `resolve_query(argv)` — first CLI argument, stripped; if absent or blank, `DEFAULT_QUERY`. The
   blank guard matters because `python rag.py ""` (easy to produce from shell quoting) would
   otherwise embed an empty string and retrieve noise rather than nothing.
2. `make_embeddings()` — Voyage client on `EMBED_MODEL`. Must stay in sync with ingestion; that is
   what `config.EMBED_MODEL` is for.
3. **Retrieve wide, rerank, keep a few.** `build_candidate_retriever()` pulls `CANDIDATE_K=20`
   candidates via `MongoDBAtlasHybridSearchRetriever` (vector + full-text, fused by Reciprocal
   Rank Fusion), then `score_candidates()` scores them with Voyage `rerank-2.5` in **one** call and
   `keep_above()` keeps the `TOP_K=3` that clear `RERANK_THRESHOLD`. `retrieve()` retries once at
   `RELAXED_RERANK_THRESHOLD` by re-filtering **the same scores** — not re-querying Atlas and not
   re-scoring, since rerank scores are deterministic and a second call would pay twice for an
   identical answer. The second chance is about the gate being strict, never the net being small.

   Both network calls — the Atlas candidate query and the Voyage rerank — go through `retryer()`
   (`RETRIEVE_MAX_ATTEMPTS=8`, exponential to 90s), with `_report_retry` printing each attempt so a
   rate limit reads as a slow answer rather than a hang. A query costs two Voyage requests against
   a 3 RPM free tier, so a 429 is the routine failure, not the exceptional one. Only `RETRYABLE_ERRORS`
   (Voyage rate-limit/timeout/connection/server errors, pymongo connection-loss errors) trigger a
   retry — a bad API key or a malformed query fails on the first attempt instead of being retried
   for ~4.5 minutes and misread as a rate limit. `reraise=True` is load-bearing: swallowing an
   exhausted retry would return no documents, making "retrieval is broken" indistinguishable from
   "the corpus has no answer" — the exact silent failure `NO_CONTEXT_MESSAGE` exists to make loud.
   Retriever *construction* stays outside the retry, being a cached index lookup rather than the
   flaky part. `retryer()` has no leading underscore: `evaluation/calibrate.py` needs the identical
   policy for a `top_k=1` query shape `retrieve()` doesn't support, and a private name would mark
   that as reaching into an implementation detail rather than a shared one.

   Three deliberate deletions from the old design, all of which cost recall:
   - **No `score_threshold` at the vector stage.** Gating there caps what the reranker can ever
     see, which defeats retrieving 20 instead of 3. The old `0.75`/`0.71` numbers do **not** carry
     over — they scored vector similarity, a different quantity on a different stage.
   - **No `pre_filter`.** This used to carry `{"hasCode": {"$eq": False}}`, making every
     code-bearing page permanently unreachable. "How do I create an index" is answered by page 24,
     which that filter hid. Excluding code was an assumption, never a requirement.
   - **Lexical search is back.** Embeddings blur exact tokens (`$unwind`, `4.0`, `createIndex`) —
     precisely the strings users quote. RRF merges both ranked lists without either needing a
     calibrated score, which is why it survives an embedding-model swap that would invalidate any
     tuned cutoff.

   `RERANK_THRESHOLD` (0.55) and `RELAXED_RERANK_THRESHOLD` (0.45) were calibrated 2026-08-05 with
   `python -m evaluation.calibrate`: across the 22 golden questions the highest off-topic control
   scored 0.4023 and the lowest answerable question scored 0.6875 — a clean gap with no overlap.
   Re-run calibrate if the corpus, `EMBED_MODEL`, or `RERANK_MODEL` changes. If
   `FULLTEXT_INDEX_NAME` is absent,
   `build_candidate_retriever` prints a note and falls back to vector-only, so a collection
   ingested before the index existed still answers. The check is cached per namespace: index
   presence cannot change mid-run, and asking per query cost an extra Atlas round-trip each time.
4. `format_context(docs)` — numbers each chunk and labels it with `citation_label(doc)`
   (`file p.N`, 1-based, since metadata pages are 0-indexed). Source and page ride along so an
   answer is traceable; the LLM-extracted tags (`title`, `keywords`, `hasCode`) still stay out,
   because they are retrieval metadata and would read as content. `format_sources(docs)` prints
   the same numbering under the answer — both route through `usable_chunks()` precisely so they
   cannot drift, since a mismatch makes `[2]` in the answer cite a different page than `[2]` on
   screen. Blank chunks are dropped, so `""` always means "nothing worth answering from" — never
   "chunks that happen to be whitespace".
5. `main()` refuses on empty context: if `format_context` returns `""` it prints
   `NO_CONTEXT_MESSAGE` and returns without constructing the LLM. This is deliberately structural
   rather than relying on the prompt's "do not answer if there is no given context" line — a small
   local model may ignore that instruction and answer from parametric memory instead.
6. `main()` feeds that context to `PromptTemplate | ChatOpenAI | StrOutputParser`. Retrieved chunk
   text never reaches the console — only the query and the streamed answer are printed. (A
   `format_results` snippet renderer existed for this and was deleted once `main()` stopped
   calling it.)
7. `stream_answer(chunks, out)` — writes each token from `rag_chain.stream()` and flushes after
   every one. The flush is load-bearing: stdout is block-buffered when piped, so without it the
   whole answer lands at once and the stream is invisible. `RunnablePassthrough` is deliberately
   absent — `main()` builds the prompt input dict itself, so there is nothing to pass through.

## Evaluation (`evaluation/`)

Retrieval fails silently — a wrong chunk produces a fluent, confident, wrong answer with no
exception and no red test. The harness is the only thing that catches it, so run it before and
after any retrieval change (threshold, reranker, filter, hybrid).

- `evaluation/golden.json` — 22 questions against `sample_files/mongodb.pdf`, ground truth as the
  0-indexed `page` values that actually answer each one. 18 answerable, 4 off-topic **controls**
  with `expected_pages: []`, where retrieving nothing is the correct outcome. If the source PDF is
  ever replaced, regenerate `expected_pages`.
- `evaluation/metrics.py` — pure scoring: `hit_rate`, `mrr`, `abstention_rate`. Reported
  separately on purpose: lowering a threshold always lifts `hit_rate` while quietly wrecking
  `abstention_rate`, and a single number hides that trade.
- `evaluation/run_eval.py` — live runner (`python -m evaluation.run_eval --verbose --json out.json`).
  Calls `rag.retrieve` directly, so what is measured is what runs — including its backoff. The
  harness used to own a `with_backoff` wrapper of its own; that protected the measurement while
  leaving the thing users actually run bare, so it moved into `rag.py` and the wrapper was deleted
  rather than nested.
- `evaluation/calibrate.py` — prints the top reranker score per question, split answerable vs
  control, and reports whether a gap exists. This is how the two thresholds get set. If the two
  groups ever overlap, no threshold separates them and the fix is better retrieval or a better
  golden set — not a number nudged until it looks right.

**Measured 2026-08-05 against the live 171-chunk corpus:**

| | hit_rate | mrr | abstention_rate |
|---|---|---|---|
| before (vector threshold 0.75/0.71 + `hasCode` filter, k=3) | 0.389 | 0.389 | 1.000 |
| hybrid + rerank, uncalibrated floor (0.5) | 1.000 | 1.000 | 0.250 |
| hybrid + rerank, calibrated floor (0.55) | 1.000 | 1.000 | 1.000 |

Read the middle row as the warning it is: hit_rate tripled while three of four off-topic controls
leaked through. Either number alone would have been reported as a success. In the before row,
`mrr == hit_rate` exactly, meaning every hit landed at rank 1 and the other 11 answerable
questions returned nothing or the wrong pages — the old thresholds bought perfect abstention by
refusing 61% of questions the corpus could actually answer.

## Tests

- `tests/test_load_data.py` covers `filter_pages`, `merge_tags`, `tag_page`, `make_batches`,
  `chunk_id`, `store_batch` (including retry and id-replay on retry), `ensure_fulltext_index`,
  and `parse_args`.
- `tests/test_config.py` covers `missing_secrets`/`require_secrets`.
- `tests/test_metrics.py` covers the scoring functions and validates `golden.json` itself — a
  typo in ground truth silently corrupts every metric.
- `tests/test_rag.py` covers `resolve_query`, `format_context`, `format_sources`, `citation_label`,
  `score_candidates`, `keep_above`, `retrieve` (including that it scores once
  across both thresholds), the retry policy on both network calls (retry-then-succeed,
  reraise-on-exhaustion for each, and that the happy path adds no round-trip),
  `build_candidate_retriever` (both the hybrid and the vector-only
  fallback branch), `stream_answer`, embedding/namespace agreement with `config`, and two
  structural guards: importing `rag` must construct no `MongoClient`, and must not load
  `langchain_community` (checked in a subprocess, since the ingest module is already in
  `sys.modules` inside the pytest session).
  `stream_answer` is tested through a `RecordingStream` fake that logs write/flush order, so the
  "flush after every token" guarantee is asserted rather than assumed.

All tests use mocked LLM/`Document` fixtures (`tests/conftest.py`) — no real Mongo/Voyage/LM Studio
calls. `main()` in both scripts is I/O-only and verified by running the scripts, not by tests.

## Secrets

`key_param.py` (gitignored, see `key_param.example.py` for the template) holds all secrets:

| Variable | Required | Used by | Description |
|----------|----------|---------|-------------|
| `MONGODB_URI` | Yes | both | Atlas connection string (`mongodb+srv://user:pass@cluster.mongodb.net`) |
| `VOYAGE_API_KEY` | Yes | both | Voyage AI key for embeddings |
| `LLM_API_KEY` | Yes | both | Ignored by LM Studio, but the client requires a value |
| `LLM_BASE_URL` | Yes | both | OpenAI-compatible endpoint, e.g. `http://127.0.0.1:1234/v1` |
| `LLM_MODEL` | Yes | both | Model name exactly as shown in LM Studio |

All five are validated by `config.require_secrets(key_param)` at the top of both `main()`s, so a
missing or blank value fails in one readable line before any work or any API spend.

Never commit this file. Non-secret settings belong in `config.py`, not here.
