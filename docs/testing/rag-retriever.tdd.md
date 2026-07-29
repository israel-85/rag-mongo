# TDD Evidence — rag.py retriever

Source plan: none; journeys derived during this TDD run (2026-07-29).

## User journeys

1. As a dev, importing `rag` must not fire Mongo/Voyage calls, so it is unit-testable.
2. As a dev, query embeddings must match the ingest model (`voyage-3.5-lite`, 1024-dim), so vector search returns hits.
3. As a user, results print as readable page-numbered snippets, not raw `Document` repr.

## Task report

| Task | Validation | Result |
|------|-----------|--------|
| Reproducer added | `.venv/bin/python -m pytest tests/test_rag.py -x -q` | RED — `pymongo.errors.ConfigurationError: need to specify at least one host` at import time (`rag.py:9`), proving the module-level side effect |
| Fix applied | `.venv/bin/python -m pytest tests/ -q` | GREEN — `13 passed` |
| End-to-end run | `.venv/bin/python rag.py` | 3 chunks printed (pages 31, 32, 32) for the transactions query |

Root cause: `rag.py` embedded queries with `OpenAIEmbeddings` (1536-dim, using the LM Studio
key against api.openai.com) while `loda_data.py` stored Voyage `voyage-3.5-lite` vectors
(1024-dim). The Atlas `vector_index` also declared `numDimensions: 1536` against 1024-dim data;
updated in place via `update_search_index` and polled to `READY`.

## Test specification

| # | Guarantee | Test | Type | Result |
|---|-----------|------|------|--------|
| 1 | Importing `rag` constructs no `MongoClient` | `tests/test_rag.py::test_import_opens_no_mongo_connection` | unit | PASS |
| 2 | Query embedding model + namespace match ingestion | `tests/test_rag.py::test_embeddings_match_ingest_model` | unit | PASS |
| 3 | Results render as truncated page-numbered snippets | `tests/test_rag.py::test_format_results_prints_source_and_snippet` | unit | PASS |

## Follow-up cycle — review findings M1/M2 (2026-07-29)

Second RED/GREEN cycle addressing `.claude/reviews/rag-retriever-review.md`:

| Task | Validation | Result |
|------|-----------|--------|
| M1 reproducer: assert one shared `EMBED_MODEL` | `pytest tests/test_rag.py -q` | RED — `AttributeError: module 'loda_data' has no attribute 'EMBED_MODEL'` |
| M1+M2 fix: hoist `EMBED_MODEL`, import shared constants into `rag` | `pytest tests/ -q` | GREEN — `13 passed` |
| Typecheck | `npx pyright rag.py loda_data.py tests/test_rag.py` | 0 errors |
| End-to-end rerun | `.venv/bin/python rag.py` | 3 chunks printed (pages 31, 32, 32) |

M2 (`test_import_triggers_no_side_effects` asserted only `callable(rag.main)`) needed no
production change — it was a weak test, not a bug. It now reloads `rag` under a patched
`pymongo.MongoClient` and asserts the client is never constructed, plus asserts the patch reaches
`rag`'s namespace so the check cannot pass vacuously.

Known limitation: with `rag` importing `EMBED_MODEL` from `loda_data`, the equality assertion in
test 2 is near-tautological. Its remaining value is catching a re-hardcoded model literal inside
`make_embeddings`. The structural guarantee now comes from the single definition, not the test.

## Coverage and known gaps

No coverage tooling configured (`pyproject.toml` holds pytest config only). All pure logic in
`rag.py` (`make_embeddings`, `format_results`) is covered; `main()` is I/O-only and verified by
the live run above rather than by a test — same convention as `loda_data.main()`.

## Follow-up cycle — review findings L1/L2 (2026-07-29)

| Task | Validation | Result |
|------|-----------|--------|
| L1+L2 reproducers | `pytest tests/test_rag.py -q` | RED — `assert 'no matching' in ''`; `module 'rag' has no attribute 'resolve_query'` (x2) |
| Fix applied | `pytest tests/ -q` | GREEN — `16 passed` |
| Typecheck | `npx pyright rag.py` | 0 errors |
| Default query | `.venv/bin/python rag.py` | 3 chunks (pages 31, 32, 32) |
| CLI query | `.venv/bin/python rag.py "how does sharding work?"` | 3 chunks (page 33, Change Streams) |

| # | Guarantee | Test | Type | Result |
|---|-----------|------|------|--------|
| 4 | Zero hits print an explicit message, not a blank line | `tests/test_rag.py::test_format_results_signals_zero_hits` | unit | PASS |
| 5 | First CLI argument becomes the query | `tests/test_rag.py::test_resolve_query_prefers_cli_argument` | unit | PASS |
| 6 | No argument falls back to `DEFAULT_QUERY` | `tests/test_rag.py::test_resolve_query_falls_back_to_default` | unit | PASS |

`QUERY` was renamed to `DEFAULT_QUERY`; argv parsing is isolated in the pure `resolve_query(argv)`
so it is testable without touching `sys.argv`. Deliberately no `argparse` — one positional
argument does not need it; add it when flags appear.

## Follow-up cycle — review findings M3/M4 (2026-07-29)

| Task | Validation | Result |
|------|-----------|--------|
| M3+M4 reproducers | `pytest tests/test_rag.py -q` | RED — `ModuleNotFoundError: No module named 'config'` (collection-time) |
| M3 isolated reproducer | standalone script, patch+reload without the `finally` | RED — `rag.MongoClient left as MagicMock` |
| Fix applied | `pytest tests/ -q` | GREEN — `17 passed` |
| Typecheck | `npx pyright rag.py loda_data.py config.py tests/` | 0 errors |
| Import cost | `python -X importtime -c "import rag"` | 330 ms, down from 581 ms; `loda_data` and `langchain_community` no longer loaded |
| End-to-end | `.venv/bin/python rag.py "how does sharding work?"` | 3 chunks (page 33) |

| # | Guarantee | Test | Type | Result |
|---|-----------|------|------|--------|
| 7 | The reload test restores the real `MongoClient` for later tests | `tests/test_rag.py::test_import_opens_no_mongo_connection` | unit | PASS |
| 8 | Importing `rag` does not load `langchain_community` | `tests/test_rag.py::test_import_stays_out_of_the_ingest_stack` | unit | PASS |

`DB_NAME`, `COLLECTION_NAME` and `EMBED_MODEL` now live in `config.py`, imported by both
`loda_data.py` and `rag.py`. Test 8 runs the probe in a subprocess because `loda_data` is already
in `sys.modules` within the pytest session, which would mask the regression.
