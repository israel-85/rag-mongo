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
| 1 | Importing `rag` performs no I/O; `main()` exists | `tests/test_rag.py::test_import_triggers_no_side_effects` | unit | PASS |
| 2 | Query embedding model + namespace match ingestion | `tests/test_rag.py::test_embeddings_match_ingest_model` | unit | PASS |
| 3 | Results render as truncated page-numbered snippets | `tests/test_rag.py::test_format_results_prints_source_and_snippet` | unit | PASS |

## Coverage and known gaps

No coverage tooling configured (`pyproject.toml` holds pytest config only). All pure logic in
`rag.py` (`make_embeddings`, `format_results`) is covered; `main()` is I/O-only and verified by
the live run above rather than by a test — same convention as `loda_data.main()`.
