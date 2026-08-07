# CLAUDE.md

Two-script RAG demo over MongoDB Atlas Vector Search:

- **Ingest** (`load_data.py`): PDF → cleaned/filtered pages → LLM metadata tagging → chunking →
  Voyage embeddings → Atlas.
- **Retrieve** (`rag.py`): query → hybrid search → Voyage rerank → LLM answer with citations.

`config.py` holds the settings both halves must agree on.

## Running

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

## Gotchas

- **No `requirements.txt`.** Deps live only in `.venv`; `pyproject.toml` holds pytest config, not
  packaging metadata. Don't add a package expecting a manifest to install it.
- **Dev tooling runs ephemerally** so the `.venv` stays as-is: `uvx ruff check`, `uvx vulture`,
  `uvx pyright` (config in `pyrightconfig.json`). Use `uvx`, **not** `npx` — `npx pyright` resolves
  to something else on at least one dev machine and fails with `Unknown command: "pyright"`.
- **Both halves need a running LM Studio** (or other OpenAI-compatible server) at `LLM_BASE_URL`
  serving `LLM_MODEL` — ingestion to tag pages, retrieval to generate. Only
  `evaluation/run_eval.py` skips it.
- **Query and stored vectors must come from the same `EMBED_MODEL`.** A mismatch is silent: Atlas
  returns nothing or nonsense rather than erroring. Switching it also means updating the Atlas
  vector index `numDimensions`, which fails silently too.
- **Retrieval fails silently.** A wrong chunk yields a fluent, confident, wrong answer with no
  exception and no red test. Run `python -m evaluation.run_eval` **before and after** any
  retrieval change (threshold, reranker, filter, hybrid). Unit tests will not catch it.
- **Atlas M0 caps the cluster at 3 search indexes total.** Index creation failing with
  `The maximum number of FTS indexes has been reached for this instance size.` means the cap, not
  a bug.
- Tests mock all LLM/Mongo/Voyage calls (`tests/conftest.py`) — no network. `main()` in both
  scripts is I/O-only and verified by running the scripts, not by tests.

## Secrets

`key_param.py` (gitignored; template in `key_param.example.py`) holds `MONGODB_URI`,
`VOYAGE_API_KEY`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`. All five are required by **both**
halves and validated by `config.require_secrets(key_param)` at the top of each `main()`, so a
missing or blank value fails in one readable line before any work or API spend.

**Never commit this file.** Non-secret settings belong in `config.py`.

## Where the detail lives

Loaded automatically when you open a matching file — read directly if you need them sooner:

| File | Covers | Loads when touching |
|---|---|---|
| `.claude/rules/config.md` | shared settings, `require_secrets`, why config is its own module | `config.py` |
| `.claude/rules/ingestion.md` | Atlas index setup, the 9-step ingest pipeline, chunk ids, backoff | `load_data.py` |
| `.claude/rules/retrieval.md` | hybrid + rerank design, thresholds, retry policy, prompt hardening | `rag.py` |
| `.claude/rules/evaluation.md` | golden set, metrics, calibration, measured before/after numbers | `evaluation/**` |

Prose docs for humans: `docs/ONBOARDING.md`, `docs/CODE_TOUR.md`, `docs/CODE_TOUR_RAG.md`,
`docs/RUNBOOK.md` (Atlas state, re-ingest, troubleshooting, rollback), `docs/CONTRIBUTING.md`
(setup, quality gates, PR checklist).
