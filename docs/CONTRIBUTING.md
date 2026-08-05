# Contributing

## Prerequisites

| Requirement | Why |
|---|---|
| Python 3.13 | `pyrightconfig.json` pins this version |
| MongoDB Atlas cluster | Vector + full-text search indexes (see `RUNBOOK.md`) |
| Voyage AI API key | Embeddings and reranking |
| LM Studio (or any OpenAI-compatible server) | Page tagging on ingest, answer generation on query |
| `uv` | Dev tooling runs ephemerally via `uvx` |

## Setup

```bash
cp key_param.example.py key_param.py   # fill in all five values - gitignored
```

There is no `requirements.txt`. Dependencies live only in `.venv`, and
`pyproject.toml` holds pytest config, not packaging metadata. Use the existing
`.venv` rather than recreating it.

<!-- AUTO-GENERATED: commands -->

## Commands

| Command | Description |
|---|---|
| `python load_data.py` | Run the ingestion pipeline against the sample PDF |
| `python load_data.py <path.pdf>` | Ingest a different source document |
| `python load_data.py --fresh` | Delete existing chunks before loading |
| `python rag.py` | Query with the built-in demo question |
| `python rag.py "your question"` | Query with your own question |
| `python -m evaluation.run_eval` | Score retrieval against the golden set |
| `python -m evaluation.run_eval --verbose` | Same, with a line per question |
| `python -m evaluation.run_eval --json out.json` | Same, written machine-readably for diffing runs |
| `python -m evaluation.calibrate` | Print the rerank score distribution, for setting thresholds |
| `pytest tests/ -v` | Run unit tests (no external services needed) |
| `pytest tests/ --cov=rag --cov=load_data --cov-report=term-missing` | Tests with coverage |

<!-- END AUTO-GENERATED -->

The eval and calibrate commands need a live Atlas cluster and Voyage key. The
test suite does not — everything external is mocked.

## Quality gates

Run all three after any Python change. They must be clean before you commit:

```bash
uvx ruff check .        # add --fix for the auto-fixable ones
uvx pyright
pytest tests/
```

Use `uvx`, not `npx` — `npx pyright` resolves to a different package on at least
one dev machine and fails with `Unknown command: "pyright"`.

Two ruff findings and four pyright findings are pre-existing on `main`. Judge your
change by whether it *adds* to those counts, not by whether the output is empty.

## Testing

Write the failing test first, watch it fail, then implement. Both scripts follow
the same shape, and it is load-bearing:

- **Pure functions at module top level.** Anything worth testing goes here.
- **All I/O inside `main()`**, guarded by `if __name__ == "__main__"`.

Two structural guards in `tests/test_rag.py` enforce this: importing `rag` must
construct no `MongoClient`, and must not pull in `langchain_community` (checked in
a subprocess, since the ingest module is already in `sys.modules` during a pytest
run). Moving I/O out of `main()` breaks them.

Tests use Arrange-Act-Assert and descriptive names stating the behaviour
(`test_returns_empty_array_when_no_markets_match_query`), not the function name.
Every test carries a docstring saying *why* the behaviour matters — several
guard failure modes that are silent in production.

### Changing retrieval

**Measure before and after.** Retrieval fails silently: a wrong chunk yields a
fluent, confident, wrong answer with no exception and no red test.

```bash
python -m evaluation.run_eval --verbose --json before.json
# ... make the change ...
python -m evaluation.run_eval --verbose --json after.json
```

Watch `hit_rate` and `abstention_rate` **together**. They trade against each
other, and either one alone will flatter a regression — a change that tripled
`hit_rate` in this repo also leaked three of four off-topic controls, and only
the second number caught it.

If you change a threshold, `EMBED_MODEL`, or `RERANK_MODEL`, re-run
`python -m evaluation.calibrate` — the thresholds are calibrated to a measured
score gap, not chosen by feel.

Note the free Voyage tier allows 3 requests/minute and each question costs two,
so a full eval run takes roughly fifteen minutes.

## Code style

- Immutable by default — return new objects, don't mutate arguments.
- Functions under 50 lines, files under 800.
- Early returns over nesting beyond 4 levels.
- Named constants, not magic numbers.
- Handle errors explicitly; never swallow silently. Where a broad `except` is
  deliberate (LLM tagging, index listing), it prints and carries a comment
  saying why.

## Commits

```
<type>: <description>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.

## PR checklist

- [ ] `uvx ruff check .` adds no new findings
- [ ] `uvx pyright` adds no new findings
- [ ] `pytest tests/` passes
- [ ] New behaviour has a test that fails without the change
- [ ] Retrieval changes include before/after eval numbers in the description
- [ ] `CLAUDE.md` updated if architecture or config changed
- [ ] Tour line anchors re-verified if `rag.py` or `load_data.py` line numbers shifted
- [ ] No secrets committed (`key_param.py` stays gitignored)
