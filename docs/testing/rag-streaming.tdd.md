# TDD Evidence: streamed answer output in `rag.py`

**Source plan** — inline plan from `/ecc:plan "Add streaming, perhaps RunnablePassthrough is
required"` (conversational mode, no `*.plan.md` artifact written).
**Runner** — pytest (`.venv/bin/python -m pytest`). Not npm; the placeholder commands in the
tdd-workflow skill were translated to this project's runner before use.
**Branch** — `main`. Checkpoints `fa3cc84` (RED) and `621d524` (GREEN) are reachable from `HEAD`.

## User journey

> As someone asking the RAG app a question, I want the answer to appear token by token as the LLM
> produces it, so that I see progress instead of staring at a blank terminal until the whole
> response is finished.

## Scope decision: `RunnablePassthrough` is not required

The plan asked whether `RunnablePassthrough` was needed. It is not, and this is worth recording
because the two concerns look related but are not:

- `RunnablePassthrough` is LCEL plumbing for chains invoked with a **bare string**, where the
  prompt's input dict has to be assembled inside the chain. `main()` builds that dict explicitly,
  so a passthrough would be an unused indirection.
- Streaming is `.stream()` instead of `.invoke()`. It works on any Runnable regardless of whether
  the input is a bare value or a dict.

No passthrough was added. If the chain is ever exposed outside `main()` as a reusable object taking
a raw query string, that is the point at which the import earns its place.

## Task report

### Task 1 — stream the answer instead of printing it in one block

Replaced the buffered `rag_chain.invoke(...)` + `print(f"...{answer}")` with a new
`stream_answer(chunks, out)` helper driven by `rag_chain.stream(...)`.

RED — `.venv/bin/python -m pytest tests/test_rag.py -q`:

```
E       AttributeError: module 'rag' has no attribute 'stream_answer'
FAILED tests/test_rag.py::test_stream_answer_writes_each_token_separately
FAILED tests/test_rag.py::test_stream_answer_flushes_after_every_token
FAILED tests/test_rag.py::test_stream_answer_terminates_the_line
FAILED tests/test_rag.py::test_stream_answer_handles_empty_stream
4 failed, 7 passed, 1 warning in 2.31s
```

The failure is the intended one: the function did not exist yet. No unrelated collection or import
errors were involved.

GREEN — `.venv/bin/python -m pytest tests/ -q`:

```
21 passed, 1 warning in 1.58s
```

Guaranteed by the passing tests: tokens are written individually, `flush()` is called after every
token, output ends with a newline, and an empty stream does not raise.

### Task 2 — no test for `main()`

`main()` remains I/O-only, per the convention documented in CLAUDE.md. Verified by running the real
script against Atlas and LM Studio — `.venv/bin/python rag.py "What is a replica set?"`:

```
Answer:
A replica set consists of ideally three or more servers that hold the same data.
```

## Test specification

| # | What is guaranteed | Test | Type | Result | Evidence |
|---|--------------------|------|------|--------|----------|
| 1 | Each token is written on its own `write()` call, not accumulated into one string | `tests/test_rag.py:test_stream_answer_writes_each_token_separately` | unit | PASS | `pytest tests/test_rag.py -q` |
| 2 | `flush()` follows every token, so piped/block-buffered stdout still streams | `tests/test_rag.py:test_stream_answer_flushes_after_every_token` | unit | PASS | `pytest tests/test_rag.py -q` |
| 3 | Output ends with a newline, since the streamed answer has none of its own | `tests/test_rag.py:test_stream_answer_terminates_the_line` | unit | PASS | `pytest tests/test_rag.py -q` |
| 4 | An LLM returning zero chunks does not crash the run | `tests/test_rag.py:test_stream_answer_handles_empty_stream` | unit | PASS | `pytest tests/test_rag.py -q` |
| 5 | Pre-existing guarantees still hold (query resolution, snippet rendering, import-time isolation) | `tests/test_rag.py` (7 tests), `tests/test_load_data.py` (10 tests) | unit | PASS | `pytest tests/ -q` → 21 passed |

The write/flush ordering in #2 is asserted through a `RecordingStream` test double that appends
`("write", text)` and `("flush", "")` tuples to a list. Asserting on the *sequence* — not just the
final text — is what makes "does not buffer" a real guarantee rather than an assumption.

## Coverage and known gaps

`pytest-cov` was added to `.venv` for this step and recorded in CLAUDE.md's dependency list.

`.venv/bin/python -m pytest tests/ -q --cov=rag --cov-report=term-missing`:

```
Name     Stmts   Miss  Cover   Missing
--------------------------------------
rag.py      48     15    69%   55-93, 97
```

**69% is below the 80% bar, and the shortfall is entirely by design.** The 15 uncovered statements
are `main()` (lines 55-93) plus the `main()` call under `if __name__` (line 97) — the module's
I/O-only section, which CLAUDE.md explicitly excludes from unit tests because covering it would
require live Atlas, Voyage, and LM Studio calls. Excluding that block, the testable surface is
33/33 statements = **100%**.

No `# pragma: no cover` was added to make the headline number look better. `load_data.py` has the
same top-level-pure / `main()`-does-I/O shape and no pragma, and annotating only `rag.py` would make
the two files inconsistent for a purely cosmetic gain.

Known gaps, stated rather than hidden:

- Whether LM Studio actually emits incremental chunks is server-side behavior. The client-side
  contract (write + flush per chunk) is tested; end-to-end token-by-token arrival was confirmed by
  eye during the live run above and is not covered by an automated test.
- `main()` is unverified by tests, as above.
- `uvx ruff check rag.py` reports a pre-existing `I001` (unsorted import block) that predates this
  change. Left alone as out of scope; fix with `uvx ruff check --fix` if the import order is being
  tidied deliberately.

## Merge evidence

If these checkpoints are squashed, preserve:

- RED `fa3cc84` — `test: add reproducer for streamed answer output` — 4 failed / 7 passed,
  `AttributeError: module 'rag' has no attribute 'stream_answer'`.
- GREEN `621d524` — `feat: stream answer tokens as the LLM produces them` — 21 passed.
- Refactor — none needed; the implementation is a five-line loop and was clean at GREEN.
