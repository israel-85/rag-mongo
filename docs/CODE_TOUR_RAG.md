# rag-mongo: Retrieval Walkthrough

Onboarding tour of the query → embed → Atlas vector search → streamed answer path.
**Ref:** `main`

Scope: retrieval only (`rag.py`). The ingestion half (`load_data.py`) is not covered here — see
`docs/CODE_TOUR.md`.

## Step 1 — Start Here
`CLAUDE.md:79`

Situation: this is the retrieval half's map. Mechanism: it documents the 7-stage query pipeline and where the shared config comes in. Implication: read this before touching rag.py, it explains why config.py exists as its own module. Gotcha: it's the source of truth, not code comments — if they disagree, trust this file first, then verify against code.

## Step 2 — One Script, Same Shape as Ingest
`rag.py:1`

Situation: 129-line script, same convention as load_data.py. Mechanism: pure functions at module top level (`make_embeddings`, `resolve_query`, `format_context`, `retriever_config`, `stream_answer`); all I/O — MongoClient, Atlas, the LLM — confined to `main()` (line 75+), guarded by `if __name__ == "__main__"`. Implication: importing this module (as tests do) opens no MongoClient and pulls no PDF/ingest deps into the query path. Gotcha: don't move I/O out of main() or you break the two structural test guards that assert exactly this.

## Step 3 — Resolving the Query
`rag.py:33`

Situation: `resolve_query(argv)` decides what question gets asked. Mechanism: first CLI arg, stripped; if it's absent or blank, `DEFAULT_QUERY`. Implication: `python rag.py "how does sharding work?"` and `python rag.py` both just work, no flag parsing needed — and `python rag.py ""` falls back instead of embedding an empty string. Gotcha: that's the *only* validation. A non-empty argument is trusted as-is, so a query that's pure punctuation still reaches Voyage.

## Step 4 — The Embedding Model Must Match Ingest
`rag.py:28`

Situation: `make_embeddings()` builds the Voyage client used to embed the query. Mechanism: reads `EMBED_MODEL` from `config.py`, the same constant `load_data.py` uses to embed stored chunks. Implication: query and stored vectors must come from the same model or Atlas silently returns nothing/nonsense — no error, just bad results. Gotcha: this is exactly why `config.py` is its own module (see `CLAUDE.md` "Import cost") — importing from `load_data` here would drag `PyPDFLoader` into every query.

## Step 5 — Building the Retriever
`rag.py:78`

Situation: `main()` wires `MongoDBAtlasVectorSearch` against `INDEX_NAME`, then calls `retriever_config()` (`rag.py:52`) to get the `search_type`/`search_kwargs` pair before turning it into a retriever. Mechanism: `retriever_config(relaxed=False)` returns `search_type="similarity_score_threshold"` with `k=TOP_K` (3), a `pre_filter` excluding `hasCode: True` chunks, and `score_threshold: SCORE_THRESHOLD` (0.75, `rag.py:48`). Implication: code-heavy chunks are filtered by `pre_filter` before ranking runs; low-relevance chunks are filtered by `score_threshold` after ranking — two different filters, two different jobs. Gotcha: three here. `INDEX_NAME` must match the actual Atlas Search index name (`"vector_index"`) and that index's `numDimensions` must match `EMBED_MODEL` — see the "Atlas vector index" section of `CLAUDE.md`. `search_type` and `search_kwargs` must agree, or `score_threshold` is silently dropped rather than erroring — that's exactly what shipped for a while, discovered only by inspecting `VectorStoreRetriever`'s dispatch and confirming `similarity_search`'s `**kwargs: Any` swallows unknown keys without complaint. And `0.75` isn't arbitrary: a live probe against the 171-chunk corpus (3 on-topic/adjacent queries, 3 off-topic controls) found on-topic top-1 scores ≥0.79 and off-topic top-1 scores ≤0.70 — 0.75 sits in that gap. `retriever_config(relaxed=True)` is the second, lower gear used by Step 8's retry — see there for why. Re-probe both numbers if the corpus changes materially; they're calibrated to this data, not derived from first principles.

## Step 6 — Retrieval Happens, Maybe Twice
`rag.py:88`

Situation: `docs = retriever.invoke(query)` is the primary retrieval call. Mechanism: the resulting `docs` list is handed to `format_context`; if that comes back empty, Step 8's retry calls `invoke` again on a second retriever built with the relaxed config. Implication: whatever changes retrieval quality (k, pre_filter, score_threshold) affects both calls identically, since both go through `retriever_config()`. Gotcha: the retrieved chunk text is never printed — the console shows only the query, an optional retry notice, and the streamed answer. A `format_results` snippet renderer used to print it and was deleted once `main()` stopped calling it.

## Step 7 — Chunks Into Context
`rag.py:39`

Situation: `format_context(docs)` turns the retrieved `Document` list into the prompt's `context` string. Mechanism: joins `page_content` with a blank line between passages so the LLM doesn't read two chunks as one sentence, and skips any chunk that is entirely whitespace. Implication: an empty return value means exactly one thing — "nothing worth answering from" — which is what the refusal in the next step branches on. Blank chunks can't disguise themselves as context. Gotcha: only *fully* blank chunks are dropped; leading whitespace inside a real chunk is preserved, because indentation can be meaningful. And only `page_content` goes in — the `title`/`keywords`/`hasCode` metadata that ingestion worked to attach is used for *filtering* (Step 5) and never shown to the LLM.

## Step 8 — Retry Once, Then Refuse Instead of Guessing
`rag.py:91`

Situation: when the primary pass's `format_context` comes back empty, `main()` doesn't refuse immediately — it retries once at `retriever_config(relaxed=True)` (`score_threshold: RELAXED_SCORE_THRESHOLD`, 0.71) before giving up. Only if *that* also comes back empty does it print `NO_CONTEXT_MESSAGE` and return, before the prompt or `ChatOpenAI` is built. Mechanism: two sequential `if not context:` checks (`rag.py:91` and `rag.py:98`), the first doing a second `retriever_config()`/`as_retriever()`/`invoke()` round, the second the actual early return. Implication: the refusal is still structural — the prompt's "do not answer without context" line is a backstop a small local model can ignore, not the primary guard — but it's no longer trigger-happy on a single below-threshold pass. `0.71` was chosen deliberately close to the primary `0.75`: it sits just above the highest off-topic score seen in calibration (0.7025, Step 5), so the retry can rescue a genuine near-miss without meaningfully reopening the door to noise. Gotcha: an empty *final* result now has three possible causes — a setup problem (ingestion never run, wrong `INDEX_NAME`, `numDimensions` mismatch), a query correctly rejected by both passes as off-topic, or (rarest) a borderline on-topic query that missed even the relaxed threshold. `NO_CONTEXT_MESSAGE` still points at ingestion/index state, which fits the first cause and is an over-steer for the other two — don't assume every refusal means something is broken. The retry itself prints a one-line notice to stdout, so a relaxed-pass rescue is visible, not silent.

## Step 9 — Prompt Assembly, No RunnablePassthrough
`rag.py:120`

Situation: the LCEL chain is `custom_rag_prompt | llm | StrOutputParser()`. Mechanism: `main()` builds the `{"context": ..., "question": ...}` dict itself at the `.stream()` call site (`rag.py:98`) instead of using `RunnablePassthrough`. Implication: there's nothing to "pass through" — retrieval already happened, so the chain only needs to format and generate. Gotcha: if you see `RunnablePassthrough` missing and assume it's a bug, check `CLAUDE.md` first — it's deliberate, not an oversight.

## Step 10 — Streaming the Answer, Flush Is Load-Bearing
`rag.py:66`

Situation: `stream_answer(chunks, out)` writes each token from `rag_chain.stream()` as it arrives. Mechanism: `out.write(chunk)` then `out.flush()` after every single token, not just at the end. Implication: stdout is block-buffered when piped (not a TTY) — skip the per-token flush and the whole answer lands at once, the stream becomes invisible even though tokens are arriving one at a time. Gotcha: `out` defaults to `sys.stdout` but accepts anything with `write`/`flush` — that's what makes it testable without a real terminal (see next step).

## Step 11 — How It's Tested Without Live Services
`tests/test_rag.py:17`

Situation: `test_import_opens_no_mongo_connection` and `test_import_stays_out_of_the_ingest_stack` are structural guards, not behavior tests — they prove the module-import guarantee from Step 2 actually holds. Mechanism: the first patches `pymongo.MongoClient` and reloads `rag`, asserting it's never called; the second imports `rag` in a subprocess and checks `langchain_community` never lands in `sys.modules`. Implication: this is your template for guarding any future import-time side effect. The behavior tests sit alongside them — `resolve_query`, `format_context`, and `stream_answer` are each covered by plain equality assertions, because all three are pure. Gotcha: the subprocess check exists because `load_data` (which does import `langchain_community`) may already be in `sys.modules` within the same pytest session — a same-process check would be a false negative.

## Step 12 — Next Steps

You've walked the query path end to end: resolve query → embed → retrieve (filtered) → format context → refuse-or-answer → build prompt → stream answer, plus the two guards that keep it side-effect-free at import time. To run it for real: fill in `key_param.py`, start your LM Studio server, then `python rag.py` or `python rag.py "your question"`. To extend it, add a pure function near the top of `rag.py` and a matching test in `tests/test_rag.py` first — same TDD pattern as `load_data.py`.
