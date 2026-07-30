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

Situation: 105-line script, same convention as load_data.py. Mechanism: pure functions at module top level (`make_embeddings`, `resolve_query`, `format_context`, `stream_answer`); all I/O — MongoClient, Atlas, the LLM — confined to `main()` (line 57+), guarded by `if __name__ == "__main__"`. Implication: importing this module (as tests do) opens no MongoClient and pulls no PDF/ingest deps into the query path. Gotcha: don't move I/O out of main() or you break the two structural test guards that assert exactly this.

## Step 3 — Resolving the Query
`rag.py:33`

Situation: `resolve_query(argv)` decides what question gets asked. Mechanism: first CLI arg, stripped; if it's absent or blank, `DEFAULT_QUERY`. Implication: `python rag.py "how does sharding work?"` and `python rag.py` both just work, no flag parsing needed — and `python rag.py ""` falls back instead of embedding an empty string. Gotcha: that's the *only* validation. A non-empty argument is trusted as-is, so a query that's pure punctuation still reaches Voyage.

## Step 4 — The Embedding Model Must Match Ingest
`rag.py:28`

Situation: `make_embeddings()` builds the Voyage client used to embed the query. Mechanism: reads `EMBED_MODEL` from `config.py`, the same constant `load_data.py` uses to embed stored chunks. Implication: query and stored vectors must come from the same model or Atlas silently returns nothing/nonsense — no error, just bad results. Gotcha: this is exactly why `config.py` is its own module (see `CLAUDE.md` "Import cost") — importing from `load_data` here would drag `PyPDFLoader` into every query.

## Step 5 — Building the Retriever
`rag.py:60`

Situation: `main()` wires `MongoDBAtlasVectorSearch` against `INDEX_NAME` and turns it into a retriever. Mechanism: `search_kwargs` sets `k=TOP_K` (3), a `pre_filter` excluding `hasCode: True` chunks, and a `score_threshold` of 0.01. Implication: code-heavy chunks are filtered out before similarity ranking even runs — this is a hard filter, not a ranking signal. Gotcha: two of them here. `INDEX_NAME` must match the actual Atlas Search index name (`"vector_index"`) and that index's `numDimensions` must match `EMBED_MODEL` — see the "Atlas vector index" section of `CLAUDE.md`. And the `score_threshold` does nothing: LangChain only applies it when `search_type="similarity_score_threshold"`, but this call passes `search_type="similarity"`, so weak matches are *not* being dropped today.

## Step 6 — Retrieval Happens Once
`rag.py:72`

Situation: `docs = retriever.invoke(query)` is the single retrieval call for the whole run. Mechanism: the resulting `docs` list is handed to exactly one thing below — `format_context`. Implication: whatever changes retrieval quality (k, pre_filter, score_threshold) affects exactly one code path, easy to reason about. Gotcha: the retrieved chunk text is never printed — the console shows only the query and the streamed answer. A `format_results` snippet renderer used to print it and was deleted once `main()` stopped calling it.

## Step 7 — Chunks Into Context
`rag.py:39`

Situation: `format_context(docs)` turns the retrieved `Document` list into the prompt's `context` string. Mechanism: joins `page_content` with a blank line between passages so the LLM doesn't read two chunks as one sentence, and skips any chunk that is entirely whitespace. Implication: an empty return value means exactly one thing — "nothing worth answering from" — which is what the refusal in the next step branches on. Blank chunks can't disguise themselves as context. Gotcha: only *fully* blank chunks are dropped; leading whitespace inside a real chunk is preserved, because indentation can be meaningful. And only `page_content` goes in — the `title`/`keywords`/`hasCode` metadata that ingestion worked to attach is used for *filtering* (Step 5) and never shown to the LLM.

## Step 8 — Refusing Instead of Guessing
`rag.py:74`

Situation: when `format_context` comes back empty, `main()` prints `NO_CONTEXT_MESSAGE` and returns — before the prompt or `ChatOpenAI` is built. Mechanism: a plain `if not context:` early return. Implication: the refusal is structural. The prompt does also say "do not answer the question if there is no given context", but that's an instruction a small local model is free to ignore and answer from memory instead; this guard removes the opportunity entirely. Gotcha: an empty result is far more often a *setup* problem than a genuine miss — ingestion never run, wrong `INDEX_NAME`, or an Atlas index whose `numDimensions` disagrees with `EMBED_MODEL`. That's why the message points at ingestion and index state rather than just saying "no results".

## Step 9 — Prompt Assembly, No RunnablePassthrough
`rag.py:96`

Situation: the LCEL chain is `custom_rag_prompt | llm | StrOutputParser()`. Mechanism: `main()` builds the `{"context": ..., "question": ...}` dict itself at the `.stream()` call site (`rag.py:98`) instead of using `RunnablePassthrough`. Implication: there's nothing to "pass through" — retrieval already happened, so the chain only needs to format and generate. Gotcha: if you see `RunnablePassthrough` missing and assume it's a bug, check `CLAUDE.md` first — it's deliberate, not an oversight.

## Step 10 — Streaming the Answer, Flush Is Load-Bearing
`rag.py:48`

Situation: `stream_answer(chunks, out)` writes each token from `rag_chain.stream()` as it arrives. Mechanism: `out.write(chunk)` then `out.flush()` after every single token, not just at the end. Implication: stdout is block-buffered when piped (not a TTY) — skip the per-token flush and the whole answer lands at once, the stream becomes invisible even though tokens are arriving one at a time. Gotcha: `out` defaults to `sys.stdout` but accepts anything with `write`/`flush` — that's what makes it testable without a real terminal (see next step).

## Step 11 — How It's Tested Without Live Services
`tests/test_rag.py:17`

Situation: `test_import_opens_no_mongo_connection` and `test_import_stays_out_of_the_ingest_stack` are structural guards, not behavior tests — they prove the module-import guarantee from Step 2 actually holds. Mechanism: the first patches `pymongo.MongoClient` and reloads `rag`, asserting it's never called; the second imports `rag` in a subprocess and checks `langchain_community` never lands in `sys.modules`. Implication: this is your template for guarding any future import-time side effect. The behavior tests sit alongside them — `resolve_query`, `format_context`, and `stream_answer` are each covered by plain equality assertions, because all three are pure. Gotcha: the subprocess check exists because `load_data` (which does import `langchain_community`) may already be in `sys.modules` within the same pytest session — a same-process check would be a false negative.

## Step 12 — Next Steps

You've walked the query path end to end: resolve query → embed → retrieve (filtered) → format context → refuse-or-answer → build prompt → stream answer, plus the two guards that keep it side-effect-free at import time. To run it for real: fill in `key_param.py`, start your LM Studio server, then `python rag.py` or `python rag.py "your question"`. To extend it, add a pure function near the top of `rag.py` and a matching test in `tests/test_rag.py` first — same TDD pattern as `load_data.py`.
