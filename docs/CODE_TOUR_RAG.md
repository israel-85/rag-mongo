# rag-mongo: Retrieval Walkthrough

Onboarding tour of the query → embed → Atlas vector search → streamed answer path.
**Ref:** `main`

Scope: retrieval only (`rag.py`). The ingestion half (`load_data.py`) is not covered here — see
`docs/CODE_TOUR.md`.

## Step 1 — Start Here
`CLAUDE.md:119`

Situation: this is the retrieval half's map. Mechanism: it documents the query pipeline (hybrid candidates -> rerank -> cite -> refuse-or-answer), where the shared config comes in, and the measured before/after numbers for every retrieval change. Implication: read this before touching rag.py, it explains why config.py exists as its own module. Gotcha: it's the source of truth, not code comments — if they disagree, trust this file first, then verify against code.

## Step 2 — One Script, Same Shape as Ingest
`rag.py:1`

Situation: ~300-line script, same convention as load_data.py. Mechanism: pure and near-pure functions at module top level (`make_embeddings`, `resolve_query`, `citation_label`, `format_context`, `usable_chunks`, `format_sources`, `retriever_config`, `build_candidate_retriever`, `score_candidates`, `keep_above`, `build_vector_store`, `retrieve`, `stream_answer`); all I/O — MongoClient, Atlas, the LLM — confined to `main()` (`rag.py:263`), guarded by `if __name__ == "__main__"`. Implication: importing this module (as tests do) opens no MongoClient and pulls no PDF/ingest deps into the query path. Gotcha: don't move I/O out of main() or you break the two structural test guards that assert exactly this. Note `retrieve()` and `build_candidate_retriever()` do talk to Atlas — they take the store as an argument rather than building one, which is what keeps them testable and lets `evaluation/run_eval.py` score the exact code path that runs in production.

## Step 3 — Resolving the Query
`rag.py:48`

Situation: `resolve_query(argv)` decides what question gets asked. Mechanism: first CLI arg, stripped; if it's absent or blank, `DEFAULT_QUERY`. Implication: `python rag.py "how does sharding work?"` and `python rag.py` both just work, no flag parsing needed — and `python rag.py ""` falls back instead of embedding an empty string. Gotcha: that's the *only* validation. A non-empty argument is trusted as-is, so a query that's pure punctuation still reaches Voyage.

## Step 4 — The Embedding Model Must Match Ingest
`rag.py:43`

Situation: `make_embeddings()` builds the Voyage client used to embed the query. Mechanism: reads `EMBED_MODEL` from `config.py`, the same constant `load_data.py` uses to embed stored chunks. Implication: query and stored vectors must come from the same model or Atlas silently returns nothing/nonsense — no error, just bad results. Gotcha: this is exactly why `config.py` is its own module (see `CLAUDE.md` "Import cost") — importing from `load_data` here would drag `PyPDFLoader` into every query.

## Step 5 — Candidate Generation: Wide Net, Hybrid
`rag.py:117`

Situation: `main()` wires `MongoDBAtlasVectorSearch` against `INDEX_NAME`, then `build_candidate_retriever()` (`rag.py:131`) produces the retriever that generates candidates. Mechanism: when `FULLTEXT_INDEX_NAME` exists, it returns a `MongoDBAtlasHybridSearchRetriever` running vector search and full-text search over the same query and fusing the two ranked lists by Reciprocal Rank Fusion; otherwise it prints a note and falls back to `retriever_config()`'s plain `similarity` retriever. Either way it asks for `CANDIDATE_K` (20), not `TOP_K` (3). Implication: this stage owns *recall* only — precision is Step 6's job. Retrieving 20 to keep 3 is the entire point, so there is deliberately no `score_threshold` here: gating at this stage would cap what the reranker can ever see.

Gotcha: three worth knowing. First, the old `pre_filter` of `{"hasCode": {"$eq": False}}` is **gone** — it made every code-bearing page permanently unreachable, and the measured cost was severe (see Step 12). "How do I create an index" is answered by page 24, which that filter hid. Second, RRF needs no calibrated score from either side, which is why hybrid survives an embedding-model swap that would invalidate any tuned similarity cutoff — embeddings blur exact tokens like `$unwind` or `4.0`, and lexical search is precisely the half that catches them. Third, the full-text index check is cached per namespace (`rag.py:163`): index presence cannot change mid-run, and asking per query cost an extra Atlas round-trip and reprinted the same warning every time.

## Step 6 — Reranking: The Gate Moved Here
`rag.py:194`

Situation: `score_candidates()` scores the 20 candidates in one API call and `keep_above()` drops anything under the floor. Mechanism: it sends the query and the candidates' `page_content` to Voyage `rerank-2.5` — a cross-encoder that reads each query/document *pair* together — then keeps the results scoring at or above `threshold`, in the reranker's order. Implication: relevance is now judged by a model that reads both sides, not by cosine distance between two independently-computed summaries of meaning. That is what separates "mentions indexes" from "explains creating an index". It returns `Document`s rather than scores precisely so chunk metadata survives for Step 7's citations.

Gotcha: `RERANK_THRESHOLD` (0.55) and `RELAXED_RERANK_THRESHOLD` (0.45) are calibrated, not guessed. `python -m evaluation.calibrate` on 2026-08-05 measured the top rerank score for all 22 golden questions: the highest off-topic control scored `0.4023`, the lowest answerable question `0.6875` — a clean gap with no overlap, and `0.55` sits inside it with room on both sides. The old `0.75`/`0.71` numbers do **not** carry over; they scored vector similarity, a different quantity produced by a different stage. The first pass at this shipped with a guessed `0.5`, which scored `hit_rate 1.000` and `abstention_rate 0.250` — three of four controls leaked. That is the failure mode this floor exists to prevent, and it was invisible until measured. Also: an empty candidate list short-circuits before the API call, because spending a request to discover there is nothing to score is pure waste.

## Step 7 — Chunks Into Context
`rag.py:65`

Situation: `format_context(docs)` turns the retrieved `Document` list into the prompt's `context` string. Mechanism: numbers each surviving chunk and prefixes it with `citation_label(doc)` — `mongodb.pdf p.24`, 1-based because PyPDF's `page` metadata is 0-indexed and a citation the reader cannot check against the actual book is worthless. Chunks are separated by a blank line so the LLM doesn't read two as one sentence, and any entirely-whitespace chunk is skipped. Implication: an empty return value means exactly one thing — "nothing worth answering from" — which is what the refusal in the next step branches on. The prompt (Step 9) instructs the model to cite `[1]`/`[2]` inline, and `main()` prints the matching key under the answer via `format_sources`.

Gotcha: `format_context` and `format_sources` both route through `usable_chunks()` (`rag.py:82`) on purpose. If they filtered independently they could drift, and then `[2]` in the answer would cite a different page than `[2]` on screen — a citation that is confidently wrong is worse than no citation. Only *fully* blank chunks are dropped; leading whitespace inside a real chunk is preserved, because indentation can be meaningful. And only `source`/`page` ride along: the `title`/`keywords`/`hasCode` tags ingestion worked to attach stay out of the prompt, because they are retrieval metadata and would read as content. (`hasCode` no longer filters anything either — see Step 5.)

## Step 8 — Retry Once, Then Refuse Instead of Guessing
`rag.py:232`

Situation: when the first rerank pass leaves nothing above `RERANK_THRESHOLD`, `retrieve()` does not refuse immediately — it re-scores at `RELAXED_RERANK_THRESHOLD` before giving up. Only if *that* is also empty does `main()` print `NO_CONTEXT_MESSAGE` (`rag.py:276`) and return, before the prompt or `ChatOpenAI` is built. Mechanism: a two-iteration loop over the two thresholds, with the `format_context` emptiness check as the exit condition. Implication: the refusal stays structural — the prompt's "do not answer without context" line is a backstop a small local model can ignore, not the primary guard.

Gotcha: the relaxed pass **re-scores the candidates already in hand** rather than querying Atlas again. That is a deliberate change from the previous design, which issued a second vector query. The second chance is about the gate being too strict, never about the net being too small — the net is already 20 wide. One Atlas round-trip per query, not two.

An empty *final* result still has three possible causes — a setup problem (ingestion never run, wrong `INDEX_NAME`, `numDimensions` mismatch), a query correctly rejected as off-topic, or a borderline on-topic query that missed even the relaxed floor. `NO_CONTEXT_MESSAGE` points at ingestion/index state, which fits the first and over-steers for the other two; don't assume every refusal means something is broken.

## Step 9 — Prompt Assembly, No RunnablePassthrough
`rag.py:300`

Situation: the LCEL chain is `custom_rag_prompt | llm | StrOutputParser()`. Mechanism: `main()` builds the `{"context": ..., "question": ...}` dict itself at the `.stream()` call site (`rag.py:302`) instead of using `RunnablePassthrough`. Implication: there's nothing to "pass through" — retrieval already happened, so the chain only needs to format and generate. Gotcha: if you see `RunnablePassthrough` missing and assume it's a bug, check `CLAUDE.md` first — it's deliberate, not an oversight.

## Step 10 — Streaming the Answer, Flush Is Load-Bearing
`rag.py:254`

Situation: `stream_answer(chunks, out)` writes each token from `rag_chain.stream()` as it arrives. Mechanism: `out.write(chunk)` then `out.flush()` after every single token, not just at the end. Implication: stdout is block-buffered when piped (not a TTY) — skip the per-token flush and the whole answer lands at once, the stream becomes invisible even though tokens are arriving one at a time. Gotcha: `out` defaults to `sys.stdout` but accepts anything with `write`/`flush` — that's what makes it testable without a real terminal (see next step).

## Step 11 — How It's Tested Without Live Services
`tests/test_rag.py:17`

Situation: `test_import_opens_no_mongo_connection` and `test_import_stays_out_of_the_ingest_stack` are structural guards, not behavior tests — they prove the module-import guarantee from Step 2 actually holds. Mechanism: the first patches `pymongo.MongoClient` and reloads `rag`, asserting it's never called; the second imports `rag` in a subprocess and checks `langchain_community` never lands in `sys.modules`. Implication: this is your template for guarding any future import-time side effect. The behavior tests sit alongside them — `resolve_query`, `format_context`, and `stream_answer` are each covered by plain equality assertions, because all three are pure. Gotcha: the subprocess check exists because `load_data` (which does import `langchain_community`) may already be in `sys.modules` within the same pytest session — a same-process check would be a false negative.

## Step 12 — Next Steps

You've walked the query path end to end: resolve query → hybrid candidates (20) → rerank to 3 → format context with citations → refuse-or-answer → build prompt → stream answer + sources, plus the two guards that keep it side-effect-free at import time. To run it for real: fill in `key_param.py`, start your LM Studio server, then `python rag.py` or `python rag.py "your question"`.

**Before changing anything in the retrieval path, measure it.** Retrieval fails silently — a wrong chunk yields a fluent, confident, wrong answer with no exception and no failing test. `python -m evaluation.run_eval --verbose` scores 22 golden questions and reports three numbers. Watch `hit_rate` and `abstention_rate` together, because they trade against each other and either one alone will flatter a regression:

| | hit_rate | mrr | abstention_rate |
|---|---|---|---|
| vector threshold 0.75/0.71 + `hasCode` filter, k=3 | 0.389 | 0.389 | 1.000 |
| hybrid + rerank, uncalibrated floor (0.5) | 1.000 | 1.000 | 0.250 |
| hybrid + rerank, calibrated floor (0.55) | 1.000 | 1.000 | 1.000 |

The first row is what this tour used to describe: perfect abstention bought by refusing 61% of questions the corpus could actually answer, with the `hasCode` filter hiding whole pages. The second row is the trap — a tripled hit rate that also leaked three of four off-topic controls.

To extend it, add a pure function near the top of `rag.py` and a matching test in `tests/test_rag.py` first — same TDD pattern as `load_data.py`.
