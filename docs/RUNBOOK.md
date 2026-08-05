# Runbook

Operational procedures for the ingest and query halves. There is no deployment
target — both are scripts run by hand — so this covers the Atlas/Voyage state they
depend on, and the failures actually observed against it.

## Required Atlas state

Two search indexes on `book_mongodb_chunks.chunked_data`:

| Index | Type | Field | Created by |
|---|---|---|---|
| `vector_index` | `vectorSearch` | `embedding`, 1024 dims, cosine | By hand (see below) |
| `text_index` | `search` | `text` | `load_data.py` automatically |

**Atlas M0 caps the whole cluster at 3 search indexes.** The `sample_mflix` sample
dataset ships with two of its own, which is enough to exhaust the quota on a fresh
free-tier cluster.

### Creating the vector index

```python
collection.update_search_index("vector_index", {"fields": [
    {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
    {"type": "filter", "path": "hasCode"}]})
```

`numDimensions` must match `EMBED_MODEL` (`voyage-3.5-lite` → 1024). A mismatch
fails **silently** — Atlas returns nothing rather than erroring.

### Checking index state

```python
[(i["name"], i.get("status")) for i in collection.list_search_indexes()]
```

Both must read `READY`. Index builds are asynchronous; a query issued against a
building full-text index degrades to vector-only rather than failing.

## Procedures

### Re-ingesting after a source change

```bash
python load_data.py                    # same PDF, same chunking
python load_data.py --fresh            # after changing chunk_size/overlap or the PDF
```

Chunk `_id`s are a deterministic hash of `source|page|content`, so a plain re-run
**upserts** — it rewrites the same documents rather than appending a second copy.
`--fresh` is needed only when chunking settings or the source change, because new
chunks then hash differently and leave the old ones orphaned.

Expect roughly 171 chunks in 4 batches with 25s sleeps between them.

### Verifying retrieval health

```bash
python -m evaluation.run_eval --verbose
```

Current expected numbers (2026-08-05, 171-chunk corpus):

| metric | expected |
|---|---|
| `hit_rate` | 1.000 |
| `mrr` | 1.000 |
| `abstention_rate` | 1.000 |

A drop in `hit_rate` means the corpus or index changed. A drop in
`abstention_rate` means the reranker floor is too low and off-topic queries are
being answered.

### Recalibrating thresholds

Required after changing `EMBED_MODEL`, `RERANK_MODEL`, the corpus, or chunking:

```bash
python -m evaluation.calibrate
```

It prints the top rerank score per question, split answerable vs control, and
reports whether a separating gap exists. Set `RERANK_THRESHOLD` inside that gap.
If the two groups **overlap**, no threshold separates them — the fix is better
retrieval or a better golden set, not a number nudged until it looks right.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `key_param.py is missing or has blank values for: ...` | Unset secret | Fill in the named field; all five are required by both scripts |
| `The maximum number of FTS indexes has been reached for this instance size.` | M0's 3-index cap | Drop an unused index (`sample_mflix` ones are the usual candidates) or upgrade the tier |
| `SSL handshake failed: ... TLSV1_ALERT_INTERNAL_ERROR` | Cluster paused, or IP not allowlisted | Resume the cluster in Atlas; check Network Access |
| `ServerSelectionTimeoutError` | Wrong `MONGODB_URI`, or no network | Verify the URI; check Atlas IP allowlist |
| `voyageai.error.RateLimitError` | Free tier: 3 req/min, 10K tokens/min | Expected. Ingest and eval both retry with backoff; wait it out or add a payment method |
| `note: full-text index 'text_index' not found` | Index never created or still building | Re-run `load_data.py`; check `list_search_indexes()` for `READY` |
| Query returns `NO_CONTEXT_MESSAGE` unexpectedly | Ingestion not run, wrong index name, dimension mismatch, or a genuinely off-topic query | Check collection count and index status first; the message over-steers toward setup problems |
| Answers cite pages that don't support them | Threshold too low, or stale chunks from an old ingest | Run the eval; if `abstention_rate` dropped, recalibrate. If `hit_rate` dropped, `--fresh` re-ingest |
| Duplicate-looking results crowding out others | Pre-`chunk_id` duplicate chunks from an old double-ingest | `python load_data.py --fresh` |

## Rollback

Ingest is idempotent, so there is no partial-write state to repair — re-running
converges on the same documents. To return to a clean corpus:

```bash
python load_data.py --fresh
```

To revert a retrieval change, revert the code and re-run the eval to confirm the
numbers return to the table above. The eval is the rollback verification.

## External dependencies at query time

Three, all of which can fail independently:

1. **MongoDB Atlas** — vector and full-text search.
2. **Voyage AI** — query embedding *and* reranking (two requests per query).
3. **HuggingFace Hub** — Voyage's client fetches a tokenizer, unauthenticated.
   Setting `HF_TOKEN` raises the rate limit. This one is easy to miss because it
   only surfaces as a warning until it doesn't.

Plus the LLM server (LM Studio) for generation, which is local by default.
