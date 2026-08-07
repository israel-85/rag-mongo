import sys
from collections.abc import Iterable, Sequence
from typing import Any, TextIO

import pymongo.errors
import voyageai
import voyageai.error
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr
from pymongo import MongoClient
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import key_param
from config import (
    COLLECTION_NAME,
    DB_NAME,
    EMBED_MODEL,
    FULLTEXT_INDEX_NAME,
    TEXT_KEY,
    VECTOR_INDEX_NAME,
    require_secrets,
)

INDEX_NAME = VECTOR_INDEX_NAME

# Chunks that reach the prompt.
TOP_K = 3
# Chunks pulled from Atlas for the reranker to choose among. Wide net, narrow gate:
# recall is cheap here (one vector query), precision is what the reranker is for.
CANDIDATE_K = 20

DEFAULT_QUERY = "When did MongoDB begin supporting multi-document transactions?"

NO_CONTEXT_MESSAGE = (
    "No relevant context found in the collection - not answering.\n"
    "If this is unexpected, check that ingestion has run and that the Atlas index is READY."
)


def make_embeddings() -> VoyageAIEmbeddings:
    """Same model as ingestion - query and stored vectors must share dimensions."""
    return VoyageAIEmbeddings(api_key=SecretStr(key_param.VOYAGE_API_KEY), model=EMBED_MODEL)


def resolve_query(argv: list[str]) -> str:
    """First CLI argument is the query; blank or absent falls back to the demo question."""
    query = argv[1].strip() if len(argv) > 1 else ""
    return query or DEFAULT_QUERY


def citation_label(doc: Document) -> str:
    """Human-readable origin of a chunk: file name plus 1-based page number.

    Metadata pages are 0-indexed (PyPDFLoader), printed pages are not - the +1 is
    what makes a citation checkable against the actual book.
    """
    source = str(doc.metadata.get("source", "")).rsplit("/", 1)[-1] or "unknown source"
    page = doc.metadata.get("page")
    return f"{source} p.{page + 1}" if isinstance(page, int) else source


def format_context(docs: Iterable[Document]) -> str:
    """Number the retrieved chunks and label each with its origin, for citation.

    Only source and page ride along - the LLM-extracted tags (title, keywords,
    hasCode) stay out, because they are retrieval metadata and would read as
    content. Without the labels an answer is unverifiable: the reader cannot tell
    which page it came from, or whether it came from the corpus at all.

    Blank chunks are dropped rather than joined, so an empty return value always
    means "nothing worth answering from" - that is what main() branches on.
    """
    return "\n\n".join(
        f"[{number}] ({citation_label(doc)})\n{doc.page_content}"
        for number, doc in enumerate(usable_chunks(docs), start=1)
    )


def usable_chunks(docs: Iterable[Document]) -> list[Document]:
    """Chunks with actual content, in rank order - the one source of citation numbering.

    format_context and the printed source list must number identically, or [2] in the
    answer points at a different page than [2] on screen. Sharing this is what stops
    those two from drifting apart.
    """
    return [doc for doc in docs if doc.page_content.strip()]


def format_sources(docs: Iterable[Document]) -> str:
    """The citation key printed under the answer - labels only, never chunk text."""
    return "\n".join(
        f"  [{number}] {citation_label(doc)}"
        for number, doc in enumerate(usable_chunks(docs), start=1)
    )


RERANK_MODEL = "rerank-2.5"

# Relevance floor applied to reranker scores, not vector similarity. Calibrated
# 2026-08-05 with `python -m evaluation.calibrate` against the live 171-chunk corpus:
# across 22 golden questions the highest off-topic control scored 0.4023 and the
# lowest answerable question scored 0.6875 - a clean gap with no overlap. 0.55 sits
# inside it with room on both sides.
#
# The old vector thresholds (0.75/0.71) do NOT carry over: they scored cosine
# similarity, a different quantity produced by a different stage. Re-run calibrate
# if the corpus, the embedding model, or RERANK_MODEL changes.
RERANK_THRESHOLD = 0.55
# The retry floor, still above the highest control seen (0.4023), so a genuine
# near-miss gets a second chance without letting off-topic noise through.
RELAXED_RERANK_THRESHOLD = 0.45


# Both network calls in the query path retry with the same policy. The numbers come
# from evaluation/run_eval.py, where they were proven against this exact Voyage free
# tier (3 requests/min): a query costs two Voyage requests, so a 429 is the routine
# failure here, not the exceptional one. A slow answer beats a traceback.
RETRIEVE_MAX_ATTEMPTS = 8
RETRIEVE_RETRY_WAIT_MULTIPLIER = 10
RETRIEVE_RETRY_MAX_WAIT_SECONDS = 90

# Only these are worth retrying. Voyage separates transient failures (rate limits,
# timeouts, its own outages) from ones that will never succeed on a second try
# (bad key, malformed request) - retrying the latter would turn an instant, readable
# error into an 8-attempt, ~4.5-minute wait that still ends in the same failure.
# pymongo's equivalents cover connection loss and a cluster mid-election, not a
# query error like a bad filter.
RETRYABLE_ERRORS = (
    voyageai.error.RateLimitError,
    voyageai.error.ServiceUnavailableError,
    voyageai.error.APIConnectionError,
    voyageai.error.Timeout,
    voyageai.error.TryAgain,
    voyageai.error.ServerError,
    pymongo.errors.ConnectionFailure,
    pymongo.errors.ServerSelectionTimeoutError,
    pymongo.errors.ExecutionTimeout,
    pymongo.errors.NotPrimaryError,
)


def _report_retry(state: Any) -> None:
    """Say why the answer is slow, so a rate limit is visible rather than a hang.

    A near-copy of load_data._report_retry, deliberately not imported: rag.py must
    not pull in the ingest module, which drags PyPDFLoader and langchain_community
    into the query path. tests/test_rag.py guards that boundary.
    """
    sleep = getattr(state.next_action, "sleep", 0)
    error = state.outcome.exception()
    print(
        f"  retrieval call failed ({type(error).__name__}: {error}); "
        f"attempt {state.attempt_number}/{RETRIEVE_MAX_ATTEMPTS}, retrying in {sleep:.0f}s"
    )


def retryer() -> Retrying:
    """A fresh backoff policy per call, shared by rag.py and evaluation/calibrate.py.

    Public (no leading underscore) because calibrate.py needs the exact same policy
    for a query shape retrieve() doesn't support (top_k=1) - a private name would
    mark that cross-module call as reaching into an implementation detail it isn't.

    Fresh rather than a module-level singleton for two reasons: a Retrying instance
    carries per-run state, and building it here means the wait constants are read at
    call time - which is what lets tests set the multiplier to 0 instead of sleeping
    through a real 90-second backoff.

    retry=retry_if_exception_type(RETRYABLE_ERRORS) so only transient failures pay
    the backoff; anything else (bad API key, programming error) surfaces immediately
    instead of being retried for several minutes and misreported as a rate limit.

    reraise=True so an exhausted retryable failure surfaces the original error.
    Swallowing it would return no documents, and "retrieval is broken" would become
    indistinguishable from "the corpus has no answer" - the exact silent failure
    NO_CONTEXT_MESSAGE exists to make loud.
    """
    return Retrying(
        stop=stop_after_attempt(RETRIEVE_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=RETRIEVE_RETRY_WAIT_MULTIPLIER, max=RETRIEVE_RETRY_MAX_WAIT_SECONDS
        ),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        before_sleep=_report_retry,
        reraise=True,
    )


def retriever_config(candidate_k: int = CANDIDATE_K) -> tuple[str, dict[str, Any]]:
    """Cast a wide net - this stage owns recall, the reranker owns precision.

    Deliberately no score_threshold: gating here would cap what the reranker can
    ever see, which is the whole point of retrieving CANDIDATE_K instead of TOP_K.

    Deliberately no pre_filter either. This used to carry
    `{"hasCode": {"$eq": False}}`, which made every code-bearing page permanently
    unreachable - "how do I create an index" is answered by a page the filter hid.
    Excluding code was never a retrieval requirement, only an assumption.
    """
    return "similarity", {"k": candidate_k}


def build_candidate_retriever(
    vector_store: MongoDBAtlasVectorSearch, candidate_k: int = CANDIDATE_K
) -> Any:
    """Hybrid candidate generator: vector search and full-text search, fused by RRF.

    Embeddings are good at meaning and weak at exact tokens - "$unwind", "4.0",
    "createIndex" are precisely the strings a user quotes and a dense vector blurs.
    Lexical search is the opposite. Reciprocal Rank Fusion merges both ranked lists
    without either needing a calibrated score, which is why it survives an embedding
    model swap that would invalidate any tuned similarity cutoff.

    Falls back to vector-only when the full-text index is absent, so a collection
    ingested before FULLTEXT_INDEX_NAME existed still answers instead of erroring.
    """
    if not _has_fulltext_index(vector_store):
        print(
            f"note: full-text index {FULLTEXT_INDEX_NAME!r} not found - "
            "using vector-only search. Re-run load_data.py to create it."
        )
        search_type, search_kwargs = retriever_config(candidate_k)
        return vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    return MongoDBAtlasHybridSearchRetriever(
        vectorstore=vector_store,
        search_index_name=FULLTEXT_INDEX_NAME,
        k=candidate_k,
    )


# Whether a namespace has the full-text index, remembered per process. Index presence
# does not change mid-run, and without this every single query pays an extra Atlas
# round-trip to ask - and reprints the same warning.
_fulltext_index_cache: dict[str, bool] = {}


def _has_fulltext_index(vector_store: MongoDBAtlasVectorSearch) -> bool:
    collection = vector_store.collection
    namespace = f"{collection.database.name}.{collection.name}"
    if namespace not in _fulltext_index_cache:
        _fulltext_index_cache[namespace] = _lookup_fulltext_index(collection)
    return _fulltext_index_cache[namespace]


def _lookup_fulltext_index(collection: Any) -> bool:
    try:
        return FULLTEXT_INDEX_NAME in {index["name"] for index in collection.list_search_indexes()}
    except Exception as error:  # noqa: BLE001 - any listing failure means "assume absent"
        print(f"note: could not list search indexes ({type(error).__name__}: {error})")
        return False


def make_reranker() -> Any:
    """Cross-encoder that scores each candidate against the full query text.

    Embedding distance compares two summaries of meaning; a reranker reads the pair
    together, so it separates "mentions indexes" from "explains creating an index"
    in a way cosine similarity on a 1024-dim vector cannot.
    """
    return voyageai.Client(  # pyright: ignore[reportPrivateImportUsage] - public, no __all__
        api_key=key_param.VOYAGE_API_KEY
    )


def score_candidates(
    query: str, docs: Sequence[Document], reranker: Any, top_k: int = TOP_K
) -> list[tuple[Document, float]]:
    """Score candidates with the reranker - one API call, best first.

    Separate from the threshold so a caller trying two floors pays for one request.
    Scores are deterministic for a given query and candidate set, so re-scoring to
    apply a lower floor would buy nothing and cost a paid round-trip.

    The empty-docs guard sits above the retry on purpose: a call that never happens
    should not spend retry budget.
    """
    if not docs:
        return []

    response = retryer()(
        reranker.rerank,
        query,
        [doc.page_content for doc in docs],
        model=RERANK_MODEL,
        top_k=top_k,
    )
    return [(docs[result.index], result.relevance_score) for result in response.results]


def keep_above(scored: Sequence[tuple[Document, float]], threshold: float) -> list[Document]:
    """Drop anything under the floor, preserving the reranker's order.

    Returns Documents, not pairs, so the caller keeps the chunk metadata that
    citations need. Empty means nothing cleared the floor - the caller treats that
    exactly like retrieving nothing.
    """
    return [doc for doc, score in scored if score >= threshold]


def build_vector_store(client: "MongoClient[dict[str, Any]]") -> MongoDBAtlasVectorSearch:
    """Bind the vector store to the shared namespace - one place, so eval and main agree."""
    return MongoDBAtlasVectorSearch(
        collection=client[DB_NAME][COLLECTION_NAME],
        embedding=make_embeddings(),
        index_name=INDEX_NAME,
        text_key=TEXT_KEY,
    )


def retrieve(
    vector_store: MongoDBAtlasVectorSearch, query: str, reranker: Any
) -> list[Document]:
    """Retrieve wide, rerank, keep the best few - retrying once at a relaxed floor.

    One Atlas query and one rerank call, then two thresholds applied to the same
    scores. The second chance is about the gate being too strict, never about the
    net being too small - and re-scoring to lower a floor would pay twice for an
    identical answer.

    Both network calls retry with backoff (see retryer). Retriever construction stays
    outside the retry: it is a cached index lookup, not the flaky part.

    The eval harness calls this, not main(), so what is measured is what runs - and
    since the backoff now lives here, the harness inherits it instead of wrapping it.
    """
    candidates = retryer()(build_candidate_retriever(vector_store).invoke, query)
    scored = score_candidates(query, candidates, reranker)

    for threshold in (RERANK_THRESHOLD, RELAXED_RERANK_THRESHOLD):
        kept = keep_above(scored, threshold)
        if format_context(kept):
            return kept
    return []


def stream_answer(chunks: Iterable[str], out: TextIO = sys.stdout) -> None:
    """Echo tokens as the LLM produces them - flush per token or stdout buffers."""
    for chunk in chunks:
        out.write(chunk)
        out.flush()
    out.write("\n")
    out.flush()


def main() -> None:
    require_secrets(key_param)

    client: MongoClient[dict[str, Any]]
    with MongoClient(key_param.MONGODB_URI) as client:
        vector_store = build_vector_store(client)

        query = resolve_query(sys.argv)
        print(f"Query: {query}\n")
        docs = retrieve(vector_store, query, make_reranker())
        context = format_context(docs)

        if not context:
            print(NO_CONTEXT_MESSAGE)
            return

        template = """
            Use the following pieces of context to answer the question at the end.
            Each piece is numbered and labelled with the file and page it came from.
            Cite the pieces you used inline, like [1] or [2], immediately after the
            claim they support. Do not cite a piece you did not use.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Do not answer the question if there is no given context.
            Do not answer the question if it is not related to the context.
            Do not give recommendations to anything other than MongoDB.
            Context:
            {context}
            Question: {question}
            """
        custom_rag_prompt = PromptTemplate.from_template(template)

        llm = ChatOpenAI(
                api_key=SecretStr(key_param.LLM_API_KEY),
                base_url=key_param.LLM_BASE_URL,
                temperature=0,
                model=key_param.LLM_MODEL,
            )
        rag_chain = custom_rag_prompt | llm | StrOutputParser()
        print("\nAnswer:")
        stream_answer(rag_chain.stream({
            "context": context,
            "question": query,
        }))
        print("Sources:")
        print(format_sources(docs))


if __name__ == "__main__":
    main()
