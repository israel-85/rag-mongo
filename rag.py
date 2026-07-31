import sys
from collections.abc import Iterable
from typing import Any, TextIO

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr
from pymongo import MongoClient

import key_param
from config import COLLECTION_NAME, DB_NAME, EMBED_MODEL

INDEX_NAME = "vector_index"
TOP_K = 3

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


def format_context(docs: Iterable[Document]) -> str:
    """Join retrieved chunks into the prompt's context block - text only, no metadata.

    Blank chunks are dropped rather than joined, so an empty return value always
    means "nothing worth answering from" - that is what main() branches on.
    """
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())


SCORE_THRESHOLD = 0.75
RELAXED_SCORE_THRESHOLD = 0.71


def retriever_config(relaxed: bool = False) -> tuple[str, dict[str, Any]]:
    """search_type must be similarity_score_threshold or score_threshold is silently ignored.

    relaxed=True is the retry pass after an empty first result - it lowers only the
    threshold, to just above the highest off-topic score seen in calibration (0.7025),
    so a genuine near-miss can still surface without reopening the door to noise.
    """
    return "similarity_score_threshold", {
        "k": TOP_K,
        "pre_filter": {"hasCode": {"$eq": False}},
        "score_threshold": RELAXED_SCORE_THRESHOLD if relaxed else SCORE_THRESHOLD,
    }


def stream_answer(chunks: Iterable[str], out: TextIO = sys.stdout) -> None:
    """Echo tokens as the LLM produces them - flush per token or stdout buffers."""
    for chunk in chunks:
        out.write(chunk)
        out.flush()
    out.write("\n")
    out.flush()


def main() -> None:
    client: MongoClient[dict[str, Any]]
    with MongoClient(key_param.MONGODB_URI) as client:
        vector_store = MongoDBAtlasVectorSearch(
            collection=client[DB_NAME][COLLECTION_NAME],
            embedding=make_embeddings(),
            index_name=INDEX_NAME,
        )

        search_type, search_kwargs = retriever_config()
        retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        query = resolve_query(sys.argv)
        print(f"Query: {query}\n")
        docs = retriever.invoke(query)
        context = format_context(docs)

        if not context:
            print("No hits above the primary threshold - retrying with a relaxed one.")
            search_type, search_kwargs = retriever_config(relaxed=True)
            retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
            docs = retriever.invoke(query)
            context = format_context(docs)

        if not context:
            print(NO_CONTEXT_MESSAGE)
            return

        template = """
            Use the following pieces of context to answer the question at the end.
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


if __name__ == "__main__":
    main()
