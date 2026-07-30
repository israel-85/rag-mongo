import sys
from collections.abc import Iterable
from typing import Any, TextIO

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


def make_embeddings() -> VoyageAIEmbeddings:
    """Same model as ingestion - query and stored vectors must share dimensions."""
    return VoyageAIEmbeddings(api_key=SecretStr(key_param.VOYAGE_API_KEY), model=EMBED_MODEL)


def resolve_query(argv: list[str]) -> str:
    """First CLI argument is the query; fall back to the demo question."""
    return argv[1] if len(argv) > 1 else DEFAULT_QUERY


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

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K,
                    "pre_filter": { "hasCode": { "$eq": False } },
                    "score_threshold": 0.01
    })
        query = resolve_query(sys.argv)
        print(f"Query: {query}\n")
        docs = retriever.invoke(query)
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
            "context": "\n\n".join(d.page_content for d in docs),
            "question": query,
        }))


if __name__ == "__main__":
    main()
