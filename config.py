"""Settings shared by ingestion (load_data.py) and retrieval (rag.py).

Kept in its own module so the query path does not import the PDF/ingest stack.
Query and stored vectors must come from the same embedding model, or vector
search silently returns nothing.
"""

DB_NAME = "book_mongodb_chunks"
COLLECTION_NAME = "chunked_data"
EMBED_MODEL = "voyage-3.5-lite"
