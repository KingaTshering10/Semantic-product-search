"""Catalog loading and semantic search shared by the notebook and Streamlit app."""

import csv
from math import isfinite
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

CATALOG_PATH = Path(__file__).resolve().parent / "OutdoorClothingCatalog_1000.csv"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def load_catalog_documents(path=CATALOG_PATH):
    """Preserve multiline descriptions and keep row identity separate from the text."""
    path = Path(path)
    documents = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not {"name", "description"}.issubset(reader.fieldnames or []):
            raise ValueError("Catalog must contain name and description columns.")
        for row_number, row in enumerate(reader, start=1):
            name = (row.get("name") or "").strip()
            description = (row.get("description") or "").strip()
            if not name or not description:
                raise ValueError(f"Catalog row {row_number} has an empty name or description.")
            documents.append(Document(
                id=str(row_number),
                page_content=f"name: {name}\ndescription: {description}",
                metadata={"name": name, "description": description,
                          "source": path.name, "row": row_number},
            ))
    if not documents:
        raise ValueError("Catalog has no product rows.")
    return documents


def create_embeddings(model_name=EMBEDDING_MODEL):
    # Import lazily so catalog and vector-store checks need no model download.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"normalize_embeddings": True, "prompt": QUERY_INSTRUCTION},
    )


def build_search_index(documents, embeddings):
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(documents)
    return store


def find_products(store, query, k=4, min_score=-1.0):
    """Return up to k nearest rows, filtered by their cosine similarity score."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a non-empty product query.")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not isfinite(min_score) or not -1.0 <= min_score <= 1.0:
        raise ValueError("Minimum similarity must be between -1 and 1.")
    matches = store.similarity_search_with_score(query.strip(), k=k)
    return [(document, float(score)) for document, score in matches
            if float(score) >= min_score]
