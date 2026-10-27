# Preparation notes

## Source project

The uploaded archive contains two notebooks, a Streamlit app, a 1,000-row catalog, an existing short README, and `req.txt`. Its product Q&A implementation retrieves catalog records; it does not use a generative language model.

## Changes

- Retain the original catalog byte for byte and exclude macOS metadata.
- Rename the notebooks to `sentence_similarity.ipynb` and `product_search.ipynb`; clear old outputs and machine-specific kernel metadata.
- Correct the product notebook's BGE description. The code uses the English `BAAI/bge-small-en-v1.5` checkpoint, not BGE-M3.
- Replace deprecated community embedding imports with `langchain_huggingface.HuggingFaceEmbeddings`.
- Replace the DocArray/community vector-store dependency with `langchain_core.vectorstores.InMemoryVectorStore`.
- Load CSV rows with the Python standard library, preserving multiline descriptions. Store names, descriptions, and stable one-based row IDs in document metadata. Repeated product names remain separate records.
- Share catalog and search functions between the app and retrieval notebook through `search.py`.
- Normalize embeddings and use the BGE retrieval instruction for queries.
- Display cosine similarity and provide an adjustable score filter. The default 0.5 value is not calibrated and cannot guarantee relevance.
- Retain resource caching, including the catalog modification timestamp, and include the embedding model name in the cache inputs.
- Add input validation and user-facing initialization/search failure handling.
- Rename app headings and buttons to describe product search accurately.
- Replace the dependency list with the packages used by the project; remove unused TensorFlow and TensorFlow Hub dependencies.
- Respect the current Hugging Face integration's published dependency range by using Sentence Transformers 5.2 or newer within the 5.x series. The original 6.1.0 pin was outside the integration's `full` extra range at preparation time.
- Retain `req.txt` as a shortcut to `requirements.txt`.

## Verification performed

- Scripts and notebook code parse successfully.
- All 1,000 source rows produce documents with unique row IDs.
- Each loaded name and complete description matches its CSV source after trimming outer whitespace.
- The installed LangChain Core 1.7.0 vector store was exercised with deterministic three-dimensional test embeddings: nearest-neighbor ranking, top-k limits, and score filtering worked as expected.
- Empty queries, invalid result counts, invalid score bounds, and malformed catalog headers are rejected.
- No embedding-model calls, model downloads, or Streamlit server runs were part of these checks.

The deterministic embeddings test the software's retrieval mechanics; they provide no evidence of the real model's semantic retrieval quality. A complete dependency installation and end-to-end application run remain unverified.

## Remaining limitations

The application has no relevance-labeled evaluation set, reranker, exact attribute filtering, document chunking, persistent vector database, upload feature, or generated-answer stage. Long entries may be truncated by the embedding model. The score filter is a user control rather than a validated out-of-domain detector.

The original archive did not establish the catalog's licensing or attribution, and no project code license was supplied. These are not assigned or inferred during preparation.
