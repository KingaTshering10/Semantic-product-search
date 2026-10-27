# Semantic Product Search

### Find catalog products by describing what you need

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Embeddings-Hugging_Face-FFD21E)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Catalog](https://img.shields.io/badge/Catalog-1%2C000_products-2563EB)

A learning project that uses sentence embeddings to search an outdoor clothing catalog. Describe the products you need, such as **shirts with sun protection** or **comfortable sheet sets**, and inspect up to four catalog matches in a Streamlit interface.

The repository includes a sentence-similarity tutorial, a product retrieval notebook, and shared search functions used by the app. Results are product records and their similarity scores; there is no generative answer model.

**[Quick start](#quick-start) · [How it works](#how-it-works) · [Notebooks](#notebooks) · [Project files](#project-files) · [Verification](#verification)**

## Features

- Semantic retrieval using **`BAAI/bge-small-en-v1.5`** embeddings.
- Local Streamlit interface with product names, full descriptions, and similarity scores.
- Adjustable minimum match score, with clear handling when no results pass the filter.
- An in-memory search index cached across questions within the Streamlit process.
- Catalog re-indexing when its file modification timestamp changes.
- A separate **MiniLM** notebook explaining sentence embeddings and cosine similarity.
- The original 1,000-row training catalog, included unchanged.

## Quick start

Use **Python 3.11 or 3.12** in a separate virtual environment. These commands are suitable for macOS Terminal or the VS Code terminal.

### 1. Clone the repository

```bash
git clone https://github.com/KingaTshering10/Semantic-product-search.git semantic-product-search
cd semantic-product-search
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`req.txt` remains as a compatibility shortcut to the same requirements. TensorFlow and TensorFlow Hub are not required by this implementation.

### 3. Launch the app

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Enter a product description and select **Find products**. Open a result to inspect its description and match score.

The first submitted query loads the pretrained BGE model and builds the index. It requires internet access for the initial model download and can take longer than later searches. Embedding inference runs on the CPU in this example. No hosted generation API or API key is required by the code.

## How it works

1. Load each CSV row as a document containing the product name and description. Keep its row number and display fields in metadata.
2. Encode the catalog with BGE and store the vectors in LangChain's `InMemoryVectorStore`.
3. Encode a search query using the model's recommended retrieval instruction.
4. Rank catalog rows by cosine similarity and retrieve up to four matches.
5. Apply the selected minimum score and display the remaining catalog records.

The app defaults to a minimum score of **0.5**. This is an adjustable demonstration setting, not a threshold calibrated against relevance judgments. Scores are **similarities, not probabilities**, and a high score does not guarantee that a product satisfies every condition in a query.

The app searches the included catalog only. It does not accept document uploads, generate prose answers, verify inventory, or check live prices.

## Notebooks

To use JupyterLab:

```bash
python -m jupyterlab
```

Open a notebook and run its cells from top to bottom. In VS Code, select the `.venv` interpreter as the notebook kernel. Start from the project folder so imports and data paths resolve.

| Notebook | Contents |
| --- | --- |
| `sentence_similarity.ipynb` | Encode two lists of sentences with MiniLM and inspect their pairwise cosine similarities |
| `product_search.ipynb` | Load the catalog, create BGE embeddings, build the shared index, and retrieve products |

The MiniLM tutorial illustrates similarity; it is not a validated metric for evaluating LLM answers. Notebook outputs are cleared so results are generated in your own environment.

## Embedding models

| Model | Purpose | Reference |
| --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | Sentence-similarity tutorial; 384-dimensional embeddings | [Model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| `BAAI/bge-small-en-v1.5` | English product retrieval; 384-dimensional embeddings and a 512-token sequence limit | [Model card](https://huggingface.co/BAAI/bge-small-en-v1.5) |

The BGE checkpoint used here is distinct from BGE-M3. This project does not implement multilingual, sparse, or multi-vector BGE-M3 retrieval. Long product descriptions may be truncated by the model; document chunking is not implemented.

## Catalog

`OutdoorClothingCatalog_1000.csv` contains **1,000 rows**, with `name` and `description` fields plus an unnamed source index column. Every row has a name and description. Some product names repeat, so results retain their catalog row identity rather than treating names as unique identifiers.

The loader preserves internal line breaks in descriptions and excludes the unnamed CSV index from the embedding text. Display fields come from document metadata, avoiding fragile parsing of multiline strings.

The CSV is retained from the supplied training materials. The original archive did not include a dataset license or a definitive attribution record; catalog ownership is not assigned to the maintainer.

## Caching

The Streamlit resource cache stores the model-backed index using the catalog path, modification timestamp, and embedding model name as cache inputs. Later questions reuse that index. Restarting Streamlit rebuilds the in-memory index when needed; vectors are not persisted to a database file.

Restart the app after changing search code or embedding settings. Updating the catalog's modification timestamp causes it to be re-indexed automatically.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit search interface and resource caching |
| `search.py` | Catalog loading, embeddings, index construction, and result filtering |
| `sentence_similarity.ipynb` | MiniLM embedding tutorial |
| `product_search.ipynb` | Product retrieval walkthrough |
| `OutdoorClothingCatalog_1000.csv` | Included training catalog |
| `requirements.txt` | Dependencies |
| `req.txt` | Compatibility shortcut to `requirements.txt` |
| `.gitignore` | Excludes environments, caches, model downloads, and local credentials |
| `docs/preparation-notes.md` | Changes and verification scope |

## Verification

Python scripts and notebook code were syntax-checked. All 1,000 catalog records were loaded and checked against the source, and the actual LangChain vector store was exercised with deterministic test embeddings to verify ranking, filtering, and input validation.

**The pretrained embedding models and Streamlit interface were not run during preparation.** No retrieval-quality benchmark or accuracy result is claimed. See [preparation notes](docs/preparation-notes.md) for details.

## Acknowledgments

Maintained by **[Kinga Tshering](https://github.com/KingaTshering10)**. Adapted from the supplied sentence embedding and product Q&A training material. Pretrained models and third-party packages retain their upstream terms. No project code license was provided in the source archive.
