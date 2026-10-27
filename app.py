"""Streamlit product search over the included clothing catalog."""

import streamlit as st

from search import (
    CATALOG_PATH, EMBEDDING_MODEL, build_search_index, create_embeddings,
    find_products, load_catalog_documents,
)


@st.cache_resource(show_spinner="Preparing product search...")
def load_search_index(catalog_path: str, catalog_mtime_ns: int, model_name: str):
    # The timestamp is part of the cache key, so changed catalogs are re-indexed.
    documents = load_catalog_documents(catalog_path)
    return build_search_index(documents, create_embeddings(model_name))


def main():
    st.set_page_config(page_title="Semantic Product Search", page_icon="🔎")
    st.title("Semantic Product Search")
    st.write("Find products in the outdoor clothing catalog by describing what you need.")

    if not CATALOG_PATH.is_file():
        st.error(f"Product catalog not found: {CATALOG_PATH.name}")
        st.stop()

    min_score = st.sidebar.slider(
        "Minimum match score", min_value=-1.0, max_value=1.0, value=0.5, step=0.05,
        help="Higher values show fewer matches. Similarity is not a probability of relevance.",
    )
    with st.form("query_form"):
        query = st.text_input("What are you looking for?", placeholder="Shirts with sun protection")
        submitted = st.form_submit_button("Find products", type="primary")
    if submitted:
        if query.strip():
            st.session_state["last_query"] = query.strip()
        else:
            st.warning("Enter a description to search the catalog.")
            st.session_state.pop("last_query", None)
    if "last_query" not in st.session_state:
        return

    try:
        store = load_search_index(str(CATALOG_PATH), CATALOG_PATH.stat().st_mtime_ns, EMBEDDING_MODEL)
        with st.spinner("Searching the catalog..."):
            matches = find_products(store, st.session_state["last_query"], k=4, min_score=min_score)
    except Exception as exc:
        st.error(
            f"Search could not be completed ({type(exc).__name__}). "
            "Check the project dependencies and model download access, then try again."
        )
        return

    st.subheader("Matching products")
    st.caption(f"Results for: {st.session_state['last_query']}")
    if not matches:
        st.info("No products met the selected match score. Try rephrasing or lowering the filter.")
        return
    for index, (document, score) in enumerate(matches, start=1):
        with st.expander(document.metadata["name"], expanded=index == 1):
            st.write(document.metadata["description"])
            st.caption(f"Similarity: {score:.3f} · Catalog row: {document.metadata['row']}")


if __name__ == "__main__":
    main()
