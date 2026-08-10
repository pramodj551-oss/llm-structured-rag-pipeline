# app.py
"""
Production-style Streamlit application
LLM-Powered Structured Insight & RAG Retrieval Pipeline

Features:
- Structured Support Ticket Extraction
- Pydantic Validation
- RAG Retrieval
- Top-K Retrieved Chunks
- Similarity/Relevance Scores
- RAG Sources
- Grounded Answer
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LLM Structured Insight & RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS / ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
KB_PATH = DATA_DIR / "knowledge_base.txt"


# ============================================================
# SAFE IMPORTS
# ============================================================

IMPORT_ERRORS: list[str] = []


try:
    from src.schemas import (
        ExtractedSupportTicket,
        UrgencyLevel,
        TicketCategory,
        SentimentLevel,
    )
except Exception as exc:
    ExtractedSupportTicket = None
    UrgencyLevel = None
    TicketCategory = None
    SentimentLevel = None

    IMPORT_ERRORS.append(
        f"schemas.py import failed: {exc}"
    )


try:
    from src.extraction_pipeline import (
        extract_ticket,
    )
except Exception as exc:
    extract_ticket = None

    IMPORT_ERRORS.append(
        f"extraction_pipeline.py import failed: {exc}"
    )


try:
    from src.rag_pipeline import (
        initialize_rag,
        retrieve_documents,
        generate_grounded_answer,
    )
except Exception as exc:
    initialize_rag = None
    retrieve_documents = None
    generate_grounded_answer = None

    IMPORT_ERRORS.append(
        f"rag_pipeline.py import failed: {exc}"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = None

if "rag_embedder" not in st.session_state:
    st.session_state.rag_embedder = None

if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []

if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_ticket" not in st.session_state:
    st.session_state.last_ticket = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .source-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
        background-color: rgba(128,128,128,0.05);
    }

    .score {
        font-weight: 700;
    }

    .grounded-answer {
        border-left: 4px solid #4CAF50;
        padding: 12px 16px;
        margin-top: 10px;
        background-color: rgba(76,175,80,0.08);
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 LLM-Powered Structured Insight & RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Pydantic-validated ticket extraction + local semantic retrieval +
    grounded LLM answers
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    st.divider()

    st.subheader("🔐 API Configuration")

    api_key_available = bool(
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if api_key_available:
        st.success("Google API key detected")
    else:
        st.warning(
            "Google API key not detected.\n\n"
            "Add GOOGLE_API_KEY to your .env file."
        )

    st.divider()

    st.subheader("📚 Knowledge Base")

    if KB_PATH.exists():
        kb_size = KB_PATH.stat().st_size

        st.success("Knowledge base found")

        st.caption(
            f"File: {KB_PATH.name}"
        )

        st.caption(
            f"Size: {kb_size:,} bytes"
        )
    else:
        st.error(
            "knowledge_base.txt not found"
        )

    st.divider()

    if st.button(
        "🔄 Initialize / Reload RAG",
        use_container_width=True,
    ):

        if initialize_rag is None:
            st.error(
                "RAG pipeline could not be imported."
            )
        elif not KB_PATH.exists():
            st.error(
                "knowledge_base.txt not found."
            )
        else:

            with st.spinner(
                "Initializing embedding model and vector database..."
            ):

                try:

                    result = initialize_rag(
                        str(KB_PATH)
                    )

                    if isinstance(result, tuple):

                        if len(result) >= 2:
                            st.session_state.rag_collection = result[0]
                            st.session_state.rag_embedder = result[1]

                        if len(result) >= 3:
                            st.session_state.rag_chunks = result[2]

                    else:
                        st.session_state.rag_collection = result

                    st.session_state.rag_initialized = True

                    st.success(
                        "RAG initialized successfully."
                    )

                except Exception as exc:

                    st.session_state.rag_initialized = False

                    st.error(
                        f"RAG initialization failed: {exc}"
                    )

                    with st.expander(
                        "Technical details"
                    ):
                        st.code(
                            traceback.format_exc()
                        )


# ============================================================
# IMPORT STATUS
# ============================================================

if IMPORT_ERRORS:

    with st.expander(
        "⚠️ Component Import Diagnostics"
    ):

        for error in IMPORT_ERRORS:
            st.warning(error)


# ============================================================
# TABS
# ============================================================

tab_rag, tab_ticket, tab_status = st.tabs(
    [
        "🔎 RAG Assistant",
        "🎫 Ticket Extraction",
        "🩺 System Status",
    ]
)


# ============================================================
# RAG ASSISTANT
# ============================================================

with tab_rag:

    st.header("🔎 Grounded RAG Assistant")

    st.write(
        """
        Ask a question about the internal technical documentation.
        The system retrieves the most relevant chunks and generates
        an answer grounded in those sources.
        """
    )

    query = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What should I do if a user cannot log in?"
        ),
        height=120,
    )

    col1, col2 = st.columns(
        [1, 4]
    )

    with col1:

        search_clicked = st.button(
            "🔍 Search & Answer",
            type="primary",
            use_container_width=True,
        )

    with col2:

        if st.session_state.rag_initialized:
            st.success(
                "RAG system ready"
            )
        else:
            st.info(
                "Initialize RAG from the sidebar first."
            )

    if search_clicked:

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        elif retrieve_documents is None:

            st.error(
                "retrieve_documents() could not be imported."
            )

        elif not st.session_state.rag_initialized:

            st.error(
                "Please initialize the RAG system first."
            )

        else:

            with st.spinner(
                "Retrieving relevant knowledge..."
            ):

                try:

                    retrieval_result = retrieve_documents(
                        query=query,
                        collection=st.session_state.rag_collection,
                        embedder=st.session_state.rag_embedder,
                        top_k=top_k,
                    )

                    st.session_state.last_retrieval = (
                        retrieval_result
                    )

                except TypeError:

                    # Compatibility fallback for simpler
                    # retrieve_documents implementations.

                    try:

                        retrieval_result = retrieve_documents(
                            query,
                            st.session_state.rag_collection,
                            st.session_state.rag_embedder,
                            top_k,
                        )

                        st.session_state.last_retrieval = (
                            retrieval_result
                        )

                    except Exception as exc:

                        st.error(
                            f"Retrieval failed: {exc}"
                        )

                        with st.expander(
                            "Technical details"
                        ):
                            st.code(
                                traceback.format_exc()
                            )

                        retrieval_result = None

                except Exception as exc:

                    st.error(
                        f"Retrieval failed: {exc}"
                    )

                    with st.expander(
                        "Technical details"
                    ):
                        st.code(
                            traceback.format_exc()
                        )

                    retrieval_result = None


            # ------------------------------------------------
            # NORMALIZE RETRIEVAL RESULTS
            # ------------------------------------------------

            normalized_results: list[dict[str, Any]] = []

            if retrieval_result:

                if isinstance(
                    retrieval_result,
                    dict,
                ):

                    documents = retrieval_result.get(
                        "documents",
                        [],
                    )

                    metadatas = retrieval_result.get(
                        "metadatas",
                        [],
                    )

                    distances = retrieval_result.get(
                        "distances",
                        [],
                    )

                    if documents and isinstance(
                        documents[0],
                        list,
                    ):
                        documents = documents[0]

                    if metadatas and isinstance(
                        metadatas[0],
                        list,
                    ):
                        metadatas = metadatas[0]

                    if distances and isinstance(
                        distances[0],
                        list,
                    ):
                        distances = distances[0]

                    for index, document in enumerate(
                        documents
                    ):

                        metadata = (
                            metadatas[index]
                            if index < len(metadatas)
                            else {}
                        )

                        distance = (
                            distances[index]
                            if index < len(distances)
                            else None
                        )

                        normalized_results.append(
                            {
                                "rank": index + 1,
                                "text": document,
                                "metadata": metadata or {},
                                "distance": distance,
                            }
                        )

                elif isinstance(
                    retrieval_result,
                    list,
                ):

                    for index, item in enumerate(
                        retrieval_result
                    ):

                        if isinstance(
                            item,
                            dict,
                        ):

                            normalized_results.append(
                                {
                                    "rank": index + 1,
                                    "text": item.get(
                                        "text",
                                        item.get(
                                            "document",
                                            "",
                                        ),
                                    ),
                                    "metadata": item.get(
                                        "metadata",
                                        {},
                                    ),
                                    "distance": item.get(
                                        "distance",
                                        item.get(
                                            "score"
                                        ),
                                    ),
                                }
                            )

                        else:

                            normalized_results.append(
                                {
                                    "rank": index + 1,
                                    "text": str(item),
                                    "metadata": {},
                                    "distance": None,
                                }
                            )


            # ------------------------------------------------
            # RETRIEVAL RESULTS
            # ------------------------------------------------

            if normalized_results:

                st.session_state.last_retrieval = (
                    normalized_results
                )

                st.subheader(
                    "📚 Retrieved Context"
                )

                st.caption(
                    f"Top {len(normalized_results)} "
                    f"relevant chunks retrieved"
                )

                for result in normalized_results:

                    rank = result["rank"]
                    text = result["text"]
                    metadata = result["metadata"]
                    distance = result["distance"]

                    if distance is not None:

                        try:

                            distance_value = float(
                                distance
                            )

                            # ChromaDB commonly returns
                            # distance where lower is better.
                            relevance = (
                                1.0
                                / (
                                    1.0
                                    + distance_value
                                )
                            )

                            score_text = (
                                f"{relevance:.4f}"
                            )

                        except (
                            ValueError,
                            TypeError,
                        ):

                            score_text = str(
                                distance
                            )

                    else:

                        score_text = "N/A"

                    source = (
                        metadata.get(
                            "source",
                            metadata.get(
                                "file",
                                "knowledge_base.txt",
                            ),
                        )
                    )

                    with st.expander(
                        f"#{rank} — "
                        f"Source: {source} — "
                        f"Relevance: {score_text}"
                    ):

                        st.markdown(
                            f"""
                            <div class="source-card">

                            <b>Source:</b> {source}<br>

                            <b>Chunk Rank:</b> {rank}<br>

                            <b>Similarity/Relevance:</b>
                            <span class="score">
                            {score_text}
                            </span>

                            <hr>

                            {text}

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            else:

                st.warning(
                    "No relevant chunks were retrieved."
                )


            # ------------------------------------------------
            # GROUNDED ANSWER
            # ------------------------------------------------

            if (
                normalized_results
                and generate_grounded_answer is not None
            ):

                context_parts = []

                for item in normalized_results:

                    source = item[
                        "metadata"
                    ].get(
                        "source",
                        "knowledge_base.txt",
                    )

                    context_parts.append(
                        f"""
SOURCE: {source}

{item['text']}
"""
                    )

                context = "\n\n".join(
                    context_parts
                )

                with st.spinner(
                    "Generating grounded answer..."
                ):

                    try:

                        answer = (
                            generate_grounded_answer(
                                query=query,
                                context=context,
                            )
                        )

                        st.session_state.last_answer = (
                            answer
                        )

                    except TypeError:

                        try:

                            answer = (
                                generate_grounded_answer(
                                    query,
                                    context,
                                )
                            )

                            st.session_state.last_answer = (
                                answer
                            )

                        except Exception as exc:

                            st.error(
           
