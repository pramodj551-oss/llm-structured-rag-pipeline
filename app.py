"""
Production-style Streamlit application
LLM-Powered Structured Insight & RAG Retrieval Pipeline

Features
--------
1. Structured support-ticket extraction using Gemini + Pydantic
2. RAG-based question answering using:
   - Sentence Transformers
   - ChromaDB
   - Gemini
3. Top-K retrieved chunks
4. Similarity / relevance scores
5. Grounded answer with source context
6. Graceful error handling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

TICKETS_FILE = DATA_DIR / "support_tickets.json"
KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.txt"

APP_TITLE = "LLM-Powered Structured Insight & RAG"
APP_ICON = "🧠"

DEFAULT_TOP_K = 5
MAX_TOP_K = 10


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


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
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .source-card {
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }

        .score {
            font-weight: 700;
        }

        .grounded-answer {
            border-left: 4px solid #2563eb;
            padding: 1rem;
            background: #f8fafc;
            border-radius: 6px;
        }

        .status-success {
            color: #15803d;
            font-weight: 600;
        }

        .status-warning {
            color: #b45309;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_api_key() -> str | None:
    """
    Retrieve Gemini API key.

    Priority:
    1. Streamlit secrets
    2. Environment variable
    """

    # Streamlit Cloud / local secrets.toml
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    # Environment variable
    key = os.getenv("GEMINI_API_KEY")

    if key:
        return key.strip()

    # Backward-compatible environment variable
    key = os.getenv("GOOGLE_API_KEY")

    if key:
        return key.strip()

    return None


def load_ticket_data() -> list[dict[str, Any]]:
    """Load support-ticket records safely."""

    if not TICKETS_FILE.exists():
        return []

    try:
        with open(TICKETS_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            return data

        return []

    except Exception as exc:
        st.error(f"Unable to load support tickets: {exc}")
        return []


def validate_project_structure() -> list[str]:
    """Check important project files."""

    missing = []

    required_files = [
        TICKETS_FILE,
        KNOWLEDGE_BASE_FILE,
        SRC_DIR / "__init__.py",
        SRC_DIR / "schemas.py",
        SRC_DIR / "extraction_pipeline.py",
        SRC_DIR / "rag_pipeline.py",
    ]

    for file_path in required_files:
        if not file_path.exists():
            missing.append(str(file_path.relative_to(BASE_DIR)))

    return missing


# ============================================================
# IMPORT PIPELINES
# ============================================================

@st.cache_resource(show_spinner=False)
def load_pipelines():
    """
    Import application pipelines.

    Expected functions:

    extraction_pipeline.py
        extract_ticket(...)

    rag_pipeline.py
        initialize_rag(...)
        retrieve_context(...)
        generate_grounded_answer(...)
    """

    try:
        from src.extraction_pipeline import extract_ticket
        from src.rag_pipeline import (
            initialize_rag,
            retrieve_context,
            generate_grounded_answer,
        )

        return {
            "extract_ticket": extract_ticket,
            "initialize_rag": initialize_rag,
            "retrieve_context": retrieve_context,
            "generate_grounded_answer": generate_grounded_answer,
        }

    except Exception as exc:
        raise RuntimeError(
            f"Unable to load application pipelines: {exc}"
        ) from exc


# ============================================================
# RAG INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model and ChromaDB...")
def initialize_rag_system(_initialize_rag):
    """
    Initialize RAG system once per Streamlit process.
    """

    try:
        return _initialize_rag(
            knowledge_base_path=str(KNOWLEDGE_BASE_FILE)
        )

    except TypeError:
        # Fallback for simpler implementations
        return _initialize_rag(str(KNOWLEDGE_BASE_FILE))

    except Exception as exc:
        raise RuntimeError(
            f"Unable to initialize RAG system: {exc}"
        ) from exc


# ============================================================
# RAG RETRIEVAL NORMALIZER
# ============================================================

def normalize_retrieved_chunks(results: Any) -> list[dict[str, Any]]:
    """
    Normalize different possible retrieval outputs.

    Supported formats:

    [
        {
            "text": "...",
            "score": 0.91,
            "source": "knowledge_base.txt"
        }
    ]

    or Chroma-like results.
    """

    normalized: list[dict[str, Any]] = []

    if results is None:
        return normalized

    # --------------------------------------------------------
    # Already normalized list
    # --------------------------------------------------------

    if isinstance(results, list):

        for index, item in enumerate(results):

            if isinstance(item, dict):

                text = (
                    item.get("text")
                    or item.get("document")
                    or item.get("content")
                    or ""
                )

                score = (
                    item.get("score")
                    or item.get("similarity")
                    or item.get("relevance_score")
                )

                source = (
                    item.get("source")
                    or item.get("metadata", {}).get("source")
                    or "knowledge_base.txt"
                )

                normalized.append(
                    {
                        "rank": index + 1,
                        "text": str(text),
                        "score": score,
                        "source": source,
                        "metadata": item.get("metadata", {}),
                    }
                )

            else:
                normalized.append(
                    {
                        "rank": index + 1,
                        "text": str(item),
                        "score": None,
                        "source": "knowledge_base.txt",
                        "metadata": {},
                    }
                )

        return normalized

    # --------------------------------------------------------
    # Chroma-style result
    # --------------------------------------------------------

    if isinstance(results, dict):

        documents = results.get("documents", [[]])
        metadatas = results.get("metadatas", [[]])
        distances = results.get("distances", [[]])

        documents = documents[0] if documents else []
        metadatas = metadatas[0] if metadatas else []
        distances = distances[0] if distances else []

        for index, document in enumerate(documents):

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

            # Chroma distance is lower = more similar.
            # Convert distance into an intuitive relevance score.
            relevance = None

            if distance is not None:
                try:
                    relevance = 1 / (1 + float(distance))
                except (TypeError, ValueError):
                    relevance = None

            normalized.append(
                {
                    "rank": index + 1,
                    "text": str(document),
                    "score": relevance,
                    "source": metadata.get(
                        "source",
                        "knowledge_base.txt",
                    ),
                    "metadata": metadata,
                }
            )

        return normalized

    return normalized


# ============================================================
# SCORE DISPLAY
# ============================================================

def format_score(score: Any) -> str:
    """Format relevance score."""

    if score is None:
        return "N/A"

    try:
        return f"{float(score):.4f}"
    except (TypeError, ValueError):
        return str(score)


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    st.sidebar.title("🧠 RAG Control Panel")

    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Select Mode",
        [
            "🔎 RAG Q&A",
            "🎫 Ticket Extraction",
            "📊 System Status",
        ],
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Retrieval Settings")

    top_k = st.sidebar.slider(
        "Top-K Documents",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
    )

    st.sidebar.caption(
        "Higher Top-K retrieves more context but may increase "
        "prompt size."
    )

    st.sidebar.markdown("---")

    api_key = get_api_key()

    if api_key:
        st.sidebar.success("Gemini API Key: Configured")
    else:
        st.sidebar.error("Gemini API Key: Missing")

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Embedding Model\n"
        "`all-MiniLM-L6-v2`"
    )

    st.sidebar.caption(
        "Vector Database\n"
        "`ChromaDB`"
    )

    st.sidebar.caption(
        "LLM\n"
        "`Gemini`"
    )

    return mode, top_k


# ============================================================
# RAG Q&A PAGE
# ============================================================

def render_rag_page(pipelines, top_k: int):

    st.markdown(
        '<div class="main-title">🔎 RAG Knowledge Assistant</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Ask questions about the internal technical knowledge base. "
        "Answers are generated using retrieved source context."
        "</div>",
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: How should a critical security incident "
            "be escalated?"
        ),
        height=120,
    )

    col1, col2 = st.columns([1, 5])

    with col1:
        search_clicked = st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True,
        )

    if not search_clicked:
        st.info(
            "Enter a question and click Search to run the RAG pipeline."
        )
        return

    if not query.strip():
        st.warning("Please enter a question.")
        return

    if not get_api_key():
        st.error(
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to Streamlit Secrets or environment variables."
        )
        return

    # --------------------------------------------------------
    # Initialize RAG
    # --------------------------------------------------------

    with st.spinner("Initializing RAG system..."):

        try:
            rag_system = initialize_rag_system(
                pipelines["initialize_rag"]
            )

        except Exception as exc:
            st.error(str(exc))
            return

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    with st.spinner("Searching knowledge base..."):

        try:
            raw_results = pipelines["retrieve_context"](
                rag_system,
                query,
                top_k=top_k,
            )

        except TypeError:

            try:
                raw_results = pipelines["retrieve_context"](
                    rag_system,
                    query,
                    top_k,
                )

            except Exception as exc:
                st.error(
                    f"RAG retrieval failed: {exc}"
                )
                return

        except Exception as exc:
            st.error(
                f"RAG retrieval failed: {exc}"
            )
            return

    retrieved_chunks = normalize_retrieved_chunks(
        raw_results
    )

    if not retrieved_chunks:
        st.warning(
            "No relevant documents were retrieved."
        )
        return

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.subheader("📊 Retrieval Summary")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Documents Retrieved",
        len(retrieved_chunks),
    )

    valid_scores = [
        item["score"]
        for item in retrieved_chunks
        if item["score"] is not None
    ]

    if valid_scores:
        metric2.metric(
            "Best Relevance",
            f"{max(valid_scores):.4f}",
        )

        metric3.metric(
            "Average Relevance",
            f"{sum(valid_scores) / len(valid_scores):.4f}",
        )

    else:
        metric2.metric(
            "Best Relevance",
            "N/A",
        )

        metric3.metric(
            "Average Relevance",
            "N/A",
        )

    # --------------------------------------------------------
    # Retrieved Sources
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        f"📚 Top-{len(retrieved_chunks)} Retrieved Sources"
    )

    for item in retrieved_chunks:

        rank = item["rank"]
        score = item["score"]
        source = item["source"]
        text = item["text"]
        metadata = item.get("metadata", {})

        with st.expander(
            f"#{rank} — {source} — Relevance: {format_score(score)}",
            expanded=(rank == 1),
        ):

            source_col, score_col = st.columns(2)

            with source_col:
                st.markdown("**Source**")
                st.code(str(source))

            with score_col:
                st.markdown("**Similarity / Relevance Score**")
                st.code(format_score(score))

            st.markdown("**Retrieved Chunk**")

            st.markdown(
                f"""
                <div class="source-card">
                    {text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if metadata:
                st.markdown("**Metadata**")
                st.json(metadata)

    # --------------------------------------------------------
    # Build Context
    # --------------------------------------------------------

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE: {item['source']}
RELEVANCE: {format_score(item['score'])}

{item['text']}
"""
        )

    context = "\n\n---\n\n".join(context_parts)

    # --------------------------------------------------------
    # Grounded Answer
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🤖 Grounded Answer")

    with st.spinner(
        "Generating grounded answer from retrieved context..."
    ):

        try:

            answer = pipelines["generate_grounded_answer"](
                query=query,
                context=context,
            )

        except TypeError:

            try:
                answer = pipelines[
                    "generate_grounded_answer"
                ](
                    query,
                    context,
                )

            except Exception as exc:
                st.error(
                    f"Answer generation failed: {exc}"
                )
                return

        except Exception as exc:
            st.error(
                f"Answer generation failed: {exc}"
            )
            return

    if isinstance(answer, dict):
        answer_text = (
            answer.get("answer")
            or answer.get("response")
            or answer.get("text")
            or str(answer)
        )
    else:
        answer_text = str(answer)

    st.markdown(
        f"""
        <div class="grounded-answer">
        {answer_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Grounding Information
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🔗 Answer Grounding")

    st.success(
        f"The answer was generated using "
        f"{len(retrieved_chunks)} retrieved knowledge-base chunk(s)."
    )

    source_names = list(
        dict.fromkeys(
            str(item["source"])
            for item in retrieved_chunks
        )
    )

    st.markdown("**Sources used:**")

    for source in source_names:
        st.markdown(f"- `{source}`")

    st.caption(
        "The assistant is instructed to ground responses in "
        "retrieved internal documentation. Always verify critical "
        "operational decisions against authoritative procedures."
    )


# ============================================================
# TICKET EXTRACTION PAGE
# ============================================================

def render_extraction_page(pipelines):

    st.markdown(
        '<div class="main-title">'
        "🎫 Structured Support Ticket Extraction"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Convert unstructured support tickets into "
        "Pydantic-validated structured records."
        "</div>",
        unsafe_allow_html=True,
    )

    ticket_text = st.text_area(
        "Support Ticket",
        placeholder=(
            "Example:\n"
            "I was charged twice for my subscription and "
            "need one of the charges refunded immediately."
        ),
        height=180,
    )

    if st.button(
        "⚙️ Extract & Validate",
        type="primary",
    ):

        if not ticket_text.strip():
            st.warning("Please enter a support ticket.")
            return

        if not get_api_key():
            st.error(
                "Gemini API key is not configured."
            )
            return

        with st.spinner(
            "Extracting structured ticket..."
            
