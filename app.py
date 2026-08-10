import os
import json
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

# ---------------------------------------------------------------------
# PATH / ENVIRONMENT
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------
# OPTIONAL IMPORTS
# ---------------------------------------------------------------------

try:
    from src.schemas import ExtractedSupportTicket
except ImportError as exc:
    ExtractedSupportTicket = None
    SCHEMA_IMPORT_ERROR = str(exc)

try:
    from google import genai
except ImportError:
    genai = None

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Structured Insight & RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .source-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 0.8rem;
    }

    .score {
        font-weight: 700;
    }

    .grounded-answer {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #ddd;
    }

    .status-ok {
        color: #15803d;
        font-weight: 600;
    }

    .status-error {
        color: #b91c1c;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.markdown(
    '<div class="main-title">🧠 LLM-Powered Structured Insight & RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Pydantic-validated structured extraction + local semantic retrieval +
    grounded Gemini answers
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False

if "rag_collection" not in st.session_state:
    st.session_state.rag_collection = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def get_api_key() -> str | None:
    """
    Read Gemini API key from environment.
    """

    key = os.getenv("GOOGLE_API_KEY")

    if key:
        return key.strip()

    return None


def load_knowledge_base() -> str:
    """
    Load knowledge_base.txt.
    """

    kb_path = BASE_DIR / "data" / "knowledge_base.txt"

    if not kb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {kb_path}"
        )

    return kb_path.read_text(encoding="utf-8")


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    """
    Split knowledge base into overlapping chunks.
    """

    if not text.strip():
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def initialize_embeddings():
    """
    Load SentenceTransformer embedding model.
    """

    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed."
        )

    if st.session_state.embedding_model is None:

        st.session_state.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return st.session_state.embedding_model


def initialize_chroma(chunks: list[str]):
    """
    Create an in-memory ChromaDB collection.
    """

    if chromadb is None:
        raise ImportError(
            "chromadb is not installed."
        )

    embedder = initialize_embeddings()

    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    client = chromadb.Client()

    collection_name = "technical_knowledge_base"

    # Recreate collection for clean application startup.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    ids = [
        f"chunk_{index}"
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
    )

    st.session_state.rag_collection = collection
    st.session_state.rag_initialized = True

    return collection


def initialize_gemini():
    """
    Initialize Google GenAI client.
    """

    if genai is None:
        raise ImportError(
            "google-genai is not installed."
        )

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )

    if st.session_state.gemini_client is None:

        st.session_state.gemini_client = genai.Client(
            api_key=api_key
        )

    return st.session_state.gemini_client


def calculate_similarity(distance: float) -> float:
    """
    Convert Chroma distance into a simple similarity score.

    With normalized embeddings, cosine distance is generally:
        distance = 1 - cosine_similarity

    Therefore:
        similarity = 1 - distance
    """

    try:
        similarity = 1.0 - float(distance)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, similarity))


def retrieve_chunks(
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve top-K chunks from ChromaDB.
    """

    collection = st.session_state.rag_collection

    if collection is None:
        raise RuntimeError(
            "RAG collection is not initialized."
        )

    embedder = initialize_embeddings()

    query_embedding = embedder.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=[
            "documents",
            "distances",
        ],
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for index, document in enumerate(documents):

        distance = (
            distances[index]
            if index < len(distances)
            else 1.0
        )

        similarity = calculate_similarity(distance)

        retrieved.append(
            {
                "rank": index + 1,
                "document": document,
                "distance": distance,
                "similarity": similarity,
            }
        )

    return retrieved


def generate_grounded_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """
    Generate answer using only retrieved context.
    """

    client = initialize_gemini()

    if not retrieved_chunks:
        return (
            "I could not retrieve relevant information "
            "from the knowledge base."
        )

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"""
SOURCE {item['rank']}
Similarity: {item['similarity']:.4f}

{item['document']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a technical support knowledge assistant.

Answer the user's question using ONLY the provided
knowledge-base context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   explicitly say that the knowledge base does not
   contain enough information.
4. Give a concise but useful answer.
5. Mention relevant source numbers such as [Source 1]
   or [Source 2].
6. Prefer factual, grounded explanations.

USER QUESTION:
{query}

KNOWLEDGE BASE CONTEXT:
{context}

GROUNDED ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    if response is None:
        return "No response was generated."

    answer = getattr(response, "text", None)

    if not answer:
        return "The model returned an empty response."

    return answer.strip()


def extract_structured_ticket(ticket_text: str):
    """
    Extract a support ticket using Gemini and validate
    the result with Pydantic.

    Uses JSON mode through Gemini's response_mime_type.
    """

    client = initialize_gemini()

    if ExtractedSupportTicket is None:
        raise ImportError(
            f"Unable to import src.schemas: "
            f"{SCHEMA_IMPORT_ERROR}"
        )

    schema = ExtractedSupportTicket.model_json_schema()

    prompt = f"""
Extract the support ticket information from the text below.

Return ONLY valid JSON matching this schema.

SCHEMA:
{json.dumps(schema, indent=2)}

TICKET:
{ticket_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
        },
    )

    raw_text = getattr(response, "text", "")

    if not raw_text:
        raise ValueError(
            "Gemini returned an empty extraction response."
        )

    parsed = json.loads(raw_text)

    validated = ExtractedSupportTicket.model_validate(
        parsed
    )

    return validated


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
    )

    st.divider()

    st.subheader("System Status")

    api_key_exists = bool(get_api_key())

    if api_key_exists:
        st.markdown(
            '<span class="status-ok">✓ Gemini API key detected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-error">✗ Gemini API key missing</span>',
            unsafe_allow_html=True,
        )

    if st.session_state.rag_initialized:
        st.markdown(
            '<span class="status-ok">✓ RAG initialized</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "○ RAG not initialized",
            unsafe_allow_html=True,
        )

    st.divider()

    if st.button(
        "🔄 Initialize / Rebuild RAG",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Loading knowledge base and building vector index..."
            ):

                kb_text = load_knowledge_base()

                chunks = chunk_text(kb_text)

                if not chunks:
                    raise ValueError(
                        "Knowledge base contains no usable text."
                    )

                initialize_chroma(chunks)

            st.success(
                f"RAG initialized successfully with "
                f"{len(chunks)} chunks."
            )

        except Exception as exc:

            st.error(
                f"RAG initialization failed: {exc}"
            )

    st.divider()

    st.caption(
        "Embedding: all-MiniLM-L6-v2"
    )

    st.caption(
        "Vector DB: ChromaDB"
    )

    st.caption(
        "LLM: Gemini 2.5 Flash"
    )


# ---------------------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------------------

tab_rag, tab_extract, tab_validation, tab_about = st.tabs(
    [
        "🔎 RAG Search",
        "🧠 Structured Extraction",
        "🛡️ Validation Demo",
        "ℹ️ About",
    ]
)


# =====================================================================
# RAG SEARCH
# =====================================================================

with tab_rag:

    st.header("🔎 Retrieval-Augmented Generation")

    st.write(
        "Ask a question about the internal technical "
        "knowledge base."
    )

    query = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What should I do if the application "
            "cannot connect to the database?"
        ),
        height=120,
    )

    run_rag = st.button(
        "🚀 Run RAG",
        type="primary",
        use_container_width=True,
    )

    if run_rag:

        if not query.strip():

            st.warning(
                "Please enter a question first."
            )

        elif not st.session_state.rag_initialized:

            st.error(
                "RAG is not initialized. "
                "Click 'Initialize / Rebuild RAG' "
                "from the sidebar first."
            )

        elif not get_api_key():

            st.error(
                "GOOGLE_API_KEY is missing. "
                "Configure it in your environment or .env file."
            )

        else:

            try:

                with st.spinner(
                    "Retrieving relevant knowledge..."
                ):

                    retrieved = retrieve_chunks(
                        query=query,
                        top_k=top_k,
                    )

                st.session_state.last_results = retrieved

                if not retrieved:

                    st.warning(
                        "No relevant chunks were retrieved."
                    )

                else:

                    st.success(
                        f"Retrieved {len(retrieved)} relevant chunks."
                    )

                    # -------------------------------------------------
                    # RETRIEVAL METRICS
                    # -------------------------------------------------

                    best_score = retrieved[0]["similarity"]

                    avg_score = sum(
                        item["similarity"]
                        for item in retrieved
                    ) / len(retrieved)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Chunks Retrieved",
                            len(retrieved),
                        )

                    with col2:
                        st.metric(
                            "Best Similarity",
                            f"{best_score:.3f}",
                        )

                    with col3:
                        st.metric(
                            "Average Similarity",
                            f"{avg_score:.3f}",
                        )

                    st.divider()

                    # -------------------------------------------------
                    # TOP-K SOURCES
                    # -------------------------------------------------

                    st.subheader(
                        "📚 RAG Sources / Top-K Retrieved Chunks"
                    )

                    for item in retrieved:

                        similarity_pct = (
                            item["similarity"] * 100
                        )

                        with st.expander(
                            f"Source {item['rank']}  |  "
                            f"Similarity: {item['similarity']:.4f} "
                            f"({similarity_pct:.1f}%)",
                            expanded=(
                                item["rank"] == 1
                            ),
                        ):

                            st.markdown(
                                f"""
                                **Rank:** {item['rank']}  
                                **Similarity Score:** {item['similarity']:.4f}  
                                **Distance:** {item['distance']:.4f}
                                """
                            )

                            st.progress(
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        item["similarity"],
                                    ),
                                )
                            )

                            st.markdown(
                                "**Retrieved Chunk:**"
                            )

                            st.code(
                                item["document"],
                                language="text",
                            )

                    st.divider()

                    # -------------------------------------------------
                    # GROUNDED ANSWER
                    # -------------------------------------------------

                    with st.spinner(
                        "Generating grounded answer..."
                    ):

                        answer = generate_grounded_answer(
                            query=query,
                            retrieved_chunks=retrieved,
                        )

                    st.session_state.last_answer = answer

                    st.subheader(
                        "🎯 Grounded Answer"
                    )

                    st.markdown(
                        '<div class="grounded-answer">',
                        unsafe_allow_html=True,
                    )

                    st.markdown(answer)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        "Answer generated using retrieved "
                        "knowledge-base context only."
                    )

            except Exception as exc:

                st.error(
                    f"RAG execution failed: {exc}"
                )

                with st.expander(
                    "Technical error details"
                ):

                    st.exception(exc)


# =====================================================================
# STRUCTURED EXTRACTION
# =====================================================================

with tab_extract:

    st.header("🧠 Structured Support Ticket Extraction")

    st.write(
        "Convert an unstructured support ticket into a "
        "Pydantic-validated structured object."
    )

    ticket_text = st.text_area(
        "Support Ticket",
        placeholder=(
            "Example: My payment was charged twice and "
            "I need an urgent refund."
