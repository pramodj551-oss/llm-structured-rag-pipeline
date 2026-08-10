# app.py

import json
from pathlib import Path

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LLM Structured Insight & RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid rgba(128,128,128,0.25);
        }

        .success-box {
            padding: 1rem;
            border-radius: 8px;
            background-color: rgba(34, 197, 94, 0.10);
            border: 1px solid rgba(34, 197, 94, 0.30);
        }

        .error-box {
            padding: 1rem;
            border-radius: 8px;
            background-color: rgba(239, 68, 68, 0.10);
            border: 1px solid rgba(239, 68, 68, 0.30);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SAFE IMPORTS
# ============================================================

@st.cache_resource
def load_pipeline_modules():
    """
    Import pipeline modules lazily so the dashboard can display
    a useful error instead of crashing during startup.
    """

    try:
        from src.schemas import ExtractedSupportTicket

        return {
            "schema": ExtractedSupportTicket,
            "extraction": __import__(
                "src.extraction_pipeline",
                fromlist=["*"],
            ),
            "rag": __import__(
                "src.rag_pipeline",
                fromlist=["*"],
            ),
            "error": None,
        }

    except Exception as exc:
        return {
            "schema": None,
            "extraction": None,
            "rag": None,
            "error": str(exc),
        }


modules = load_pipeline_modules()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 AI Assistant")

    st.caption("LLM-Powered Structured Insight & RAG")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🎫 Ticket Extraction",
            "🔎 RAG Q&A",
            "📚 Knowledge Base",
            "🧪 Schema Validation",
            "ℹ️ About",
        ],
    )

    st.divider()

    st.markdown("### Technology")

    st.markdown(
        """
        - **Gemini 2.5 Flash**
        - **Pydantic v2**
        - **Sentence Transformers**
        - **ChromaDB**
        - **Streamlit**
        """
    )


# ============================================================
# HEADER
# ============================================================

def render_header():
    st.markdown(
        '<div class="main-title">🧠 LLM-Powered Structured Insight & RAG</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        "Structured support-ticket extraction + grounded technical Q&A"
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    render_header()

    st.info(
        "This application combines Pydantic-validated structured "
        "LLM extraction with a local Retrieval-Augmented Generation "
        "(RAG) pipeline."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("LLM", "Gemini 2.5 Flash")

    with col2:
        st.metric("Embeddings", "MiniLM-L6-v2")

    with col3:
        st.metric("Vector DB", "ChromaDB")

    with col4:
        st.metric("Validation", "Pydantic v2")

    st.divider()

    st.subheader("🚀 Pipeline")

    steps = [
        ("1", "Raw Support Ticket", "Unstructured customer text"),
        ("2", "LLM Extraction", "Gemini extracts structured fields"),
        ("3", "Pydantic Validation", "Schema + enum enforcement"),
        ("4", "RAG Retrieval", "Relevant technical context"),
        ("5", "Grounded Answer", "Source-backed response"),
    ]

    for number, title, description in steps:
        c1, c2, c3 = st.columns([1, 3, 6])

        with c1:
            st.markdown(f"### {number}")

        with c2:
            st.markdown(f"**{title}**")

        with c3:
            st.write(description)


# ============================================================
# TICKET EXTRACTION
# ============================================================

elif page == "🎫 Ticket Extraction":

    render_header()

    st.subheader("🎫 Structured Support Ticket Extraction")

    st.write(
        "Enter an unstructured support ticket and extract a "
        "validated structured representation."
    )

    default_ticket = (
        "I was charged twice for my monthly subscription. "
        "I only purchased the plan once and would like one of "
        "the charges refunded. This is very frustrating."
    )

    ticket_text = st.text_area(
        "Support Ticket",
        value=default_ticket,
        height=180,
        placeholder="Enter support ticket text...",
    )

    if st.button(
        "🚀 Extract & Validate",
        type="primary",
        use_container_width=True,
    ):

        if not ticket_text.strip():
            st.warning("Please enter a support ticket.")
            st.stop()

        if modules["error"]:
            st.error(
                "Unable to load the extraction pipeline.\n\n"
                f"{modules['error']}"
            )
            st.stop()

        extraction_module = modules["extraction"]

        try:

            # Try commonly used function names.
            extraction_function = None

            for function_name in [
                "extract_support_ticket",
                "extract_ticket",
                "extract_structured_ticket",
                "run_extraction",
            ]:

                if hasattr(extraction_module, function_name):
                    extraction_function = getattr(
                        extraction_module,
                        function_name,
                    )
                    break

            if extraction_function is None:

                st.error(
                    "No supported extraction function was found in "
                    "`src/extraction_pipeline.py`.\n\n"
                    "Expected one of:\n"
                    "- extract_support_ticket\n"
                    "- extract_ticket\n"
                    "- extract_structured_ticket\n"
                    "- run_extraction"
                )

                st.stop()

            with st.spinner("Analyzing ticket with Gemini..."):

                result = extraction_function(ticket_text)

            st.success("Ticket successfully extracted and validated.")

            st.subheader("📋 Structured Output")

            if hasattr(result, "model_dump"):

                result_dict = result.model_dump()

            elif isinstance(result, dict):

                result_dict = result

            else:

                result_dict = {
                    "result": str(result)
                }

            st.json(result_dict)

            st.subheader("📊 Extracted Fields")

            if isinstance(result_dict, dict):

                columns = st.columns(
                    max(1, min(len(result_dict), 4))
                )

                for index, (key, value) in enumerate(
                    result_dict.items()
                ):

                    with columns[index % len(columns)]:

                        st.metric(
                            key.replace("_", " ").title(),
                            str(value),
                        )

        except Exception as exc:

            st.error(
                "❌ Extraction failed.\n\n"
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# RAG Q&A
# ============================================================

elif page == "🔎 RAG Q&A":

    render_header()

    st.subheader("🔎 Technical Knowledge Assistant")

    st.write(
        "Ask a question about the internal technical documentation. "
        "The system retrieves relevant chunks before generating an answer."
    )

    query = st.text_area(
        "Your Question",
        height=120,
        placeholder="Example: How should a high-priority incident be escalated?",
    )

    if st.button(
        "🔍 Search Knowledge Base",
        type="primary",
        use_container_width=True,
    ):

        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        if modules["error"]:
            st.error(
                "Unable to load the RAG pipeline.\n\n"
                f"{modules['error']}"
            )
            st.stop()

        rag_module = modules["rag"]

        try:

            rag_function = None

            for function_name in [
                "answer_question",
                "query_rag",
                "rag_query",
                "ask_question",
                "run_rag",
            ]:

                if hasattr(rag_module, function_name):
                    rag_function = getattr(
                        rag_module,
                        function_name,
                    )
                    break

            if rag_function is None:

                st.error(
                    "No supported RAG function was found in "
                    "`src/rag_pipeline.py`.\n\n"
                    "Expected one of:\n"
                    "- answer_question\n"
                    "- query_rag\n"
                    "- rag_query\n"
                    "- ask_question\n"
                    "- run_rag"
                )

                st.stop()

            with st.spinner(
                "Retrieving relevant documentation and generating answer..."
            ):

                result = rag_function(query)

            st.success("Answer generated successfully.")

            st.subheader("💡 Answer")

            if isinstance(result, dict):

                answer = (
                    result.get("answer")
                    or result.get("response")
                    or result.get("result")
                )

                if answer:

                    st.write(answer)

                else:

                    st.json(result)

            else:

                st.write(result)

        except Exception as exc:

            st.error(
                "❌ RAG query failed.\n\n"
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

elif page == "📚 Knowledge Base":

    render_header()

    st.subheader("📚 Internal Knowledge Base")

    knowledge_file = DATA_DIR / "knowledge_base.txt"

    if not knowledge_file.exists():

        st.warning(
            "`data/knowledge_base.txt` was not found."
        )

    else:

        try:

            text = knowledge_file.read_text(
                encoding="utf-8"
            )

            words = len(text.split())
            characters = len(text)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Words", f"{words:,}")

            with col2:
                st.metric("Characters", f"{characters:,}")

            with col3:
                st.metric(
                    "File",
                    "knowledge_base.txt",
                )

            st.divider()

            with st.expander(
                "📖 Preview Knowledge Base"
            ):

                st.text_area(
                    "Content",
                    text[:10000],
                    height=500,
                )

        except Exception as exc:

            st.error(
                f"Unable to read knowledge base: {exc}"
            )


# ============================================================
# SCHEMA VALIDATION
# ============================================================

elif page == "🧪 Schema Validation":

    render_header()

    st.subheader("🧪 Pydantic Schema Validation")

    st.write(
        "Test whether structured ticket data satisfies the "
        "`ExtractedSupportTicket` schema."
    )

    default_json = {
        "ticket_id": "TCK-INVALID-01",
        "category": "Billing",
        "urgency": "SUPER_URGENT",
        "sentiment": "Negative",
        "one_line_summary": "User wants a refund.",
    }

    json_input = st.text_area(
        "JSON Payload",
        value=json.dumps(
            default_json,
            indent=2,
        ),
        height=250,
    )

    if st.button(
        "🧪 Validate JSON",
        type="primary",
        use_container_width=True,
    ):

        if modules["schema"] is None:

            st.error(
                "Unable to load `ExtractedSupportTicket`."
            )

            st.stop()

        try:

            payload = json.loads(json_input)

        except json.JSONDecodeError as exc:

            st.error(
                f"Invalid JSON: {exc}"
            )

            st.stop()

        try:

            schema = modules["schema"]

            validated = schema.model_validate(payload)

            st.success(
                "✅ Validation successful. "
                "The payload conforms to the schema."
            )

            st.json(
                validated.model_dump()
            )

        except Exception as exc:

            st.error(
                "❌ Validation failed."
            )

            st.code(
                str(exc),
                language="text",
            )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    render_header()

    st.subheader("ℹ️ About This Project")

    st.markdown(
        """
        ### 🧠 LLM-Powered Structured Insight & RAG Retrieval Pipeline

        This project demonstrates a production-oriented NLP workflow
        combining:

        - Structured extraction using an LLM
        - Pydantic runtime validation
        - Semantic embeddings
        - Local vector retrieval
        - Retrieval-Augmented Generation
        - Grounded technical Q&A

        ### Architecture

        **Support Ticket**
        → **Gemini 2.5 Flash**
        → **Pydantic Validation**
        → **Structured Data**

        **Technical Documentation**
        → **Chunking**
        → **MiniLM Embeddings**
        → **ChromaDB**
        → **Similarity Search**
        → **Gemini**
        → **Grounded Answer**
        """
    )

    st.divider()

    st.markdown("### 👨‍💻 Developer")

    st.write("**Pramod Prakash Jadhav**")

    st.write(
        "AI/ML Developer | Security Analyst"
    )

    st.write(
        "📧 pramodj551@gmail.com"
    )

    st.write(
        "🔗 LinkedIn: "
        "pramod-prakash-jadhav-42ba2281"
    )

    st.divider()

    st.caption(
        "Built with Python • Streamlit • Gemini • "
        "Pydantic • Sentence Transformers • ChromaDB"
    )
