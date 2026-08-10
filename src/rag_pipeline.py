# src/rag_pipeline.py

from pathlib import Path
import os

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_PATH = (
    BASE_DIR / "data" / "knowledge_base.txt"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "technical_knowledge"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 5

GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured. "
            "Add it to your .env file or Streamlit Secrets."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():

    if not KNOWLEDGE_BASE_PATH.exists():

        raise FileNotFoundError(
            f"Knowledge base not found: "
            f"{KNOWLEDGE_BASE_PATH}"
        )

    return KNOWLEDGE_BASE_PATH.read_text(
        encoding="utf-8"
    )


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start = end - overlap

        if start < 0:
            start = 0

    return chunks


# ============================================================
# CHROMADB
# ============================================================

def create_vector_store(
    chunks,
    embedding_model,
):

    client = chromadb.Client()

    # Delete old collection if it exists
    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME
    )

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
    ).tolist()

    ids = [
        f"chunk-{index}"
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
    )

    return client, collection


# ============================================================
# BUILD RAG INDEX
# ============================================================

def build_rag_index():

    text = load_knowledge_base()

    chunks = chunk_text(text)

    embedding_model = load_embedding_model()

    client, collection = create_vector_store(
        chunks,
        embedding_model,
    )

    return (
        client,
        collection,
        embedding_model,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    collection,
    embedding_model,
    top_k: int = TOP_K,
):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    return documents, distances


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_prompt(
    query: str,
    documents,
):

    context = "\n\n".join(
        [
            f"[Source {index + 1}]\n{document}"
            for index, document
            in enumerate(documents)
        ]
    )

    return f"""
You are an internal technical knowledge assistant.

Answer the user's question using ONLY the
provided knowledge-base context.

Do not invent information.

If the answer cannot be found in the
provided context, clearly state:

"I could not find sufficient information
in the knowledge base."

Always provide a concise and technically
accurate answer.

User Question:
{query}

Knowledge Base Context:
{context}
"""


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def answer_question(
    query: str,
    collection=None,
    embedding_model=None,
):

    if not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    # Build index automatically if not supplied
    if collection is None or embedding_model is None:

        (
            _,
            collection,
            embedding_model,
        ) = build_rag_index()

    documents, distances = retrieve_documents(
        query=query,
        collection=collection,
        embedding_model=embedding_model,
    )

    if not documents:

        return {
            "answer": (
                "I could not find relevant "
                "information in the knowledge base."
            ),
            "sources": [],
            "distances": [],
        }

    prompt = build_prompt(
        query,
        documents,
    )

    gemini_client = get_gemini_client()

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    answer = response.text.strip()

    sources = [
        {
            "source": f"Knowledge Base Chunk {index + 1}",
            "content": document,
            "distance": float(
                distances[index]
            )
            if index < len(distances)
            else None,
        }
        for index, document
        in enumerate(documents)
    ]

    return {
        "answer": answer,
        "sources": sources,
        "distances": distances,
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    result = answer_question(
        "How should a critical security incident be handled?"
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            f"\n{source['source']}"
        )

        print(
            source["content"]
  )
