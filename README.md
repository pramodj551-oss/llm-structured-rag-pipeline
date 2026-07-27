# 🧠 LLM-Powered Structured Insight & RAG Retrieval Pipeline

A production-grade NLP pipeline combining Pydantic-validated structured extraction with a local Retrieval-Augmented Generation (RAG) system grounded in internal technical documentation.

---

## 1. Architecture & Tool Selection

| Layer | Technology | Reason |
| :--- | :--- | :--- |
| **LLM Inference** | `gemini-2.5-flash` | Free, keyless tier via Google AI Studio; fast structured outputs via native schema enforcement. |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Open-source, no API calls or keys required; lightweight and accurate. |
| **Vector Database** | `ChromaDB` (in-memory) | Zero-overhead, embedded, keyless local vector store — ideal for prototyping and evaluation. |
| **Schema Validation** | `Pydantic v2` | Runtime validation guaranteeing type safety and enum constraints on LLM outputs. |

---

## 2. Structured Extraction & Validation

All incoming unstructured support tickets are validated against `ExtractedSupportTicket`, defined in `src/schemas.py`.

### Validation Failure Example

A synthetic malformed fixture with an invalid `urgency` value (`SUPER_URGENT`) was passed to the validator to confirm enforcement:

```json
{
  "ticket_id": "TCK-INVALID-01",
  "category": "Billing",
  "urgency": "SUPER_URGENT",
  "sentiment": "Negative",
  "one_line_summary": "User wants a refund."
}
```

**Result:** Pydantic raised a `ValidationError` on the `urgency` field, since `SUPER_URGENT` falls outside the allowed enum values (`Low`, `Medium`, `High`, `Critical`). The ticket was rejected before entering the pipeline, confirming schema enforcement works as intended.

---

## 3. RAG Retrieval Workflow

1. Internal technical documentation is chunked and embedded using `all-MiniLM-L6-v2`.
2. Embeddings are stored in a local `ChromaDB` collection.
3. At query time, the most relevant chunks are retrieved via similarity search.
4. Retrieved context is passed to `gemini-2.5-flash` to generate a grounded, source-backed response.

---

## 4. Repository Structure

```
llm-structured-rag-pipeline/
├── data/
│   ├── support_tickets.json     # 15+ uncleaned text records
│   └── knowledge_base.txt       # 2,000+ word corpus for RAG
├── src/
│   ├── schemas.py                # Pydantic schema definitions
│   ├── extraction_pipeline.py    # Structured LLM extraction + validation
│   └── rag_pipeline.py           # Chunking, embedding, vector DB & Q&A
├── notebooks/
│   └── demo_notebook.ipynb       # Visual transcript and query runs
├── requirements.txt               # Dependencies
└── README.md                      # Tool choices, error handling, demo transcript
```

---

## 5. Setup

```bash
git clone https://github.com/pramodj551-oss/llm-structured-rag-pipeline.git
cd llm-structured-rag-pipeline
pip install -r requirements.txt
```

---

## 📬 Contact

**Pramod Prakash Jadhav**
📧 pramodj551@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281)
