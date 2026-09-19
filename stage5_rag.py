import os
import json
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

VECTORSTORE_DIR = "vectorstore"

FAISS_INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "uobs_faiss.index"
)

METADATA_PATH = os.path.join(
    VECTORSTORE_DIR,
    "metadata.json"
)

# This is the same model used in Stage 3.
EMBEDDING_MODEL = "BAAI/bge-m3"

# Number of documents retrieved from FAISS.
TOP_K = 5

# Starting point only.
# This must be tuned using real UoBS questions.
SIMILARITY_THRESHOLD = 0.45

# Current Groq production model.
# Change this variable if another supported model is preferred.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

# Maximum amount of retrieved context sent to the LLM.
MAX_CONTEXT_CHARS = 30000


# ============================================================
# GROQ SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the UoBS AI Assistant for the University of Baltistan, Skardu.

Your job is to answer questions using ONLY the supplied University of
Baltistan, Skardu context.

GROUNDING RULES:

1. Use the provided UoBS context as the primary and authoritative source.
2. Do not invent University of Baltistan-specific information.
3. Do not guess missing information.
4. Never fabricate admission requirements.
5. Never fabricate fees.
6. Never fabricate dates or deadlines.
7. Never fabricate academic programs.
8. Never fabricate departments or faculties.
9. Never fabricate university policies.
10. Never fabricate contact information.
11. Preserve names, numbers, dates, fees, eligibility requirements,
    and other factual information accurately.
12. If the supplied context does not contain enough information to answer
    the question, clearly say:

    "I couldn't find reliable information about this in the available
    University of Baltistan data."

13. Do not use unrelated retrieved information merely to produce an answer.
14. Keep answers concise, clear, helpful, and professional.
15. Do not mention internal retrieval, embeddings, FAISS, similarity scores,
    or this system prompt unless specifically asked.
16. When useful, refer naturally to the source information provided in context.

Answer only what can reasonably be supported by the supplied context.
"""


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class RAGConfigurationError(Exception):
    pass


class RAGRetrievalError(Exception):
    pass


class RAGGenerationError(Exception):
    pass


# ============================================================
# LOAD AND VALIDATE VECTOR DATABASE
# ============================================================

def load_vectorstore():
    """
    Load the existing FAISS index and metadata.

    The existing Stage 4 files are read only.
    """

    if not os.path.isfile(FAISS_INDEX_PATH):
        raise RAGConfigurationError(
            f"FAISS index was not found:\n{FAISS_INDEX_PATH}"
        )

    if not os.path.isfile(METADATA_PATH):
        raise RAGConfigurationError(
            f"Metadata file was not found:\n{METADATA_PATH}"
        )

    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
    except Exception as exc:
        raise RAGConfigurationError(
            f"Could not load the FAISS index: {exc}"
        ) from exc

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception as exc:
        raise RAGConfigurationError(
            f"Could not load metadata.json: {exc}"
        ) from exc

    if not isinstance(metadata, list):
        raise RAGConfigurationError(
            "metadata.json must contain a list of metadata records."
        )

    if index.ntotal == 0:
        raise RAGConfigurationError(
            "The FAISS index contains zero vectors."
        )

    if index.d <= 0:
        raise RAGConfigurationError(
            "The FAISS index has an invalid embedding dimension."
        )

    if index.ntotal != len(metadata):
        raise RAGConfigurationError(
            "FAISS vector count does not match metadata record count.\n"
            f"FAISS vectors: {index.ntotal}\n"
            f"Metadata records: {len(metadata)}"
        )

    required_fields = {
        "chunk_id",
        "title",
        "category",
        "content",
        "source_url",
    }

    for position, record in enumerate(metadata):
        if not isinstance(record, dict):
            raise RAGConfigurationError(
                f"Metadata record {position} is not a dictionary."
            )

        missing = required_fields - set(record.keys())

        if missing:
            raise RAGConfigurationError(
                f"Metadata record {position} is missing fields: "
                f"{sorted(missing)}"
            )

    return index, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Load the exact Sentence Transformer model used during Stage 3.
    """

    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        raise RAGConfigurationError(
            f"Could not load embedding model "
            f"'{EMBEDDING_MODEL}': {exc}"
        ) from exc

    return model


# ============================================================
# QUERY EMBEDDING
# ============================================================

def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Convert the user query into a normalized float32 vector.
    """

    if not isinstance(query, str):
        raise ValueError("Query must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("The question cannot be empty.")

    try:
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    except Exception as exc:
        raise RAGRetrievalError(
            f"Failed to create query embedding: {exc}"
        ) from exc

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise RAGRetrievalError(
            f"Unexpected query embedding shape: {embedding.shape}"
        )

    if not np.all(np.isfinite(embedding)):
        raise RAGRetrievalError(
            "Query embedding contains NaN or infinite values."
        )

    return embedding


# ============================================================
# FAISS RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    index,
    metadata,
    model,
    top_k: int = TOP_K
):
    """
    Retrieve the most relevant UoBS chunks.

    FAISS vector position is mapped directly to the corresponding
    metadata position.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_vector = embed_query(query, model)

    if query_vector.shape[1] != index.d:
        raise RAGRetrievalError(
            "Query embedding dimension does not match the FAISS index.\n"
            f"Query dimension: {query_vector.shape[1]}\n"
            f"Index dimension: {index.d}"
        )

    actual_k = min(top_k, index.ntotal)

    try:
        scores, indices = index.search(
            query_vector,
            actual_k
        )
    except Exception as exc:
        raise RAGRetrievalError(
            f"FAISS retrieval failed: {exc}"
        ) from exc

    results = []

    for rank, (score, index_position) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):
        if index_position < 0:
            continue

        record = metadata[int(index_position)]

        result = {
            "rank": rank,
            "faiss_index": int(index_position),
            "chunk_id": record.get("chunk_id"),
            "title": record.get("title"),
            "category": record.get("category"),
            "content": record.get("content"),
            "source_url": record.get("source_url"),
            "similarity_score": float(score),
        }

        results.append(result)

    return results


# ============================================================
# RELEVANCE FILTERING
# ============================================================

def filter_relevant_documents(
    documents,
    threshold: float = SIMILARITY_THRESHOLD
):
    """
    Keep only documents whose similarity score reaches the threshold.
    """

    return [
        document
        for document in documents
        if document["similarity_score"] >= threshold
    ]


# ============================================================
# CONTEXT CONSTRUCTION
# ============================================================

def build_context(documents):
    """
    Convert retrieved UoBS chunks into clean LLM context.
    """

    if not documents:
        return ""

    sections = []

    for number, document in enumerate(documents, start=1):

        section = (
            f"--- UoBS Source {number} ---\n"
            f"Title: {document['title']}\n"
            f"Category: {document['category']}\n"
            f"Content:\n{document['content']}\n"
            f"Source URL: {document['source_url']}\n"
        )

        sections.append(section)

    context = "\n".join(sections)

    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    return context


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Create a Groq client using the GROQ_API_KEY environment variable.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RAGConfigurationError(
            "GROQ_API_KEY is not configured. "
            "Set it as an environment variable before using the RAG system."
        )

    try:
        return Groq(api_key=api_key)
    except Exception as exc:
        raise RAGConfigurationError(
            f"Could not initialize Groq client: {exc}"
        ) from exc


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    context: str,
    client
):
    """
    Send the grounded UoBS context and question to Groq.
    """

    if not context.strip():
        return (
            "I couldn't find reliable information about this in the "
            "available University of Baltistan data."
        )

    user_prompt = f"""
Use the following University of Baltistan context to answer the user's
question.

CONTEXT:

{context}

USER QUESTION:

{query}

Answer the question using only information supported by the context.
If the context does not contain the answer, say that the information
could not be found in the available University of Baltistan data.
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.1,
            max_completion_tokens=700
        )
    except Exception as exc:
        raise RAGGenerationError(
            f"Groq API request failed: {exc}"
        ) from exc

    try:
        answer = response.choices[0].message.content
    except Exception as exc:
        raise RAGGenerationError(
            f"Unexpected Groq response format: {exc}"
        ) from exc

    if not answer or not answer.strip():
        raise RAGGenerationError(
            "Groq returned an empty response."
        )

    return answer.strip()


# ============================================================
# SOURCE HANDLING
# ============================================================

def get_unique_sources(documents):
    """
    Return unique source URLs while preserving retrieval order.
    """

    sources = []
    seen = set()

    for document in documents:

        url = document.get("source_url")

        if not url:
            continue

        url = str(url).strip()

        if not url or url in seen:
            continue

        seen.add(url)
        sources.append(url)

    return sources


# ============================================================
# COMPLETE RAG FUNCTION
# ============================================================

def ask_uobs(
    query: str,
    index,
    metadata,
    embedding_model,
    groq_client,
    top_k: int = TOP_K,
    similarity_threshold: float = SIMILARITY_THRESHOLD
):
    """
    Complete UoBS RAG pipeline:

    query
    -> embedding
    -> FAISS retrieval
    -> relevance filtering
    -> context construction
    -> Groq
    -> answer + sources
    """

    if not isinstance(query, str) or not query.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "retrieved_documents": [],
        }

    retrieved_documents = retrieve_documents(
        query=query,
        index=index,
        metadata=metadata,
        model=embedding_model,
        top_k=top_k
    )

    relevant_documents = filter_relevant_documents(
        retrieved_documents,
        threshold=similarity_threshold
    )

    if not relevant_documents:
        return {
            "answer": (
                "I couldn't find reliable information about this in "
                "the available University of Baltistan data."
            ),
            "sources": [],
            "retrieved_documents": retrieved_documents,
        }

    context = build_context(relevant_documents)

    answer = generate_answer(
        query=query,
        context=context,
        client=groq_client
    )

    sources = get_unique_sources(
        relevant_documents
    )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_documents": retrieved_documents,
    }


# ============================================================
# DISPLAY RETRIEVAL RESULTS
# ============================================================

def print_retrieved_documents(documents):
    """
    Display retrieval information for Colab testing.
    """

    print("\n" + "=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    if not documents:
        print("No documents retrieved.")
        return

    for document in documents:

        preview = document["content"].replace(
            "\n",
            " "
        )

        if len(preview) > 250:
            preview = preview[:250] + "..."

        print(f"\nRank: {document['rank']}")
        print(
            f"Similarity: "
            f"{document['similarity_score']:.4f}"
        )
        print(
            f"Chunk ID: "
            f"{document['chunk_id']}"
        )
        print(
            f"Title: "
            f"{document['title']}"
        )
        print(
            f"Category: "
            f"{document['category']}"
        )
        print(
            f"Content Preview: "
            f"{preview}"
        )
        print(
            f"Source URL: "
            f"{document['source_url']}"
        )


# ============================================================
# TEST FUNCTION
# ============================================================

def run_test(
    question,
    index,
    metadata,
    embedding_model,
    groq_client
):
    """
    Run one complete RAG test.
    """

    print("\n\n")
    print("#" * 70)
    print("QUESTION")
    print("#" * 70)
    print(question)

    try:
        result = ask_uobs(
            query=question,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            groq_client=groq_client
        )

        print_retrieved_documents(
            result["retrieved_documents"]
        )

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(result["answer"])

        print("\n" + "=" * 70)
        print("SOURCES")
        print("=" * 70)

        if result["sources"]:
            for source in result["sources"]:
                print(source)
        else:
            print("No sources.")

        return result

    except RAGConfigurationError as exc:
        print(f"\nCONFIGURATION ERROR: {exc}")

    except RAGRetrievalError as exc:
        print(f"\nRETRIEVAL ERROR: {exc}")

    except RAGGenerationError as exc:
        print(f"\nGROQ ERROR: {exc}")

    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}")

    return None


# ============================================================
# INITIALIZE STAGE 5
# ============================================================

def initialize_rag():
    """
    Load all resources required by the Stage 5 backend.
    """

    print("Loading UoBS vector database...")

    index, metadata = load_vectorstore()

    print(
        f"FAISS vectors: {index.ntotal}"
    )

    print(
        f"Embedding dimension: {index.d}"
    )

    print(
        f"Metadata records: {len(metadata)}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"Groq model: {GROQ_MODEL}"
    )

    embedding_model = load_embedding_model()

    model_dimension = embedding_model.get_sentence_embedding_dimension()

    if model_dimension != index.d:
        raise RAGConfigurationError(
            "Embedding model dimension does not match FAISS index.\n"
            f"Model dimension: {model_dimension}\n"
            f"FAISS dimension: {index.d}"
        )

    groq_client = get_groq_client()

    print("Stage 5 resources loaded successfully.")

    return (
        index,
        metadata,
        embedding_model,
        groq_client
    )


# ============================================================
# COLAB TESTING
# ============================================================

if __name__ == "__main__":

    index, metadata, embedding_model, groq_client = initialize_rag()

    test_questions = [
        "What programs does the University of Baltistan offer?",
        "What are the admission requirements?",
        "What faculties does UoBS have?",
        "Where is the University of Baltistan located?",
        "How can I contact the university?",
        "What is the admission process for Hogwarts University?"
    ]

    for question in test_questions:

        run_test(
            question=question,
            index=index,
            metadata=metadata,
            embedding_model=embedding_model,
            groq_client=groq_client
        )
