import os
from dotenv import load_dotenv
from google import genai

from document_loader import load_and_split_documents
from vector_store import VectorStore
from prompt import create_rag_prompt


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please add it to your .env file."
    )


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# CREATE RAG SYSTEM
# ============================================================

def create_rag_system():

    print("Loading documents...")

    chunks = load_and_split_documents("documents")

    print(f"Loaded {len(chunks)} document chunks.")

    print("Creating vector store...")

    vector_store = VectorStore()

    vector_store.build_index(chunks)

    print("Vector store created successfully.")

    return vector_store


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, vector_store, top_k=3):

    # --------------------------------------------------------
    # Step 1: Retrieve relevant document chunks
    # --------------------------------------------------------

    results = vector_store.search(
        question,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Step 2: Extract retrieved documents
    # --------------------------------------------------------

    retrieved_documents = [
        result["document"]
        for result in results
    ]

    # --------------------------------------------------------
    # Step 3: Create RAG prompt
    # --------------------------------------------------------

    prompt = create_rag_prompt(
        question,
        retrieved_documents
    )

    # --------------------------------------------------------
    # Step 4: Send context + question to Gemini
    # --------------------------------------------------------

    response = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt
    )

    # --------------------------------------------------------
    # Step 5: Extract Gemini answer
    # --------------------------------------------------------

    answer = response.output_text

    return answer, results


# ============================================================
# TEST THE RAG SYSTEM
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("GENERATIVE AI & LLM KNOWLEDGE ASSISTANT")
    print("=" * 60)

    # --------------------------------------------------------
    # Create RAG system
    # --------------------------------------------------------

    vector_store = create_rag_system()

    print("\nRAG system ready!")

    # --------------------------------------------------------
    # Test questions
    # --------------------------------------------------------

    questions = [
        "What is Retrieval-Augmented Generation?",
        "What is self-attention?",
        "What is prompt engineering?",
        "What is a Large Language Model?"
    ]

    # --------------------------------------------------------
    # Ask questions
    # --------------------------------------------------------

    for question in questions:

        print("\n")
        print("-" * 60)
        print(f"QUESTION: {question}")
        print("-" * 60)

        answer, results = generate_answer(
            question,
            vector_store
        )

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        print("\nANSWER:")
        print(answer)

        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        print("\nSOURCES:")

        for result in results:

            print(
                f"- {result['source']} "
                f"(Page {result['page']})"
            )

    print("\n")
    print("=" * 60)
    print("RAG TEST COMPLETE")
    print("=" * 60)