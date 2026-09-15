from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path


class VectorStore:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the embedding model and FAISS vector store.
        """

        print("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

        self.index = None
        self.documents = []


    def create_embeddings(self, documents):
        """
        Convert document chunks into numerical embeddings.
        """

        texts = [doc.page_content for doc in documents]

        print(f"Creating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # FAISS requires float32 vectors
        embeddings = embeddings.astype("float32")

        return embeddings


    def build_index(self, documents):
        """
        Create a FAISS index from document chunks.
        """

        embeddings = self.create_embeddings(documents)

        # Get embedding dimension
        dimension = embeddings.shape[1]

        print(f"Embedding dimension: {dimension}")

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)

        # Add embeddings to FAISS
        self.index.add(embeddings)

        # Keep original documents so we can retrieve
        # their text and metadata later
        self.documents = documents

        print(
            f"FAISS index created with "
            f"{self.index.ntotal} vectors."
        )


    def search(self, query, top_k=5):
        """
        Search FAISS for the most relevant document chunks.

        Returns:
            A list containing:
            - document
            - similarity distance
            - source
            - page
        """

        if self.index is None:
            raise ValueError(
                "FAISS index has not been created."
            )

        # Convert the user's question into an embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype("float32")

        # Search FAISS
        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            # FAISS uses -1 when no result exists
            if index == -1:
                continue

            document = self.documents[index]

            results.append({
                "document": document,
                "distance": float(distance),
                "source": document.metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": document.metadata.get(
                    "page",
                    "Unknown"
                )
            })

        return results


    def save(
        self,
        folder="vector_store/saved_index"
    ):
        """
        Save FAISS index and document metadata.
        """

        folder = Path(folder)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save FAISS index
        faiss.write_index(
            self.index,
            str(folder / "index.faiss")
        )

        # Save documents and metadata
        with open(
            folder / "documents.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.documents,
                f
            )

        print(
            f"Vector store saved to: {folder}"
        )


    def load(
        self,
        folder="vector_store/saved_index"
    ):
        """
        Load an existing FAISS index
        and document metadata.
        """

        folder = Path(folder)

        index_path = folder / "index.faiss"
        documents_path = folder / "documents.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                "FAISS index not found."
            )

        if not documents_path.exists():
            raise FileNotFoundError(
                "Document metadata file not found."
            )

        # Load FAISS index
        self.index = faiss.read_index(
            str(index_path)
        )

        # Load documents
        with open(
            documents_path,
            "rb"
        ) as f:

            self.documents = pickle.load(f)

        print(
            f"Vector store loaded with "
            f"{self.index.ntotal} vectors."
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    from document_loader import (
        load_and_split_documents
    )

    print("\n")
    print("=" * 50)
    print("STARTING VECTOR STORE CREATION")
    print("=" * 50)


    # --------------------------------------------------------
    # STEP 1: Load and split PDF documents
    # --------------------------------------------------------

    chunks = load_and_split_documents(
        "documents"
    )

    print(
        f"\nTotal chunks created: {len(chunks)}"
    )


    # --------------------------------------------------------
    # STEP 2: Create Vector Store
    # --------------------------------------------------------

    vector_store = VectorStore()


    # --------------------------------------------------------
    # STEP 3: Create FAISS index
    # --------------------------------------------------------

    vector_store.build_index(
        chunks
    )


    # --------------------------------------------------------
    # STEP 4: Save Vector Store
    # --------------------------------------------------------

    vector_store.save()


    print("\n")
    print("=" * 50)
    print("VECTOR STORE CREATION COMPLETE")
    print("=" * 50)


    # ========================================================
    # RETRIEVAL TESTING
    # ========================================================

    print("\n")
    print("=" * 50)
    print("TESTING FAISS RETRIEVAL")
    print("=" * 50)


    test_questions = [

        "What is Retrieval-Augmented Generation?",

        "What is self-attention?",

        "What is prompt engineering?",

        "What is a Large Language Model?"

    ]


    # --------------------------------------------------------
    # Test every question
    # --------------------------------------------------------

    for question in test_questions:

        print("\n")
        print("-" * 50)
        print(f"QUESTION: {question}")
        print("-" * 50)


        # Retrieve top 3 chunks
        results = vector_store.search(
            question,
            top_k=3
        )


        # Display results
        for i, result in enumerate(
            results,
            start=1
        ):

            document = result["document"]

            distance = result["distance"]

            source = result["source"]

            page = result["page"]


            print(f"\nResult {i}")

            print(
                f"Source: {source}"
            )

            print(
                f"Page: {page}"
            )

            print(
                f"Distance: {distance:.4f}"
            )

            print("Text:")

            # Display first 300 characters
            text = document.page_content

            text = text.replace(
                "\n",
                " "
            )

            print(
                text[:300]
            )


    print("\n")
    print("=" * 50)
    print("RETRIEVAL TESTING COMPLETE")
    print("=" * 50)