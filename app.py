import streamlit as st
import os

from document_loader import load_and_split_documents
from vector_store import VectorStore
from rag_pipeline import generate_answer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Generative AI & LLM Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 Generative AI & LLM Knowledge Assistant")

st.write(
    "Ask questions about Generative AI, Large Language Models, "
    "Transformers, Prompt Engineering, and RAG using the uploaded documents."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

REFUSAL_MESSAGE = (
    "I could not find this information in the uploaded documents."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📄 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "🔄 Process Documents",
        use_container_width=True
    )

    if process_button:

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            os.makedirs("documents", exist_ok=True)

            # --------------------------------------------------
            # Save uploaded PDFs
            # --------------------------------------------------

            for uploaded_file in uploaded_files:

                file_path = os.path.join(
                    "documents",
                    uploaded_file.name
                )

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            try:

                with st.spinner(
                    "Reading and processing documents..."
                ):

                    # --------------------------------------------------
                    # Extract and split documents
                    # --------------------------------------------------

                    chunks = load_and_split_documents(
                        "documents"
                    )

                    # --------------------------------------------------
                    # Create vector store
                    # --------------------------------------------------

                    vector_store = VectorStore()

                    vector_store.build_index(chunks)

                    # --------------------------------------------------
                    # Save vector store in session
                    # --------------------------------------------------

                    st.session_state.vector_store = vector_store

                    st.session_state.documents_processed = True

                    # Clear previous conversation
                    st.session_state.messages = []

                st.success(
                    f"Successfully processed {len(uploaded_files)} PDF(s) "
                    f"and created {len(chunks)} chunks."
                )

            except Exception as e:

                st.session_state.documents_processed = False

                st.error(
                    f"Document processing failed: {str(e)}"
                )


    # --------------------------------------------------
    # PROCESSING STATUS
    # --------------------------------------------------

    st.divider()

    if st.session_state.documents_processed:

        st.success(
            "✅ Documents are ready!"
        )

    else:

        st.info(
            "📌 Upload and process your PDFs to begin."
        )


    # --------------------------------------------------
    # ABOUT
    # --------------------------------------------------

    st.divider()

    st.header("ℹ️ About")

    st.write(
        "This chatbot uses Retrieval-Augmented Generation (RAG) "
        "to retrieve relevant information from uploaded PDF documents "
        "before generating an answer."
    )

    st.warning(
        "AI-generated answers may not always be correct. "
        "Verify important information independently."
    )


# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # --------------------------------------------------
        # Display sources only if available
        # --------------------------------------------------

        if message.get("sources"):

            st.markdown(
                "**📚 Sources:**"
            )

            for source in message["sources"]:

                st.write(
                    f"- {source['source']} "
                    f"(Page {source['page']})"
                )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # --------------------------------------------------
    # Check whether documents are processed
    # --------------------------------------------------

    if (
        not st.session_state.documents_processed
        or st.session_state.vector_store is None
    ):

        st.warning(
            "Please upload and process your PDF documents first."
        )

    else:

        # --------------------------------------------------
        # USER MESSAGE
        # --------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                question
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # --------------------------------------------------
        # ASSISTANT RESPONSE
        # --------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                try:

                    # --------------------------------------------------
                    # Generate RAG answer
                    # --------------------------------------------------

                    answer, results = generate_answer(
                        question,
                        st.session_state.vector_store,
                        top_k=3
                    )

                    # --------------------------------------------------
                    # Display answer
                    # --------------------------------------------------

                    st.markdown(
                        answer
                    )


                    # --------------------------------------------------
                    # Handle refusal response
                    # --------------------------------------------------

                    if answer.strip() == REFUSAL_MESSAGE:

                        # Don't show sources because the answer
                        # was not found in the documents.

                        sources_to_save = []

                    else:

                        # --------------------------------------------------
                        # Remove duplicate sources
                        # --------------------------------------------------

                        unique_sources = []

                        seen_sources = set()

                        for result in results:

                            source_key = (
                                result["source"],
                                result["page"]
                            )

                            if source_key not in seen_sources:

                                seen_sources.add(
                                    source_key
                                )

                                unique_sources.append(
                                    {
                                        "source": result["source"],
                                        "page": result["page"]
                                    }
                                )

                        sources_to_save = unique_sources


                        # --------------------------------------------------
                        # Display sources
                        # --------------------------------------------------

                        if sources_to_save:

                            st.markdown(
                                "**📚 Sources:**"
                            )

                            for source in sources_to_save:

                                st.write(
                                    f"- {source['source']} "
                                    f"(Page {source['page']})"
                                )


                    # --------------------------------------------------
                    # Save assistant response
                    # --------------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_to_save
                        }
                    )


                except Exception as e:

                    st.error(
                        f"An error occurred: {str(e)}"
                    )