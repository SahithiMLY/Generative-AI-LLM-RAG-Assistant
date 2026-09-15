# 🤖 Generative AI & LLM Knowledge Assistant using RAG

## 📌 Project Overview

The **Generative AI & LLM Knowledge Assistant** is a domain-specific
question-answering chatbot built using **Retrieval-Augmented Generation (RAG)**.

The system allows users to ask questions about topics such as:

- Generative AI
- Large Language Models (LLMs)
- Transformers
- Attention and Self-Attention
- Prompt Engineering
- Retrieval-Augmented Generation (RAG)

Instead of relying only on the knowledge stored inside a language model,
the system first retrieves relevant information from uploaded PDF documents
and then provides the retrieved information as context to the language model.

This helps the chatbot generate answers that are grounded in the provided
documents.

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a domain-specific RAG chatbot.
2. Allow users to work with PDF-based knowledge sources.
3. Extract text from PDF documents.
4. Split documents into smaller chunks.
5. Convert text chunks into vector embeddings.
6. Store embeddings using FAISS.
7. Retrieve relevant document chunks for a user query.
8. Generate answers using a Large Language Model.
9. Display the source document and page number.
10. Prevent the chatbot from inventing information that is not available
    in the uploaded documents.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an approach that combines:

**Information Retrieval + Language Model Generation**

The system first searches a knowledge base for relevant information.
The retrieved information is then provided to the language model as context.

The basic workflow is:

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Relevant Document Chunks
      ↓
RAG Prompt
      ↓
Large Language Model
      ↓
Generated Answer
      ↓
Source Document + Page