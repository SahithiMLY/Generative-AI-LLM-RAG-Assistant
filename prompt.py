SYSTEM_PROMPT = """
You are a Generative AI and Large Language Model Knowledge Assistant.

Your job is to answer questions ONLY using the information
provided in the retrieved document context.

STRICT RULES:

1. Use only the supplied document context.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. If the answer cannot be found in the supplied context,
   say exactly:

   "I could not find this information in the uploaded documents."

5. Give a clear and concise answer.
6. When possible, mention the source document and page number.
7. If the retrieved context is insufficient, do not guess.
"""


def create_rag_prompt(question, retrieved_documents):
    """
    Create the prompt sent to the LLM.
    """

    context_parts = []

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        text = document.page_content

        context_parts.append(
            f"""
SOURCE {i}
Document: {source}
Page: {page}

Content:
{text}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
{SYSTEM_PROMPT}

--------------------------------
RETRIEVED DOCUMENT CONTEXT
--------------------------------

{context}

--------------------------------
USER QUESTION
--------------------------------

{question}

--------------------------------
ANSWER
--------------------------------

Answer the question using ONLY the retrieved
document context above.
"""

    return prompt