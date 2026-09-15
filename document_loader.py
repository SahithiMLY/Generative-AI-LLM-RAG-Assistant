import os
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def extract_text_from_pdfs(pdf_folder="documents"):
    """
    Extract text from every PDF page.

    Each page is stored as a LangChain Document with:
    - page content
    - source document name
    - page number
    """

    documents = []

    pdf_folder = Path(pdf_folder)

    if not pdf_folder.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {pdf_folder}"
        )

    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside {pdf_folder}"
        )

    for pdf_file in pdf_files:

        print(f"Reading: {pdf_file.name}")

        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text()

            # Skip empty pages
            if not text or not text.strip():
                continue

            document = Document(
                page_content=text.strip(),
                metadata={
                    "source": pdf_file.name,
                    "page": page_number
                }
            )

            documents.append(document)

    return documents


def split_documents(documents):
    """
    Split extracted pages into smaller overlapping chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def load_and_split_documents(pdf_folder="documents"):
    """
    Complete PDF processing pipeline:
    
    PDFs
      ↓
    Extract text
      ↓
    Split into chunks
    """

    documents = extract_text_from_pdfs(pdf_folder)

    chunks = split_documents(documents)

    return chunks


if __name__ == "__main__":

    print("\nStarting PDF processing...\n")

    chunks = load_and_split_documents("documents")

    print("\n--------------------------------")
    print("PDF PROCESSING COMPLETE")
    print("--------------------------------")

    print(f"Total chunks created: {len(chunks)}")

    if chunks:
        print("\nExample chunk:")
        print("--------------------------------")
        print(chunks[0].page_content[:500])

        print("\nMetadata:")
        print(chunks[0].metadata)