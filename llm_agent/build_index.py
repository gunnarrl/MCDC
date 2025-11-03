import os
from typing import List, Dict, Any
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Constants

CORPUS_PATH = os.path.join("llm_agent", "corpus")
DB_PATH = os.path.join("llm_agent", "db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# setup text splitters

CHUNK_SIZE = 1000  # Target size for each chunk (in characters)
CHUNK_OVERLAP = 200  # Overlap between chunks to maintain context

# Default splitter for .txt, .sh, .toml, etc.
DEFAULT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

# Python-specific splitter
PYTHON_SPLITTER = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# Markdown-specific splitter
MARKDOWN_SPLITTER = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# reStructuredText-specific splitter
RST_SPLITTER = RecursiveCharacterTextSplitter.from_language(
    language=Language.RST,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

SPLITTER_MAP = {
    ".py": PYTHON_SPLITTER,
    ".md": MARKDOWN_SPLITTER,
    ".sh": DEFAULT_SPLITTER,
    ".rst": RST_SPLITTER,
    ".toml": DEFAULT_SPLITTER,
    ".txt": DEFAULT_SPLITTER,
    ".gitignore": DEFAULT_SPLITTER,
    "LICENSE": DEFAULT_SPLITTER,
}

def load_documents(corpus_path: str) -> List[Document]:

    print(f"Loading documents from {corpus_path}...")

    loader = DirectoryLoader(
        corpus_path,
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,  # Skip files it can't read
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:

    print("Splitting documents into chunks...")
    all_splits: List[Document] = []
    
    for doc in documents:
        # get file extension to determine splitter
        file_path = doc.metadata.get("source", "")
        file_ext = os.path.splitext(file_path)[1]
        
        splitter = SPLITTER_MAP.get(file_ext, DEFAULT_SPLITTER)
        
        # split document and add chuncks to list
        chunks = splitter.split_documents([doc])
        all_splits.extend(chunks)

    print(f"Created {len(all_splits)} chunks from {len(documents)} documents.")
    return all_splits


def create_index(splits: List[Document], db_path: str) -> None:
    print("Initializing embedding model...")

    # create embedding model
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs
    )

    print(f"Creating and persisting vector index at {db_path}...")
    
    # create vector database
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_path,
    )

    print("Vector index created successfully.")


def main():

    if not os.path.exists(CORPUS_PATH):
        print(f"Error: Corpus directory not found at {CORPUS_PATH}")
        print("Please run this script from the root MCDC directory.")
        return

    # load documents
    documents = load_documents(CORPUS_PATH)

    # split documents
    splits = split_documents(documents)
    if not splits:
        print("Error: No documents were split. Check corpus directory.")
        return

    # store in vector database
    create_index(splits, DB_PATH)
    
    print("\n--- Indexing Complete ---")
    print(f"Vector store is saved in: {DB_PATH}")


if __name__ == "__main__":
    main()