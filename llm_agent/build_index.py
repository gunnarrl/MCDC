import os
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    RecursiveUrlLoader,
    PyMuPDFLoader,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Constants

CORPUS_PATH = os.path.join("llm_agent", "corpus")
WEB_URL = "https://mcdc.readthedocs.io/en/latest/"
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
    ".html": MARKDOWN_SPLITTER,
    ".sh": DEFAULT_SPLITTER,
    ".rst": RST_SPLITTER,
    ".toml": DEFAULT_SPLITTER,
    ".txt": DEFAULT_SPLITTER,
    ".gitignore": DEFAULT_SPLITTER,
    "LICENSE": DEFAULT_SPLITTER,
}

def load_documents(corpus_path: str) -> List[Document]:

    print(f"Loading documents from {corpus_path}...")

    text_loader = DirectoryLoader(
        corpus_path,
        glob="**/*[.py, .md, .html, .sh, .rst, .toml, .txt, .gitignore, LICENSE]",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,  # skip files it can't read
    )
    text_documents = text_loader.load()
    print(f"Loaded {len(text_documents)} text documents.")

    pdf_loader = DirectoryLoader(
        corpus_path,
        glob="**/*.pdf", 
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,
    )
    pdf_documents = pdf_loader.load()
    print(f"Loaded {len(pdf_documents)} PDF documents.")

    documents = text_documents + pdf_documents

    print(f"Loaded {len(documents)} total documents.")
    return documents

def load_web_documents(url: str) -> List[Document]:

    print(f"Loading web documents from {url}...")
    
    def mcdc_extractor(html: str) -> str:

        soup = BeautifulSoup(html, "lxml")
        # select the main content div, default to body if not found
        main_content = soup.find("div", {"role": "main"}) or soup.body
        return main_content.get_text(separator="\n", strip=True)

    loader = RecursiveUrlLoader(
        url=url,
        max_depth=10,  # Adjust max_depth as needed
        extractor=mcdc_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        check_response_status=True,
        exclude_dirs=[
            f"{url}_static/",
            f"{url}genindex.html",
            f"{url}py-modindex.html",
            f"{url}search.html",
        ],
    )
    
    web_documents = loader.load()
    print(f"Loaded {len(web_documents)} web pages.")
    return web_documents

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
    file_docs = load_documents(CORPUS_PATH)
    web_docs = load_web_documents(WEB_URL)  
    all_documents = file_docs + web_docs
    if not all_documents:
        print("Error: No documents were loaded from file or web.")
        return

    # chunk documents
    splits = split_documents(all_documents)
    if not splits:
        print("Error: No documents were split. Check sources.")
        return

    # save in vectordb
    create_index(splits, DB_PATH)
    
    print("\n--- Indexing Complete ---")
    print(f"Vector store is saved in: {DB_PATH}")

if __name__ == "__main__":
    main()