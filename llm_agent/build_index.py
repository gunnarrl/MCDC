import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── CONFIGURATION ──
CORPUS_ROOT = Path("llm_agent/corpus")
OUTPUT_DIR = Path("llm_agent/scraped_docs")
INDEX_DIR = Path("llm_agent/vectorstore")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Document paths
RTD_DOCS_PATH = OUTPUT_DIR / "function_docs.json"
AUTO_API_PATH = OUTPUT_DIR / "auto_api.json"
EXAMPLES_DIR = OUTPUT_DIR / "examples"

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)


def load_documents() -> List[Document]:
    """
    Load all three document types with proper metadata.
    Returns list of Documents ready for indexing.
    """
    docs: List[Document] = []
    
    # 1. RTD docs (highest quality, only 10 functions)
    if RTD_DOCS_PATH.exists():
        logging.info(f"Loading RTD docs from {RTD_DOCS_PATH}")
        rtd_data = json.loads(RTD_DOCS_PATH.read_text(encoding="utf-8"))
        for func_name, data in rtd_data.items():
            docs.append(Document(
                page_content=json.dumps(data, indent=2),
                metadata={
                    "source": "rtd",
                    "type": "api_doc",
                    "function": func_name,
                    "quality": "high",
                    "priority": "3"  # String for Chroma filtering
                }
            ))
    else:
        logging.warning(f"⚠ RTD docs not found at {RTD_DOCS_PATH}")
    
    # 2. Auto-generated stubs (covers everything else)
    if AUTO_API_PATH.exists():
        logging.info(f"Loading auto-generated docs from {AUTO_API_PATH}")
        auto_data = json.loads(AUTO_API_PATH.read_text(encoding="utf-8"))
        for qual_name, data in auto_data.items():
            # Only include top-level mcdc.* (skip internal modules)
            if qual_name.count(".") <= 2:
                docs.append(Document(
                    page_content=json.dumps(data, indent=2),
                    metadata={
                        "source": "auto",
                        "type": "api_doc",
                        "function": qual_name.split(".")[-1],
                        "quality": "medium",
                        "priority": "2"
                    }
                ))
    else:
        logging.warning(f"⚠ Auto API docs not found at {AUTO_API_PATH}")
    
    # 3. Regression examples (working code)
    if EXAMPLES_DIR.exists():
        logging.info(f"Loading examples from {EXAMPLES_DIR}")
        for example_file in EXAMPLES_DIR.glob("*.py"):
            content = example_file.read_text(encoding="utf-8")
            # Extract test name from header
            test_name = example_file.stem
            
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "example",
                    "type": "full_example",
                    "test_name": test_name,
                    "quality": "low",  # Larger, less structured
                    "priority": "1"
                }
            ))
    else:
        logging.warning(f"Examples directory not found at {EXAMPLES_DIR}")
    
    logging.info(f"Loaded {len(docs)} documents total")
    return docs


def create_vectorstore(docs: List[Document]) -> Chroma:
    """Create fresh Chroma vectorstore; deletes old index."""
    if INDEX_DIR.exists():
        logging.info(f"Removing old index at {INDEX_DIR}")
        shutil.rmtree(INDEX_DIR)
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    logging.info(f"Creating new vectorstore at {INDEX_DIR}")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )
    logging.info("Index created and persisted")
    return vectorstore


def search_with_relevance(query: str, vectorstore: Chroma, k: int = 5) -> List[Document]:
    """
    Retrieve documents with quality-based re-ranking.
    RTD > Auto > Example, regardless of raw similarity.
    """

    # Retrieve more than needed to ensure good coverage
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k * 3)
    
    # Re-rank by priority (higher = better)
    priority_map = {"3": 3.0, "2": 2.0, "1": 1.0}
    weighted = []
    for doc, score in results:
        
        # Convert relevance (0-1) to distance-like score for weighting
        priority = priority_map.get(doc.metadata.get("priority", "1"), 1.0)
        weighted_score = (1.0 - score) / priority  # Lower is better
        weighted.append((doc, weighted_score))
    
    weighted.sort(key=lambda x: x[1])
    return [doc for doc, _ in weighted[:k]]


def verify_index(vectorstore: Chroma):
    """Quick sanity check that index is queryable."""
    logging.info("\nVerifying index...")
    test_queries = [
        "define material with capture cross-section",
        "create surface cylinder",
        "run simulation with 100 particles"
    ]
    
    for query in test_queries:
        results = search_with_relevance(query, vectorstore, k=2)
        sources = [r.metadata["source"] for r in results]
        logging.info(f"  '{query}' → {sources}")


def main():
    """Full pipeline: load → index → verify."""
    logging.info("Starting RAG index build")
    
    docs = load_documents()
    if not docs:
        logging.error("No documents found. Run extract_api_static.py first.")
        exit(1)
    
    vectorstore = create_vectorstore(docs)
    verify_index(vectorstore)
    
    logging.info("Index build complete.")


if __name__ == "__main__":
    main()