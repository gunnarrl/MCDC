from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

# Config must match your build_index.py
INDEX_DIR = Path("llm_agent/vectorstore") 
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" # Updated model

def test_query(query):
    print(f"\n{'='*50}\nQuery: {query}\n{'='*50}")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(INDEX_DIR), embedding_function=embeddings, collection_name="mcdc_docs")
    
    # Fetch more results to see if the answer is buried (k=10)
    results = db.similarity_search_with_score(query, k=5)
    
    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source')} | Section: {doc.metadata.get('section')}")
        print(f"File: {doc.metadata.get('file_path')}")
        clean_content = doc.page_content.replace('\n', ' ')
        print(f"Content Preview: {clean_content}...")

if __name__ == "__main__":
    # Test a conceptual question
    test_query("How do I define a surface?")
    
    # Test a specific code question (hard for pure vector search)
    test_query("What are the arguments for mcdc.surface.PlaneX?")