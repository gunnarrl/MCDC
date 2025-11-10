import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any
import re

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


def format_api_doc_readable(func_name: str, data: Dict[str, Any]) -> str:
    """
    Convert API doc JSON to human-readable text format.
    
    Instead of JSON dumps, create natural language documentation that's
    optimized for semantic search and beginner understanding.
    """
    parts = []
    
    # Header
    parts.append(f"Function: mcdc.{func_name}")
    parts.append("=" * 60)
    
    # Signature
    if "signature" in data:
        parts.append(f"\nSignature:\n{data['signature']}")
    
    # Description
    if "description" in data:
        parts.append(f"\nDescription:\n{data['description']}")
    
    # Parameters (formatted as list)
    if "parameters" in data and data["parameters"]:
        parts.append("\nParameters:")
        for param in data["parameters"]:
            if isinstance(param, dict):
                name = param.get("name", "")
                desc = param.get("description", "")
                parts.append(f"  • {name}: {desc}")
            else:
                parts.append(f"  • {param}")
    
    # Returns
    if "returns" in data:
        parts.append(f"\nReturns:\n{data['returns']}")
    
    # Example usage (if available)
    parts.append("\nExample usage:")
    if "MaterialMG" in func_name or func_name == "material":
        parts.append("""
fuel = mcdc.MaterialMG(
    capture=np.array([0.45]),
    scatter=np.array([[0.0]]),
    fission=np.array([0.55]),
    nu_p=np.array([2.5])
)
""")
    elif "Surface" in func_name or func_name == "surface":
        parts.append("""
# Plane surface
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")

# Cylinder surface
s2 = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=1.5)
""")
    elif func_name == "cell":
        parts.append("""
# Cell with region and material fill
fuel_cell = mcdc.Cell(region=+s1 & -s2, fill=fuel_material)
""")
    elif func_name == "source":
        parts.append("""
# Isotropic point source
mcdc.Source(
    x=[0.0, 10.0],
    y=[0.0, 10.0],
    isotropic=True,
    energy_group=0
)
""")
    
    return "\n".join(parts)


def chunk_example_by_section(content: str, test_name: str) -> List[tuple[str, str, str]]:
    """
    Split example code into sections based on comment headers.
    
    Returns:
        List of (section_name, code, description) tuples
    """
    chunks = []
    
    # Define section patterns - these are the standard MCDC workflow steps
    section_patterns = {
        "material": r"#.*[Ss]et materials?",
        "surface": r"#.*[Ss]et surfaces?",
        "cell": r"#.*[Ss]et cells?",
        "source": r"#.*[Ss]et source",
        "tally": r"#.*[Ss]et tallies?",
        "settings": r"#.*[Ss]ettings?",
    }
    
    lines = content.split('\n')
    current_section = None
    current_code = []
    
    for line in lines:
        # Check if line is a section header
        matched_section = None
        for section, pattern in section_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                matched_section = section
                break
        
        if matched_section:
            # Save previous section if exists
            if current_section and current_code:
                code_text = '\n'.join(current_code)
                chunks.append((current_section, code_text, test_name))
            
            # Start new section
            current_section = matched_section
            current_code = [line]
        elif current_section:
            current_code.append(line)
    
    # Save last section
    if current_section and current_code:
        code_text = '\n'.join(current_code)
        chunks.append((current_section, code_text, test_name))
    
    return chunks


def infer_complexity(content: str, test_name: str) -> str:
    """Infer example complexity from code characteristics."""
    # Count lines (excluding comments and blank lines)
    code_lines = [l for l in content.split('\n') 
                  if l.strip() and not l.strip().startswith('#')]
    
    # Count MCDC function calls
    mcdc_calls = len(re.findall(r'mcdc\.\w+', content))
    
    # Check for advanced features
    has_lattice = 'Lattice' in content
    has_universe = 'Universe' in content
    has_eigenmode = 'eigenmode' in content or 'k_eigenvalue' in test_name
    has_time_dep = '_td' in test_name or 'time=' in content
    
    if has_lattice or has_universe or (has_eigenmode and len(code_lines) > 100):
        return "advanced"
    elif mcdc_calls > 20 or len(code_lines) > 60:
        return "intermediate"
    else:
        return "beginner"


def load_documents() -> List[Document]:
    """
    Load all documents with human-readable formatting.
    
    Returns:
        List of Documents ready for indexing with improved content and metadata
    """
    docs: List[Document] = []
    
    # ═══════════════════════════════════════════════════════════════
    # 1. RTD API DOCS (high quality, human-readable format)
    # ═══════════════════════════════════════════════════════════════
    if RTD_DOCS_PATH.exists():
        logging.info(f"Loading RTD docs from {RTD_DOCS_PATH}")
        rtd_data = json.loads(RTD_DOCS_PATH.read_text(encoding="utf-8"))
        
        for func_name, data in rtd_data.items():
            # Convert to readable format instead of JSON dump
            readable_content = format_api_doc_readable(func_name, data)
            
            docs.append(Document(
                page_content=readable_content,
                metadata={
                    "source": "rtd",
                    "type": "api_doc",
                    "function": func_name,
                    "section": infer_section_from_function(func_name),
                    "quality": "high",
                    "priority": "3"
                }
            ))
        logging.info(f"  ✓ Loaded {len(rtd_data)} RTD docs")
    else:
        logging.warning(f"  ⚠ RTD docs not found at {RTD_DOCS_PATH}")
    
    # ═══════════════════════════════════════════════════════════════
    # 2. AUTO-GENERATED STUBS (medium quality, readable format)
    # ═══════════════════════════════════════════════════════════════
    if AUTO_API_PATH.exists():
        logging.info(f"Loading auto-generated docs from {AUTO_API_PATH}")
        auto_data = json.loads(AUTO_API_PATH.read_text(encoding="utf-8"))
        
        count = 0
        for qual_name, data in auto_data.items():
            # Only include top-level mcdc.* (skip internal modules)
            if qual_name.count(".") <= 2:
                # Convert to readable format
                func_name = qual_name.split(".")[-1]
                readable_content = format_api_doc_readable(func_name, data)
                
                docs.append(Document(
                    page_content=readable_content,
                    metadata={
                        "source": "auto",
                        "type": "api_doc",
                        "function": func_name,
                        "section": infer_section_from_function(func_name),
                        "quality": "medium",
                        "priority": "2"
                    }
                ))
                count += 1
        logging.info(f"  ✓ Loaded {count} auto-generated docs")
    else:
        logging.warning(f"  ⚠ Auto API docs not found at {AUTO_API_PATH}")
    
    # ═══════════════════════════════════════════════════════════════
    # 3. REGRESSION EXAMPLES (chunked by section, with headers)
    # ═══════════════════════════════════════════════════════════════
    if EXAMPLES_DIR.exists():
        logging.info(f"Loading examples from {EXAMPLES_DIR}")
        
        example_count = 0
        chunk_count = 0
        
        for example_file in EXAMPLES_DIR.glob("*.py"):
            content = example_file.read_text(encoding="utf-8")
            test_name = example_file.stem
            complexity = infer_complexity(content, test_name)
            
            # Extract functions used
            functions_used = list(set(re.findall(r'mcdc\.(\w+)', content)))
            
            # Chunk by section
            chunks = chunk_example_by_section(content, test_name)
            
            if chunks:
                # Create documents for each section
                for section, code, test_name in chunks:
                    # Add semantic header
                    page_content = f"""Example: {test_name} ({complexity})
Demonstrates: {section.title()} creation in MCDC

Code:
{code}
"""
                    
                    docs.append(Document(
                        page_content=page_content,
                        metadata={
                            "source": "example",
                            "type": "code_section",
                            "test_name": test_name,
                            "section": section,
                            "complexity": complexity,
                            "functions_used": ','.join(functions_used[:5]),  # String for Chroma
                            "quality": "low",
                            "priority": "1"
                        }
                    ))
                    chunk_count += 1
            else:
                # If no sections found, store whole example with header
                page_content = f"""Example: {test_name} ({complexity})
Complete MCDC simulation script

Functions used: {', '.join(functions_used[:10])}

Code:
{content}
"""
                
                docs.append(Document(
                    page_content=page_content,
                    metadata={
                        "source": "example",
                        "type": "full_example",
                        "test_name": test_name,
                        "section": "complete",
                        "complexity": complexity,
                        "functions_used": ','.join(functions_used[:5]),
                        "quality": "low",
                        "priority": "1"
                    }
                ))
                chunk_count += 1
            
            example_count += 1
        
        logging.info(f"  ✓ Loaded {example_count} examples → {chunk_count} chunks")
    else:
        logging.warning(f"  ⚠ Examples directory not found at {EXAMPLES_DIR}")
    
    logging.info(f"✓ Total documents: {len(docs)}")
    return docs


def infer_section_from_function(func_name: str) -> str:
    """Map function name to workflow section."""
    func_lower = func_name.lower()
    
    if 'material' in func_lower or 'nuclide' in func_lower:
        return 'material'
    elif 'surface' in func_lower or 'plane' in func_lower or 'cylinder' in func_lower or 'sphere' in func_lower:
        return 'surface'
    elif 'cell' in func_lower or 'universe' in func_lower or 'lattice' in func_lower:
        return 'cell'
    elif 'source' in func_lower:
        return 'source'
    elif 'tally' in func_lower or 'mesh' in func_lower:
        return 'tally'
    elif 'setting' in func_lower or 'eigenmode' in func_lower:
        return 'settings'
    else:
        return 'general'


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
        collection_name="mcdc_docs",
        collection_metadata={"hnsw:space": "cosine"}
    )
    logging.info("✓ Index created and persisted")
    return vectorstore


def verify_index(vectorstore: Chroma):
    """Quick sanity check that index is queryable."""
    logging.info("\nVerifying index with test queries...")
    test_queries = [
        ("material", "how to create material with capture cross-section"),
        ("surface", "create cylindrical surface"),
        ("source", "define isotropic source")
    ]
    
    for section, query in test_queries:
        # Test without filter
        results = vectorstore.similarity_search(query, k=2)
        sources = [r.metadata.get("source", "?") for r in results]
        sections = [r.metadata.get("section", "?") for r in results]
        
        logging.info(f"  '{query}'")
        logging.info(f"    → sources: {sources}, sections: {sections}")
        
        # Show first 100 chars of top result
        if results:
            preview = results[0].page_content[:100].replace('\n', ' ')
            logging.info(f"    → preview: {preview}...")


def main():
    """Full pipeline: load → index → verify."""
    logging.info("="*60)
    logging.info("Starting RAG index build with human-readable documents")
    logging.info("="*60)
    
    docs = load_documents()
    if not docs:
        logging.error("❌ No documents found. Run extract_api_static.py first.")
        exit(1)
    
    vectorstore = create_vectorstore(docs)
    verify_index(vectorstore)
    
    logging.info("\n" + "="*60)
    logging.info("✓ Index build complete!")
    logging.info("="*60)


if __name__ == "__main__":
    main()