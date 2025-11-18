import json
import logging
import shutil
import ast
from pathlib import Path
from typing import List, Dict, Any
import re

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import warnings

# Suppress PDF parsing warnings (malformed float metadata)
warnings.filterwarnings("ignore", message="could not convert string to float")
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

# ── CONFIGURATION ──
CORPUS_ROOT = Path("llm_agent/corpus")
OUTPUT_DIR = Path("llm_agent/scraped_docs")
INDEX_DIR = Path("llm_agent/vectorstore2")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Document paths
RTD_DOCS_PATH = OUTPUT_DIR / "function_docs.json"
AUTO_API_PATH = OUTPUT_DIR / "auto_api.json"
EXAMPLES_DIR = OUTPUT_DIR / "examples"
PHYSICS_DOCS_DIR = CORPUS_ROOT / "physics_docs"  # NEW: PDFs
SOURCE_CODE_DIR = CORPUS_ROOT / "mcdc"  # NEW: Source code

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
    if "MaterialMG" in func_name:
        parts.append("""
        fuel = mcdc.MaterialMG(
            capture=np.array([0.45]),
            scatter=np.array([[0.0]]),
            fission=np.array([0.55]),
            nu_p=np.array([2.5])
        )
        """)
    elif "material" in func_name:
        parts.append("""
        # Set materials
        fuel = mcdc.Material(
            nuclide_composition={
                "U235": 0.0005581658948833916 * 7e-2,
                "U238": 0.022404594715383263 * 7e-2,
                "O16": 0.045831301393656466,
            }
        )""")
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
    elif "tally" in func_name:
        parts.append("""
        # Mesh tally
        mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
        mcdc.TallyMesh(
            mesh=mesh, mu=np.linspace(-1.0, 1.0, 32 + 1), scores=["flux", "collision"]
        )

        # Surface tally
        mcdc.TallySurface(surface=s4, scores=["net-current"])
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
    
    if has_lattice or has_universe or has_eigenmode:
        return "advanced"
    elif mcdc_calls > 20 or len(code_lines) > 60:
        return "intermediate"
    else:
        return "beginner"


# ═══════════════════════════════════════════════════════════════════════
# NEW: PDF LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_pdf_documents() -> List[Document]:
    """
    Load and chunk PDF documents from physics_docs directory.
    
    Returns documents with metadata: type=paper, paper_title, page_number
    Priority: 5 (highest - peer-reviewed theory)
    """
    docs = []
    
    if not PHYSICS_DOCS_DIR.exists():
        logging.warning(f"  ⚠  Physics docs directory not found at {PHYSICS_DOCS_DIR}")
        return docs
    
    logging.info(f"Loading PDFs from {PHYSICS_DOCS_DIR}")
    
    pdf_files = list(PHYSICS_DOCS_DIR.glob("*.pdf"))
    
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            paper_title = pdf_path.stem.replace("_", " ").title()
            
            for page_num, page in enumerate(pages, start=1):
                # Clean up text
                text = page.page_content.strip()
                
                # Skip empty pages
                if len(text) < 100:
                    continue
                
                # Add semantic header
                content = f"""Paper: {paper_title} (Page {page_num})

{text}
"""
                
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": "paper",
                        "type": "paper",
                        "paper_title": paper_title,
                        "page_number": page_num,
                        "file_path": str(pdf_path.relative_to(CORPUS_ROOT)),
                        "quality": "high",
                        "priority": 5  # Highest - peer-reviewed theory
                    }
                ))
            
            logging.info(f"  ✓ Loaded {len(pages)} pages from {pdf_path.name}")
        
        except ImportError as e:
            if "cryptography" in str(e):
                logging.error(f"  ✗ {pdf_path.name} is encrypted. Install with: pip install cryptography")
            else:
                logging.error(f"  ✗ Failed to load {pdf_path.name}: {e}")
        except Exception as e:
            logging.error(f"  ✗ Failed to load {pdf_path.name}: {e}")
    
    return docs


# ═══════════════════════════════════════════════════════════════════════
# NEW: SOURCE CODE LOADING (Function-level chunking)
# Includes BOTH public and private modules
# ═══════════════════════════════════════════════════════════════════════

def extract_function_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse Python file and extract function/class definitions with context.
    
    Returns list of dicts with: name, type, code, docstring, line_start, line_end
    """
    chunks = []
    
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        lines = content.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # Extract the function/class code
                start_line = node.lineno - 1  # 0-indexed
                end_line = node.end_lineno if node.end_lineno else start_line + 10
                
                code_lines = lines[start_line:end_line]
                code = "\n".join(code_lines)
                
                # Get docstring
                docstring = ast.get_docstring(node) or ""
                
                chunks.append({
                    "name": node.name,
                    "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                    "code": code,
                    "docstring": docstring,
                    "line_start": start_line + 1,
                    "line_end": end_line
                })
    
    except Exception as e:
        logging.debug(f"  Could not parse {file_path}: {e}")
    
    return chunks


def is_private_module(file_path: Path, source_root: Path) -> bool:
    """
    Determine if a module is private (internal implementation).
    
    Private if:
    - Filename starts with underscore (e.g., _kernel.py)
    - Any parent directory starts with underscore
    - Is in a 'test' directory
    - Contains 'kernel', 'loop', 'adapter' (common MCDC internal patterns)
    """
    # Check filename
    if file_path.stem.startswith("_"):
        return True
    
    # Check any parent directory
    relative = file_path.relative_to(source_root)
    for part in relative.parts[:-1]:  # Exclude the filename itself
        if part.startswith("_") or part == "test":
            return True
    
    # MCDC-specific internal module patterns
    # These are implementation details, not user-facing API
    internal_patterns = [
        "kernel",      # Core simulation engine
        "loop",        # Particle transport loops
        "adapter",     # Interface adapters
        "type_",       # Type definitions
        "print_",      # Internal printing utilities
    ]
    
    stem_lower = file_path.stem.lower()
    for pattern in internal_patterns:
        if pattern in stem_lower:
            return True
    
    return False


def load_source_code_documents() -> List[Document]:
    """
    Load Python source code from mcdc directory.
    
    Chunks by function/class level with full context.
    Distinguishes between public API and internal implementation.
    
    Classification:
    - Functions starting with _ are ALWAYS internal (e.g., _rotate_particle)
    - Files in internal patterns (kernel, loop, adapter) are internal
    - Everything else is public API
    
    Returns documents with metadata:
    - type: "source_code" (public API, priority 4)
    - type: "internal_code" (private functions/modules, priority 3)
    """
    docs = []
    
    if not SOURCE_CODE_DIR.exists():
        logging.warning(f"  ⚠  Source code directory not found at {SOURCE_CODE_DIR}")
        return docs
    
    logging.info(f"Loading source code from {SOURCE_CODE_DIR}")
    
    # Walk through ALL .py files (including private modules)
    py_files = list(SOURCE_CODE_DIR.rglob("*.py"))
    
    public_chunks = 0
    private_chunks = 0
    
    for py_file in py_files:
        # Get module name
        module_path = py_file.relative_to(SOURCE_CODE_DIR.parent)
        module_name = str(module_path.with_suffix("")).replace("/", ".")
        
        # Check if entire module is private
        module_is_private = is_private_module(py_file, SOURCE_CODE_DIR)
        
        # Extract function-level chunks
        chunks = extract_function_chunks(py_file)
        
        for chunk in chunks:
            # Determine if THIS specific function/class is private
            # Priority 1: Function name starts with underscore
            # Priority 2: Module is marked as private
            function_name = chunk['name']
            is_private = function_name.startswith('_') or module_is_private
            
            # Determine doc type and priority
            if is_private:
                doc_type = "internal_code"
                priority = 3  # Internal architecture
                visibility = "Internal"
            else:
                doc_type = "source_code"
                priority = 4  # Public API implementation
                visibility = "Public"
            
            # Format as readable documentation
            content = f"""Module: {module_name} ({visibility})
{chunk['type'].title()}: {chunk['name']}

Docstring:
{chunk['docstring'] or 'No docstring available'}

Source Code (lines {chunk['line_start']}-{chunk['line_end']}):
```python
{chunk['code']}
```
"""
            
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "source_code" if not is_private else "internal_code",
                    "type": doc_type,
                    "module": module_name,
                    "function_name": chunk['name'],
                    "code_type": chunk['type'],
                    "file_path": str(py_file.relative_to(CORPUS_ROOT)),
                    "line_start": chunk['line_start'],
                    "line_end": chunk['line_end'],
                    "is_private": is_private,
                    "quality": "high",
                    "priority": priority
                }
            ))
            
            if is_private:
                private_chunks += 1
            else:
                public_chunks += 1
    
    logging.info(f"  ✓ Loaded {public_chunks} public + {private_chunks} private code chunks from {len(py_files)} files")
    return docs


# ═══════════════════════════════════════════════════════════════════════
# ORIGINAL DOCUMENT LOADING (updated priorities)
# ═══════════════════════════════════════════════════════════════════════

def load_documents() -> List[Document]:
    """
    Load all documents with human-readable formatting.
    
    Priority Scale (1-5 for k=5 retrieval):
    5 = RTD API docs, Physics papers (curated, peer-reviewed)
    4 = Public source code (actual implementation)
    3 = Internal source code (architecture details)
    2 = Auto-generated API stubs
    1 = Code examples (usage patterns)
    
    Returns:
        List of Documents ready for indexing with improved content and metadata
    """
    docs: List[Document] = []
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # 1. RTD API DOCS (priority 5 - official curated docs)
    # ╚═══════════════════════════════════════════════════════════════════╝
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
                    "priority": 5  # Highest - official docs
                }
            ))
        logging.info(f"  ✓ Loaded {len(rtd_data)} RTD docs")
    else:
        logging.warning(f"  ⚠  RTD docs not found at {RTD_DOCS_PATH}")
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # 2. AUTO-GENERATED STUBS (priority 2)
    # ╚═══════════════════════════════════════════════════════════════════╝

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
                        "priority": 2  # Auto-generated stubs
                    }
                ))
                count += 1
        logging.info(f"  ✓ Loaded {count} auto-generated docs")
    else:
        logging.warning(f"  ⚠  Auto API docs not found at {AUTO_API_PATH}")
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # 3. REGRESSION EXAMPLES (priority 1 - usage patterns)
    # ╚═══════════════════════════════════════════════════════════════════╝
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
                            "type": "code_example",
                            "test_name": test_name,
                            "section": section,
                            "complexity": complexity,
                            "functions_used": ','.join(functions_used[:5]),  # String for Chroma
                            "quality": "low",
                            "priority": 1  # Examples - usage patterns
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
                        "priority": 1
                    }
                ))
                chunk_count += 1
            
            example_count += 1
        
        logging.info(f"  ✓ Loaded {example_count} examples → {chunk_count} chunks")
    else:
        logging.warning(f"  ⚠  Examples directory not found at {EXAMPLES_DIR}")
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # 4. NEW: PHYSICS PAPERS (priority 5 - peer-reviewed theory)
    # ╚═══════════════════════════════════════════════════════════════════╝
    pdf_docs = load_pdf_documents()
    docs.extend(pdf_docs)
    
    # ╔═══════════════════════════════════════════════════════════════════╗
    # 5. NEW: SOURCE CODE (priority 4 public, 3 internal)
    # ╚═══════════════════════════════════════════════════════════════════╝
    source_docs = load_source_code_documents()
    docs.extend(source_docs)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Total documents loaded: {len(docs)}")
    logging.info(f"  - API docs (RTD): {sum(1 for d in docs if d.metadata.get('source') == 'rtd')}")
    logging.info(f"  - API docs (auto): {sum(1 for d in docs if d.metadata.get('source') == 'auto')}")
    logging.info(f"  - Examples: {sum(1 for d in docs if 'example' in d.metadata.get('type', ''))}")
    logging.info(f"  - Papers: {sum(1 for d in docs if d.metadata.get('type') == 'paper')}")
    logging.info(f"  - Public source: {sum(1 for d in docs if d.metadata.get('type') == 'source_code')}")
    logging.info(f"  - Internal source: {sum(1 for d in docs if d.metadata.get('type') == 'internal_code')}")
    logging.info(f"{'='*60}\n")
    
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


def main():
    """load → index → verify."""
    logging.info("="*60)
    logging.info("Starting RAG index build with ALL document types")
    logging.info("  Priority Scale: 5=Theory/API, 4=Public Code, 3=Internal,")
    logging.info("                  2=Auto-gen, 1=Examples")
    logging.info("="*60)
    
    docs = load_documents()
    if not docs:
        logging.error("❌ No documents found. Check your corpus directories.")
        exit(1)
    
    vectorstore = create_vectorstore(docs)
    
    logging.info("\n" + "="*60)
    logging.info("✅ Index build complete!")
    logging.info("="*60)


if __name__ == "__main__":
    main()