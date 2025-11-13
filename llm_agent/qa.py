#!/usr/bin/env python3
"""
MCDC Q&A Agent - RAG-based system for in-depth technical questions.

Handles queries about:
- Simulator architecture and internals
- Physics theory and methods
- Source code implementation details
- Usage patterns and examples

Uses intelligent metadata filtering to retrieve relevant document types.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import re

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# ── CONFIGURATION ──
DB_PATH = "llm_agent/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.0-flash-exp"
RETRIEVAL_K = 5  # Retrieve top 5 documents

# ── COLORS FOR CLI ──
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_section(title: str, content: str):
    """Print colored section."""
    print(f"{Colors.BOLD}{Colors.BLUE}{title}:{Colors.END}")
    print(content)
    print()


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ Error: {text}{Colors.END}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


# ══════════════════════════════════════════════════════════════════════
# QUERY CLASSIFICATION & FILTERING
# ══════════════════════════════════════════════════════════════════════

def classify_query(question: str) -> Dict[str, Any]:
    """
    Classify the query to determine what document types to retrieve.
    
    Returns dict with:
    - query_type: "theory", "architecture", "usage", "mixed"
    - filter_metadata: dict for Chroma metadata filtering
    - explanation: why this classification was chosen
    """
    question_lower = question.lower()
    
    # Keywords for different query types
    theory_keywords = [
        "theory", "physics", "algorithm", "method", "mathematical",
        "equation", "derivation", "proof", "paper", "research",
        "monte carlo", "transport", "cross section", "neutron"
    ]
    
    architecture_keywords = [
        "architecture", "structure", "design", "internal",
        "implementation", "how does", "how is", "kernel",
        "data structure", "class hierarchy", "module",
        "under the hood", "behind the scenes"
    ]
    
    usage_keywords = [
        "how to", "example", "use", "create", "setup",
        "tutorial", "guide", "syntax", "api",
        "parameters", "arguments", "call"
    ]
    
    code_keywords = [
        "function", "class", "method", "code", "implementation",
        "source", "definition", "show me", "where is"
    ]
    
    # Count keyword matches
    theory_score = sum(1 for kw in theory_keywords if kw in question_lower)
    arch_score = sum(1 for kw in architecture_keywords if kw in question_lower)
    usage_score = sum(1 for kw in usage_keywords if kw in question_lower)
    code_score = sum(1 for kw in code_keywords if kw in question_lower)
    
    # Determine query type and filtering strategy
    if theory_score >= 2 or "paper" in question_lower:
        return {
            "query_type": "theory",
            "filter_metadata": {"type": {"$in": ["paper", "source_code", "internal_code"]}},
            "explanation": "Theory/physics question - searching papers and implementation"
        }
    
    elif arch_score >= 2 or code_score >= 2:
        return {
            "query_type": "architecture",
            "filter_metadata": {"type": {"$in": ["internal_code", "source_code", "paper"]}},
            "explanation": "Architecture question - prioritizing internal code and implementation"
        }
    
    elif usage_score >= 2:
        return {
            "query_type": "usage",
            "filter_metadata": {"type": {"$in": ["api_doc", "code_example", "source_code"]}},
            "explanation": "Usage question - searching API docs and examples"
        }
    
    else:
        # Mixed or unclear - retrieve from all types
        return {
            "query_type": "mixed",
            "filter_metadata": None,  # No filtering - retrieve all types
            "explanation": "General question - searching all document types"
        }


# ══════════════════════════════════════════════════════════════════════
# RAG COMPONENTS
# ══════════════════════════════════════════════════════════════════════

def load_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Load Gemini LLM with API key validation."""
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set it with: export GEMINI_API_KEY='your-api-key-here'"
        )
    
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        google_api_key=api_key
    )


def load_vectorstore() -> Chroma:
    """Load Chroma vectorstore."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {DB_PATH}. "
            "Please run: python llm_agent/build_index.py"
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="mcdc_docs"
    )


def format_docs_with_sources(docs: List[Document]) -> str:
    """Format retrieved documents with clear source attribution."""
    formatted = []
    
    for i, doc in enumerate(docs, 1):
        # Extract metadata
        doc_type = doc.metadata.get("type", "unknown")
        source = doc.metadata.get("source", "unknown")
        priority = doc.metadata.get("priority", "?")
        
        # Format header based on document type
        if doc_type == "paper":
            paper_title = doc.metadata.get("paper_title", "Unknown Paper")
            page = doc.metadata.get("page_number", "?")
            header = f"📄 Paper: {paper_title} (Page {page})"
        
        elif doc_type in ["source_code", "internal_code"]:
            module = doc.metadata.get("module", "unknown")
            func_name = doc.metadata.get("function_name", "")
            visibility = "Internal" if doc_type == "internal_code" else "Public"
            header = f"💻 {visibility} Code: {module}.{func_name}"
        
        elif doc_type == "api_doc":
            func_name = doc.metadata.get("function", "unknown")
            header = f"📚 API Doc: mcdc.{func_name}"
        
        elif "example" in doc_type:
            test_name = doc.metadata.get("test_name", "unknown")
            complexity = doc.metadata.get("complexity", "unknown")
            header = f"📝 Example: {test_name} ({complexity})"
        
        else:
            header = f"📋 {source}"
        
        # Format content
        content = doc.page_content.strip()
        
        formatted.append(f"[Source {i}] {header}\nPriority: {priority}\n\n{content}")
    
    return "\n\n" + "─" * 70 + "\n\n".join(formatted)


def create_qa_chain(vectorstore: Chroma, llm: ChatGoogleGenerativeAI, filter_metadata: Dict = None):
    """
    Create RAG chain with optional metadata filtering.
    
    Args:
        vectorstore: Chroma vectorstore
        llm: Language model
        filter_metadata: Optional metadata filter for retrieval
    """
    # Create retriever with optional filtering
    search_kwargs = {"k": RETRIEVAL_K}
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # Prompt template for technical Q&A
    prompt = PromptTemplate.from_template("""You are MCDC-Expert, an AI assistant specializing in the Monte Carlo / Dynamic Code (MCDC) particle transport simulator.

Your role is to answer in-depth technical questions about:
- Simulator architecture and internal implementation
- Physics methods and algorithms  
- Source code structure and design decisions
- Mathematical theory and equations

INSTRUCTIONS:
1. Answer based ONLY on the provided context documents
2. Be technical and precise - this is for experienced developers/researchers
3. Reference specific functions, classes, or equations when relevant
4. If the context doesn't contain enough information, say so clearly
5. Cite which source(s) you're using (e.g., "According to Source 1...")
6. For code questions, include relevant code snippets from the context
7. For theory questions, explain the mathematical/physical principles

Question: {question}

Context Documents:
{context}

Answer:""")
    
    # Build LCEL chain
    chain = (
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


# ══════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ══════════════════════════════════════════════════════════════════════

def print_welcome():
    """Print welcome message."""
    print_header("MCDC Q&A Agent - Technical Support System")
    print(f"{Colors.BOLD}Ask in-depth questions about:{Colors.END}")
    print("  • Simulator architecture and internals")
    print("  • Physics theory and algorithms")
    print("  • Source code implementation")
    print("  • Usage patterns and examples")
    print(f"\n{Colors.YELLOW}Type 'quit' or 'exit' to leave{Colors.END}")
    print(f"{Colors.YELLOW}Type 'help' for tips on asking good questions{Colors.END}\n")


def print_help():
    """Print help message with query examples."""
    print_header("How to Ask Good Questions")
    
    print(f"{Colors.BOLD}🔬 Theory Questions:{Colors.END}")
    print("  • 'Explain the Monte Carlo algorithm used in MCDC'")
    print("  • 'What physics methods are implemented for neutron transport?'")
    print("  • 'How does MCDC handle cross-section data?'")
    
    print(f"\n{Colors.BOLD}🏗️  Architecture Questions:{Colors.END}")
    print("  • 'How is the particle tracking implemented internally?'")
    print("  • 'Explain the kernel architecture'")
    print("  • 'What data structures are used for geometry?'")
    
    print(f"\n{Colors.BOLD}💻 Code Questions:{Colors.END}")
    print("  • 'Show me how the eigenmode solver is implemented'")
    print("  • 'Where is the tally scoring logic?'")
    print("  • 'What does the _kernel module do?'")
    
    print(f"\n{Colors.BOLD}📖 Usage Questions:{Colors.END}")
    print("  • 'How do I create a multigroup material?'")
    print("  • 'Show me an example of setting up tallies'")
    print("  • 'What are the parameters for mcdc.source()?'\n")


def interactive_mode():
    """Run interactive Q&A loop."""
    print_welcome()
    
    # Initialize components
    try:
        print(f"{Colors.YELLOW}Loading RAG system...{Colors.END}")
        vectorstore = load_vectorstore()
        llm = load_llm()
        print_success("System ready!")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)
    
    # Main loop
    while True:
        # Get question
        print(f"\n{Colors.BOLD}{Colors.GREEN}Your Question:{Colors.END} ", end="")
        question = input().strip()
        
        # Handle commands
        if question.lower() in ["quit", "exit", "q"]:
            print(f"\n{Colors.CYAN}Thanks for using MCDC Q&A! Goodbye!{Colors.END}\n")
            break
        
        if question.lower() == "help":
            print_help()
            continue
        
        if not question:
            print(f"{Colors.YELLOW}Please enter a question.{Colors.END}")
            continue
        
        # Classify query
        print(f"\n{Colors.YELLOW}🔍 Analyzing query...{Colors.END}")
        classification = classify_query(question)
        print(f"{Colors.YELLOW}   {classification['explanation']}{Colors.END}\n")
        
        # Create chain with appropriate filtering
        chain = create_qa_chain(vectorstore, llm, classification['filter_metadata'])
        
        # Get answer
        try:
            print(f"{Colors.BOLD}{Colors.BLUE}Answer:{Colors.END}\n")
            
            # Stream response
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
            
            print("\n")
        
        except Exception as e:
            print_error(f"Failed to generate answer: {e}")


def single_question_mode(question: str):
    """Answer a single question and exit."""
    try:
        vectorstore = load_vectorstore()
        llm = load_llm()
    except Exception as e:
        print_error(str(e))
        sys.exit(1)
    
    # Classify and answer
    classification = classify_query(question)
    print(f"\n{Colors.YELLOW}{classification['explanation']}{Colors.END}\n")
    
    chain = create_qa_chain(vectorstore, llm, classification['filter_metadata'])
    
    try:
        print(f"{Colors.BOLD}{Colors.BLUE}Answer:{Colors.END}\n")
        
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        
        print("\n")
    
    except Exception as e:
        print_error(f"Failed to generate answer: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    # Check if question provided as argument
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        single_question_mode(question)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()