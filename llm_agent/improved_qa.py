import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import json




from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import load_llm


# ── CONFIGURATION ──
DB_PATH = "llm_agent/vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K_INITIAL = 20
RETRIEVAL_K_FINAL = 5   
BM25_WEIGHT = 0.3        


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
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")




def print_section(title: str, content: str):
    print(f"{Colors.BOLD}{Colors.BLUE}{title}:{Colors.END}")
    print(content)
    print()




def print_error(text: str):
    print(f"{Colors.RED}✗ Error: {text}{Colors.END}")




def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")




# QUERY ANALYSIS


def analyze_query_with_llm(question: str, llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
    """
    Use LLM to deeply understand the query intent and requirements.
   
    This replaces simple keyword matching with semantic understanding.
    """
    analysis_prompt = PromptTemplate.from_template("""You are analyzing a technical question about MCDC (Monte Carlo / Dynamic Code), a particle transport simulator.


Question: {question}


Analyze this question and respond with a JSON object containing:


1. "query_type": one of ["theory", "architecture", "implementation", "usage", "debugging", "mixed"]
2. "complexity": one of ["beginner", "intermediate", "advanced"]
3. "requires_code": true if answer needs actual source code
4. "requires_theory": true if answer needs physics/math theory
5. "requires_examples": true if answer needs usage examples
6. "key_topics": list of 3-5 most important concepts/modules mentioned (e.g., "tally", "material", "geometry", "kernel")
7. "reasoning": brief explanation of your analysis


Respond ONLY with valid JSON, no markdown fences.


Example:
{{
  "query_type": "architecture",
  "complexity": "advanced",
  "requires_code": true,
  "requires_theory": false,
  "requires_examples": false,
  "key_topics": ["particle tracking", "geometry", "kernel"],
  "reasoning": "User wants to understand internal implementation details"
}}


Your analysis:""")
   
    try:
        response = llm.invoke(analysis_prompt.format(question=question))
        text = response.content.strip()
        text = text.replace('```json', '').replace('```', '').strip()
       
        analysis = json.loads(text)
        return analysis
   
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ LLM analysis failed, using fallback: {e}{Colors.END}")
        return fallback_analysis(question)




def fallback_analysis(question: str) -> Dict[str, Any]:
    """Fallback to keyword-based analysis if LLM fails."""
    question_lower = question.lower()
   
    # keyword detection
    theory_keywords = ["theory", "physics", "algorithm", "mathematical", "derivation", "monte carlo", "cross section"]
    arch_keywords = ["architecture", "how does", "internally", "implementation", "structure", "design"]
    code_keywords = ["function", "class", "code", "show me", "where is", "source"]
    usage_keywords = ["how to", "example", "create", "setup", "use", "tutorial"]
   
    theory_score = sum(3 for kw in theory_keywords if kw in question_lower)
    arch_score = sum(3 for kw in arch_keywords if kw in question_lower)
    code_score = sum(2 for kw in code_keywords if kw in question_lower)
    usage_score = sum(2 for kw in usage_keywords if kw in question_lower)
   
    scores = {
        "theory": theory_score,
        "architecture": arch_score + code_score,
        "usage": usage_score
    }
   
    query_type = max(scores, key=scores.get) if max(scores.values()) > 0 else "mixed"
   
    # key topics from question
    mcdc_terms = ["material", "surface", "cell", "tally", "source", "kernel", "geometry",
                  "particle", "transport", "eigenmode", "universe", "lattice"]
    key_topics = [term for term in mcdc_terms if term in question_lower]
   
    return {
        "query_type": query_type,
        "complexity": "intermediate",
        "requires_code": code_score > 0,
        "requires_theory": theory_score > 0,
        "requires_examples": usage_score > 0,
        "key_topics": key_topics[:5],
        "reasoning": "Fallback keyword analysis"
    }




# QUERY EXPANSION


def expand_query(question: str, analysis: Dict[str, Any], llm: ChatGoogleGenerativeAI) -> List[str]:
    """
    Generate alternative phrasings for better retrieval coverage.
   
    Especially useful for:
    - Technical terms with synonyms
    - Questions that could be phrased multiple ways
    - Queries needing domain-specific expansion
    """
    expansion_prompt = PromptTemplate.from_template("""You are helping improve search for technical documentation about MCDC (Monte Carlo particle transport simulator).


Original question: {question}
Query type: {query_type}
Key topics: {key_topics}


Generate 2-3 alternative phrasings or related queries that would help retrieve relevant documentation. Focus on:
- Technical synonyms (e.g., "neutron flux" vs "neutron density")
- Different abstraction levels (high-level concept vs specific implementation)
- Related concepts that might contain the answer


Return ONLY a JSON array of strings, no explanation.


Example: ["original phrasing", "alternative 1", "alternative 2"]


Your expansions:""")
   
    try:
        response = llm.invoke(expansion_prompt.format(
            question=question,
            query_type=analysis.get('query_type', 'mixed'),
            key_topics=', '.join(analysis.get('key_topics', []))
        ))
       
        text = response.content.strip()
        text = text.replace('```json', '').replace('```', '').strip()
       
        import json
        expansions = json.loads(text)
       
        # Always include original
        if question not in expansions:
            expansions.insert(0, question)
       
        print(f"{Colors.YELLOW}Query expansions: {len(expansions)} variants{Colors.END}")
        return expansions[:4]  # Limit to 4 total queries
   
    except Exception as e:
        print(f"{Colors.YELLOW}⚠ Query expansion failed: {e}{Colors.END}")
        return [question]




# HYBRID RETRIEVAL (Dense + Sparse)


# def create_hybrid_retriever(vectorstore: Chroma, all_docs: List[Document]) -> EnsembleRetriever:
#     """
#     Create hybrid retriever combining:
#     1. Dense retrieval (semantic similarity via embeddings)
#     2. Sparse retrieval (keyword matching via BM25)
   
#     This catches both:
#     - Semantically similar content (even with different wording)
#     - Exact keyword matches (important for technical terms)
#     """
#     # Dense retriever (semantic)
#     dense_retriever = vectorstore.as_retriever(
#         search_kwargs={"k": RETRIEVAL_K_INITIAL}
#     )
   
#     # Sparse retriever (BM25 keyword search)
#     sparse_retriever = BM25Retriever.from_documents(all_docs)
#     sparse_retriever.k = RETRIEVAL_K_INITIAL
   
#     # Ensemble: 70% semantic, 30% keyword
#     hybrid_retriever = EnsembleRetriever(
#         retrievers=[dense_retriever, sparse_retriever],
#         weights=[1.0 - BM25_WEIGHT, BM25_WEIGHT]
#     )
   
#     return hybrid_retriever


# ENHANCED RETRIEVAL with RERANKING


def retrieve_and_rerank(
    vectorstore: Chroma, 
    question: str,        
    analysis: Dict[str, Any],
    k_final: int = RETRIEVAL_K_FINAL
) -> List[Document]:
    """
    Simple retrieval with reranking - no hybrid, no expansion.
    """
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": RETRIEVAL_K_INITIAL}
    )
    
    docs = retriever.invoke(question)
    
    print(f"{Colors.YELLOW}📥 Retrieved {len(docs)} documents{Colors.END}")
    
    # Rerank
    scored_docs = []
    for doc in docs:
        score = calculate_relevance_score(doc, question, analysis)
        scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    final_docs = [doc for score, doc in scored_docs[:k_final]]
    
    print(f"{Colors.GREEN}✓ Reranked to top {len(final_docs)} documents{Colors.END}")
    
    for i, (score, doc) in enumerate(scored_docs[:k_final], 1):
        doc_type = doc.metadata.get('type', 'unknown')
        source = doc.metadata.get('source', 'unknown')
        print(f"  {i}. [{doc_type}] Score: {score:.2f} - {source}")
    
    return final_docs




def calculate_relevance_score(
    doc: Document,
    question: str,
    analysis: Dict[str, Any]
) -> float:
    """
    Score document relevance based on multiple factors.
   
    PRIORITY SCALE (1-10):
    - 10: RTD API docs
    - 9:  Physics papers
    - 7:  Public source code
    - 5:  Internal source code
    - 3:  Code examples
    - 1:  Auto-generated stubs


    Total score range: [0, ~20]
    """
    score = 0.0
   
    doc_type = doc.metadata.get('type', '')
    source = doc.metadata.get('source', '')
   
    priority_map = {
        'api_doc': 10 if source == 'rtd' else 1,  # RTD=10, auto=1
        'paper': 9,                              
        'source_code': 7,                      
        'internal_code': 5,                    
        'code_example': 3,                      
        'full_example': 3,
        'example_workflow_group': 3
    }
    score += priority_map.get(doc_type, 2)
   
    # Query-type alignment
    query_type = analysis.get('query_type', 'mixed')
   
    alignment_bonus = {
        'theory': {'paper': 4, 'api_doc': 2},
        'architecture': {'internal_code': 4, 'source_code': 3, 'paper': 2},
        'implementation': {'source_code': 4, 'internal_code': 3, 'api_doc': 2},
        'usage': {'api_doc': 4, 'code_example': 3, 'full_example': 3},
        'debugging': {'source_code': 3, 'code_example': 3, 'internal_code': 2}
    }
   
    if query_type in alignment_bonus and doc_type in alignment_bonus[query_type]:
        score += alignment_bonus[query_type][doc_type]
   
    # Requirement matching
    if analysis.get('requires_code') and 'code' in doc_type:
        score += 1.5
    if analysis.get('requires_theory') and doc_type == 'paper':
        score += 1.5
    if analysis.get('requires_examples') and 'example' in doc_type:
        score += 1.5
   
    # Key topic relevance
    key_topics = analysis.get('key_topics', [])
    content_lower = doc.page_content.lower()
   
    # Weight matches: exact match in metadata > content match
    metadata_matches = 0
    content_matches = 0
   
    for topic in key_topics:
        topic_lower = topic.lower()
        # Check metadata fields
        if any(topic_lower in str(v).lower() for v in doc.metadata.values()):
            metadata_matches += 1
        elif topic_lower in content_lower:
            content_matches += 1
   
    score += min(2.0, metadata_matches * 0.8)
    score += min(1.0, content_matches * 0.3)  
   
    # Complexity alignment
    query_complexity = analysis.get('complexity', 'intermediate')
    doc_complexity = doc.metadata.get('complexity', 'intermediate')
   
    complexity_order = ['beginner', 'intermediate', 'advanced']
    if query_complexity in complexity_order and doc_complexity in complexity_order:
        q_idx = complexity_order.index(query_complexity)
        d_idx = complexity_order.index(doc_complexity)
       
        if q_idx == d_idx:
            score += 1.5
        elif abs(q_idx - d_idx) == 1:
            score += 0.75
   
    # quality bonus
    quality = doc.metadata.get('quality', 'medium')
    quality_bonus = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
    score += quality_bonus.get(quality, 0.5)
   
    return score


# ENHANCED DOCUMENT FORMATTING

def format_docs_with_sources(docs: List[Document]) -> str:
    """Format retrieved documents with relevance context."""
    formatted = []
   
    for i, doc in enumerate(docs, 1):
        doc_type = doc.metadata.get("type", "unknown")
        source = doc.metadata.get("source", "unknown")
       
        # Format header based on document type
        if doc_type == "paper":
            paper_title = doc.metadata.get("paper_title", "Unknown Paper")
            page = doc.metadata.get("approx_page", doc.metadata.get("page_number", "?"))
            header = f"Paper: {paper_title} (Page ~{page})"
       
        elif doc_type in ["source_code", "internal_code"]:
            module = doc.metadata.get("module", "unknown")
            func_name = doc.metadata.get("function_name", "")
            visibility = "Internal" if doc_type == "internal_code" else "Public"
            header = f" {visibility} Code: {module}.{func_name}"
       
        elif doc_type == "api_doc":
            func_name = doc.metadata.get("function", "unknown")
            header = f"API Doc: mcdc.{func_name}"
       
        elif "example" in doc_type:
            test_name = doc.metadata.get("test_name", "unknown")
            complexity = doc.metadata.get("complexity", "unknown")
            workflow = doc.metadata.get("workflow_group", "")
            if workflow:
                header = f"Example: {test_name} ({complexity}) - {workflow.title()}"
            else:
                header = f"Example: {test_name} ({complexity})"
       
        else:
            header = f"📋 {source}"
       
        # Format content with clear separation
        content = doc.page_content.strip()
        if len(content) > 1500:
            content = content[:1500] + "\n\n[Content truncated...]"
       
        formatted.append(f"[Source {i}] {header}\n\n{content}")
   
    return "\n\n" + "─" * 70 + "\n\n".join(formatted)




# ENHANCED QA CHAIN


def create_enhanced_qa_chain(
    vectorstore: Chroma,
    llm: ChatGoogleGenerativeAI,
    analysis: Dict[str, Any]
):
    """
    Create RAG chain with query expansion and hybrid retrieval.
    """
   
    # Custom retriever with query expansion
    def retrieve_with_expansion(question: str) -> str:
       
        docs = retrieve_and_rerank(vectorstore, question, analysis)
        return format_docs_with_sources(docs)
   
    # prompt with query context
    prompt = PromptTemplate.from_template("""You are MCDC-Expert, an AI assistant specializing in the Monte Carlo / Dynamic Code (MCDC) particle transport simulator.


QUERY CONTEXT:
- Type: {query_type}
- Complexity: {complexity}
- User needs: {requirements}
- Key topics: {key_topics}


Your role is to provide a {complexity}-level answer about {query_type}.


INSTRUCTIONS:
1. Answer based ONLY on the provided context documents
2. Tailor your explanation to the {complexity} level:
   - Beginner: Focus on concepts, workflow, and high-level understanding
   - Intermediate: Include implementation details, best practices, and common patterns
   - Advanced: Deep-dive into algorithms, architecture decisions, and optimizations
3. **ALWAYS cite your sources** using format: "According to [Source N]..." or "As shown in [Source N]..."
4. Include relevant code snippets when available
5. If context is insufficient, clearly state what information is missing
6. For architecture questions, explain the "why" behind design decisions
7. For usage questions, provide step-by-step guidance with examples
8. Connect related concepts when relevant


Question: {question}


Context Documents:
{context}


Answer:""")
   
    # Prepare context strings
    requirements = []
    if analysis.get('requires_code'):
        requirements.append("source code")
    if analysis.get('requires_theory'):
        requirements.append("physics theory")
    if analysis.get('requires_examples'):
        requirements.append("usage examples")
    requirements_str = ", ".join(requirements) if requirements else "general explanation"
   
    key_topics_str = ", ".join(analysis.get('key_topics', [])[:5])
   
    # Build chain
    chain = (
        {
            "context": retrieve_with_expansion,
            "question": RunnablePassthrough(),
            "query_type": lambda _: analysis['query_type'],
            "complexity": lambda _: analysis['complexity'],
            "requirements": lambda _: requirements_str,
            "key_topics": lambda _: key_topics_str
        }
        | prompt
        | llm
        | StrOutputParser()
    )
   
    return chain




# CLI INTERFACE


def print_welcome():
    """Print welcome message."""
    print_header("MCDC Q&A Agent - Enhanced RAG System")
    print(f"{Colors.BOLD}Features:{Colors.END}")
    print("  • LLM-powered query understanding")
    print("  • Multi-factor relevance scoring")
    print("  • Context-aware responses")
    print(f"\n{Colors.YELLOW}Type 'quit' or 'exit' to leave{Colors.END}")
    print(f"{Colors.YELLOW}Type 'help' for tips on asking good questions{Colors.END}\n")




def print_help():
    """Print help message with query examples."""
    print_header("How to Ask Good Questions")
   
    print(f"{Colors.BOLD}Theory Questions:{Colors.END}")
    print("  • 'Explain the Monte Carlo algorithm used in MCDC'")
    print("  • 'What physics methods are implemented for neutron transport?'")
    print("  • 'How does MCDC handle cross-section data?'")
   
    print(f"\n{Colors.BOLD}Architecture Questions:{Colors.END}")
    print("  • 'How is the particle tracking implemented internally?'")
    print("  • 'Explain the kernel architecture and main loop'")
    print("  • 'What data structures are used for geometry?'")
   
    print(f"\n{Colors.BOLD}Implementation Questions:{Colors.END}")
    print("  • 'Show me how the eigenmode solver is implemented'")
    print("  • 'Where is the tally scoring logic located?'")
    print("  • 'How does the _kernel module work?'")
   
    print(f"\n{Colors.BOLD}Usage Questions:{Colors.END}")
    print("  • 'How do I create a multigroup material?'")
    print("  • 'Show me an example of setting up mesh tallies'")
    print("  • 'What are the parameters for mcdc.source()?'")
   
    print(f"\n{Colors.BOLD}Tips:{Colors.END}")
    print("  • Be specific about what you want to know")
    print("  • Mention relevant MCDC concepts (materials, tallies, etc.)")
    print("  • Specify if you want theory, implementation, or usage info\n")




def interactive_mode():
    """Run interactive Q&A loop."""
    print_welcome()
    
    try:
        print(f"{Colors.YELLOW}Loading RAG system...{Colors.END}")
        vectorstore = load_vectorstore()
        llm = load_llm()
        print_success("System ready!")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)
   
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
       
        print(f"\n{Colors.YELLOW}Analyzing query...{Colors.END}")
        analysis = analyze_query_with_llm(question, llm)
       
        print(f"{Colors.YELLOW}Query Analysis:{Colors.END}")
        print(f"   Type: {analysis['query_type']}")
        print(f"   Complexity: {analysis['complexity']}")
        print(f"   Topics: {', '.join(analysis.get('key_topics', [])[:5])}")
        print(f"   Reasoning: {analysis['reasoning']}\n")
        chain = create_enhanced_qa_chain(vectorstore, llm, analysis)
       
        try:
            print(f"{Colors.BOLD}{Colors.BLUE}Answer:{Colors.END}\n")
           
            for chunk in chain.stream(question):
                print(chunk, end="", flush=True)
           
            print("\n")
       
        except Exception as e:
            print_error(f"Failed to generate answer: {e}")
            import traceback
            traceback.print_exc()




def single_question_mode(question: str):
    """Answer a single question and exit."""
    try:
        vectorstore = load_vectorstore()
        llm = load_llm()
    except Exception as e:
        print_error(str(e))
        sys.exit(1)
    
    analysis = analyze_query_with_llm(question, llm)
    chain = create_enhanced_qa_chain(vectorstore, llm, analysis)
   
    try:
        print(f"{Colors.BOLD}{Colors.BLUE}Answer:{Colors.END}\n")
       
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
       
        print("\n")
   
    except Exception as e:
        print_error(f"Failed to generate answer: {e}")
        sys.exit(1)




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




def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        single_question_mode(question)
    else:
        interactive_mode()




if __name__ == "__main__":
    main()
