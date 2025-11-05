import argparse
import sys
from llm_agent.utils import load_llm, load_retriever, create_rag_chain, create_qa_chain

QA_PROMPT_TEMPLATE = """You are an expert AI assistant for the MCDC (Monte Carlo/Deterministic Code) project, a Python-based nuclear reactor physics simulation tool.

Your role is to help users understand and work with MCDC by answering questions based on the provided documentation, code, and examples.

IMPORTANT GUIDELINES:
- Answer questions using ONLY the information in the provided context
- explain as well as possible, if the context doesn't contain enough information, clearly state this and suggest what the user might search for instead
- Be specific and cite relevant file paths or code sections when applicable
- For code questions, provide concrete examples when possible
- If asked about simulation setup, refer to specific input file examples from the context
- Be concise but thorough - nuclear physics is complex, so explain technical terms when needed

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about the MCDC codebase using RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python -m llm_agent.ask "How do I set up a k-eigenvalue problem?"
            python -m llm_agent.ask "What is the difference between fixed source and eigenvalue modes?"
            python -m llm_agent.ask "How do I define materials in MCDC?"
            python -m llm_agent.ask -v "How do I create a material in MCDC?"
        """
    )
    
    parser.add_argument(
        "question", 
        type=str, 
        nargs="+", 
        help="The question you want to ask about MCDC"
    )
    
    parser.add_argument(
        "-k", "--num-results",
        type=int,
        default=5, 
        help="Number of document chunks to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show retrieved source documents"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)"
    )
    
    args = parser.parse_args()
    question = " ".join(args.question)

    try:
        # load components
        print("Loading AI model...")
        llm = load_llm(model=args.model)
        
        print("Loading knowledge base...")
        retriever = load_retriever(k=args.num_results)
        
        # show retrieved documents if verbose
        if args.verbose:
            print("\n📚 Retrieving documents...")
            docs = retriever.invoke(question)
            
            # print the retrieved docs
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                source = source.replace("llm_agent/corpus/", "")
                preview = doc.page_content[:200].replace("\n", " ")
                print(f"\n  [{i}] {source}")
                print(f"      {preview}...")
            print()

            formatted_context = format_docs(docs)
        
            qa_chain = create_qa_chain(llm, QA_PROMPT_TEMPLATE)
            response = qa_chain.invoke({
                "context": formatted_context,
                "question": question
            })
            
        else:
            
            qa_chain = create_rag_chain(llm, retriever, QA_PROMPT_TEMPLATE)
            response = qa_chain.invoke(question)

        print("ANSWER")
        print("=" * 80)

        print(response)
    
    except Exception as e:
        print(f"\nAn unexpected error occurred (Check GEMINI_API_KEY): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()