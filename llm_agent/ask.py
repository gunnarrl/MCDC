import argparse
from .utils import load_llm, load_retriever, create_rag_chain

QA_PROMPT_TEMPLATE = """
You are an expert AI assistant for the MCDC (Monte Carlo/Deterministic Code) project.
Your goal is to answer questions based *only* on the provided context.
If the context does not contain the answer, state that clearly.
Do not make up information.
Be concise, helpful, and professional.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def main():

    
    parser = argparse.ArgumentParser(
        description="Ask questions about the MCDC codebase."
    )

    parser.add_argument(
        "question", 
        type=str, 
        nargs="+", 
        help="The question you want to ask."
    )
    args = parser.parse_args()
    
    question = " ".join(args.question)

    try:

        llm = load_llm()
        retriever = load_retriever()
        
        # create the specific RAG chain for Q&A
        qa_chain = create_rag_chain(llm, retriever, QA_PROMPT_TEMPLATE)
        
        # invoke the chain and print the response
        print("Finding an answer...")
        response = qa_chain.invoke(question)
        print("\n--- Answer ---")
        print(response)
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your GEMINI_API_KEY is set and build_index.py has been run.")

if __name__ == "__main__":
    main()