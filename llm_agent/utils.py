import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = os.path.join("llm_agent", "db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_llm():

    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set it with: export GEMINI_API_KEY='your-api-key-here'"
        )
        
    # using gemini 1.5 flash, more powerful model could be used if needed
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=api_key)

def load_retriever(db_path: str = DB_PATH):

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Vector store not found at {db_path}. "
            "Did you run build_index.py first?"
        )
        
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"}
    )
    
    vectordb = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )
    
    # retrieve the top 5 most relevant chunks
    return vectordb.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    
    # formats a list of Document objects into a single string.
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(llm, retriever, prompt_template_str: str):

    prompt = PromptTemplate.from_template(prompt_template_str)
    
    # standard LCEL "pipe" for a RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain