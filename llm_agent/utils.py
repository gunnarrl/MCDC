import os
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = os.path.join("llm_agent", "db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_llm(temperature=0, model="gemini-2.5-flash"):

    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please set it with: export GEMINI_API_KEY='your-api-key-here'"
        )
    
    return ChatGoogleGenerativeAI(
        model=model, 
        temperature=temperature,
        google_api_key=api_key
    )

@lru_cache(maxsize=1)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, 
        model_kwargs={"device": "cpu"}
    )

def load_retriever(db_path: str = DB_PATH, k: int = 3):

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Vector store not found at {db_path}. "
            "Please run: python llm_agent/build_index.py"
        )
    
    embeddings = get_embeddings()
    
    vectordb = Chroma(
        persist_directory=db_path, 
        embedding_function=embeddings
    )
    
    # retrieve the top k most relevant chunks
    return vectordb.as_retriever(search_kwargs={"k": k})

def format_docs(docs):

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        source = source.replace("llm_agent/corpus/", "")
        content = doc.page_content
        formatted.append(f"[Source {i}: {source}]\n{content}")
    
    return "\n\n---\n\n".join(formatted)

def create_rag_chain(llm, retriever, prompt_template_str: str):

    prompt = PromptTemplate.from_template(prompt_template_str)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def create_qa_chain(llm, prompt_template_str: str):

    prompt = PromptTemplate.from_template(prompt_template_str)
    
    qa_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain