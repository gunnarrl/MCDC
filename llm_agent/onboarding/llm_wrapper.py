import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from llm_agent.utils import load_llm, load_retriever, format_docs
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_explanation_chain():
    try:
        model_name = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        llm = load_llm(temperature=0.1, model=model_name)
        retriever = load_retriever(k=2)
    except Exception as e:
        print(f"⚠️  Chain initialization failed: {e}")
        return None
    
    prompt = PromptTemplate.from_template("""
You are MCDC-Tutor. Your task is to EXPLAIN a code example retrieved from documentation.
You MUST NOT generate new code. Only explain what was found.

USER GOAL: {user_goal}
WORKFLOW STEP: {step_name}

RETRIEVED EXAMPLES:
{context}

INSTRUCTIONS:
1. Explain what this code does in plain English
2. Identify key parameters relevant to "{user_goal}"
3. Point out any dependencies or required previous steps
4. Suggest what the user should modify
5. Keep it technical and concise (3-5 sentences)

IMPORTANT: If the examples aren't relevant to "{user_goal}", say so and suggest better search terms.
DO NOT hallucinate function signatures or parameters.
""")
    
    return (
        {"context": retriever | format_docs, "user_goal": RunnablePassthrough(), "step_name": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

def explain_with_llm(code_example: str, step_name: str, user_goal: str) -> str:
    chain = create_explanation_chain()
    if not chain:
        print("⚠️  LLM not available, using fallback explanation")
        return f"Example demonstrates {step_name} setup. Focus on lines containing 'mcdc.{step_name}'."
    
    try:
        return chain.invoke(f"{step_name} {user_goal}")
    except Exception as e:
        print(f"⚠️  LLM explanation failed: {e}")
        return f"Example demonstrates {step_name} setup. Review code for patterns."
