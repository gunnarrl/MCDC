# onboarding/tutor.py

import sys
import os
import re
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import your utilities (use these, don't reinvent)
from llm_agent.utils import load_llm, load_retriever, format_docs
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ─── CONFIGURATION ───
CHROMA_PATH = os.path.join(project_root, "vectorstore")
COLLECTION_NAME = "mcdc_docs"
RTD_DOCS_PATH = os.path.join(project_root, "scraped_docs", "function_docs.json")


def load_rtd_signatures() -> Dict[str, Any]:
    """Load function signatures from RTD docs."""
    try:
        with open(RTD_DOCS_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ RTD docs not found at {RTD_DOCS_PATH}")
        return {}


RTD_SIGNATURES = load_rtd_signatures()


def get_vectorstore():
    """Initialize ChromaDB collection."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError("❌ chromadb not found. Install with: pip install chromadb")
    
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(
            f"❌ Vectorstore not found at {CHROMA_PATH}. Run build_index.py first."
        )
    
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(f"❌ Collection '{COLLECTION_NAME}' not found: {e}")


def format_search_results(results: Dict) -> List[Dict]:
    """Format ChromaDB results to standard list."""
    formatted = []
    if (results and 'documents' in results and results['documents'] and 
        len(results['documents']) > 0 and results['documents'][0]):
        
        for i, doc in enumerate(results['documents'][0]):
            result = {
                'content': doc,
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else 0
            }
            formatted.append(result)
    return formatted


def search_docs(query: str, filter_metadata: Optional[Dict[str, Any]] = None, n_results: int = 1) -> List[Dict]:
    """Search vectorstore with fallback to unfiltered search."""
    collection = get_vectorstore()
    
    # Try filtered search first
    if filter_metadata:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            formatted = format_search_results(results)
            if formatted:
                return formatted
            print(f"⚠️ Filtered search for '{query}' returned no results")
        except Exception as e:
            print(f"⚠️ Filtered search failed: {e}")
    
    # Fallback to unfiltered
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return format_search_results(results)
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []


def extract_relevant_snippet(code: str, step_name: str, user_goal: str) -> str:
    """
    Extract only the lines relevant to the current step from a full example.
    Shows ±3 lines around each mcdc.<step_name> call.
    """
    lines = code.split('\n')
    pattern = rf"mcdc\.{step_name}[a-zA-Z_]*\s*\("
    
    relevant_blocks = []
    for i, line in enumerate(lines):
        if re.search(pattern, line, re.IGNORECASE):
            start = max(0, i - 3)
            end = min(len(lines), i + 4)
            
            block = f"  # ... lines {start+1}-{end} ...\n"
            block += '\n'.join(f"  {l}" for l in lines[start:end])
            block += f"\n  # ... end block ...\n"
            
            relevant_blocks.append(block)
    
    if relevant_blocks:
        summary = f"📌 Found {len(relevant_blocks)} relevant block(s):\n\n"
        return summary + f"\n{'='*50}\n".join(relevant_blocks)
    
    return fallback_snippet(code, step_name, user_goal)


def fallback_snippet(code: str, step_name: str, user_goal: str) -> str:
    """Show imports + first 15 code lines when no direct matches."""
    lines = code.split('\n')
    imports = [l for l in lines if l.strip().startswith(('import', 'from'))][:5]
    code_lines = [l for l in lines if l.strip() and not l.startswith('#')][:15]
    
    snippet = f"⚠️ No direct matches for '{step_name}'\n"
    snippet += "Showing file start:\n\n"
    if imports:
        snippet += "**Imports:**\n" + '\n'.join(imports) + "\n\n"
    snippet += "**Code:**\n" + '\n'.join(code_lines)
    return snippet

def extract_function_calls_summary(code: str, step_name: str) -> str:
    """
    Extract ALL function calls for the step and return a summary.
    """
    lines = code.split('\n')
    pattern = rf"mcdc\.{step_name}[a-zA-Z_]*\s*\("
    
    function_calls = []
    for line in lines:
        if re.search(pattern, line, re.IGNORECASE):
            # Clean up the line
            cleaned = line.strip()
            # Remove inline comments
            if '#' in cleaned:
                cleaned = cleaned.split('#')[0].strip()
            if cleaned:
                function_calls.append(cleaned)
    
    if not function_calls:
        return f"No {step_name} function calls found in this example."
    
    # Build summary
    unique_calls = len(set(function_calls))
    total_calls = len(function_calls)
    
    if total_calls == 1:
        return f"Makes 1 call: {function_calls[0]}"
    
    if total_calls <= 3:
        # Show all calls
        calls_str = "\n  • ".join(function_calls)
        return f"Makes {total_calls} call(s):\n  • {calls_str}"
    
    # For many calls, show count and first few examples
    preview = "\n  • ".join(function_calls[:3])
    return f"Makes {total_calls} call(s) ({unique_calls} unique). Examples:\n  • {preview}\n  • ... and {total_calls - 3} more"
    
def get_rtd_signature(step_name: str) -> str:
    """Get official signature from RTD docs."""
    if step_name in RTD_SIGNATURES:
        return RTD_SIGNATURES[step_name].get('signature', f'mcdc.{step_name}(...)')
    return f"mcdc.{step_name}(...)"


def create_explanation_chain():
    """Create RAG chain using YOUR utilities."""
    try:
        llm = load_llm(temperature=0.1, model="gemini-2.5-flash")
        retriever = load_retriever(k=2)
    except Exception as e:
        print(f"⚠️  Chain init failed: {e}")
        return None
    
    prompt = PromptTemplate.from_template("""
You are MCDC-Tutor. Explain this code snippet for a reactor physicist.
USER GOAL: {query}
SNIPPET:
{context}
Explain in 3-5 sentences what this code does and which parameters matter.
DO NOT generate new code. Only explain what was retrieved.
""")
    
    return (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )


def explain_with_llm(snippet: str, step_name: str, user_goal: str) -> str:
    """Generate LLM explanation (optional, can be disabled)."""
    chain = create_explanation_chain()
    if not chain:
        return f"Look for mcdc.{step_name}() calls in the example."
    
    try:
        query = f"{step_name} {user_goal}"
        return chain.invoke(query)
    except Exception as e:
        print(f"⚠️ LLM explain failed: {e}")
        return f"Example shows {step_name} usage. Review lines with mcdc.{step_name}()."


def explain_concept(step_name: str, user_goal: str) -> Tuple[str, str]:
    """
    Search for examples and return RELEVANT SNIPPETS + explanation.
    Uses RTD signatures, not regex extraction.
    """
    valid_steps = ["material", "surface", "geometry", "source", "tally", "settings", "run"]
    if step_name not in valid_steps:
        return (f"# Invalid step: {step_name}", f"Valid: {', '.join(valid_steps)}")
    
    query = f"{step_name} {user_goal}".strip()
    results = search_docs(query, filter_metadata={"step": step_name}, n_results=1)
    
    if not results:
        return (
            f"# No example for '{user_goal}'",
            f"Try rephrasing or skip this step."
        )
    
    full_code = results[0]['content']
    metadata = results[0].get('metadata', {})
    
    # Extract snippet (not full file)
    snippet = extract_relevant_snippet(full_code, step_name, user_goal)
    
    # Build explanation using RTD signature
    signature = get_rtd_signature(step_name)
    test_name = metadata.get('test_name', metadata.get('function', 'unknown'))
    relevance = 1 - results[0].get('distance', 0)
    
    explanation = f"**Step: {step_name.title()}**\n\n"
    explanation += f"**Function**: `{signature}`\n\n"
    explanation += f"**Example**: {test_name}\n"
    explanation += f"**Relevance**: {relevance:.1%}\n\n"
    explanation += f"**Summary**: {extract_function_calls_summary(full_code, step_name)}\n\n"
    explanation += f"**Explanation**: {explain_with_llm(snippet, step_name, user_goal)}"
    
    return snippet, explanation


class OnboardingSession:
    """Track onboarding state and script building."""
    
    def __init__(self):
        self.steps = ["material", "surface", "geometry", "source", "tally", "settings", "run"]
        self.current_step_index = 0
        self.user_choices: List[Dict] = []
        self.script_lines: List[str] = []
    
    def get_current_step(self) -> str:
        return self.steps[self.current_step_index]
    
    def is_complete(self) -> bool:
        return self.current_step_index >= len(self.steps)
    
    def next_step(self):
        self.current_step_index += 1
    
    def record_choice(self, step: str, choice: str, details: str = ""):
        self.user_choices.append({"step": step, "choice": choice, "details": details})


def interactive_onboarding():
    """Main tutor loop walking user through 7-step workflow."""
    session = OnboardingSession()
    
    print("\n" + "="*70)
    print("🎓 MCDC-TUTOR INTERACTIVE ONBOARDING")
    print("="*70)
    print("I'll guide you through building a complete MCDC simulation.\n")
    
    user_goal = input("What to simulate? (e.g., 'water sphere with 1 MeV source'): ").strip()
    if not user_goal:
        user_goal = "basic simulation"
    
    print(f"\nBuilding: {user_goal}")
    print("Steps: materials → surfaces → geometry → sources → tallies → settings → run\n")
    
    while not session.is_complete():
        step = session.get_current_step()
        step_num = session.current_step_index + 1
        
        print(f"\n{'='*70}")
        print(f"STEP {step_num}/7: {step.upper()}")
        print("="*70)
        
        what_to_define = input(f"What do you want for '{step}'? (Enter to see examples): ").strip()
        if not what_to_define:
            what_to_define = step
        
        print(f"\n🔍 Searching examples for '{what_to_define}'...")
        snippet, explanation = explain_concept(step, what_to_define)
        
        print("\n📄 EXAMPLE SNIPPET:")
        print("-" * 50)
        print(snippet)
        print("-" * 50)
        
        print(f"\n💬 EXPLANATION:")
        print(explanation)
        
        # Choice loop
        while True:
            print(f"\n{'='*50}")
            print("[1] Show another example  [2] Define it yourself")
            print("[3] Skip this step        [q] Quit")
            print("="*50)
            
            choice = input("Choose: ").strip().lower()
            
            if choice == "1":
                print("⚠️ Phase 4 will add pagination. Showing same example.")
                continue
            
            elif choice == "2":
                template = f"# Define your {step} here\n# Based on example above:\n"
                if step in RTD_SIGNATURES:
                    template += f"# Function: {RTD_SIGNATURES[step]['signature']}\n\n"
                template += f"# Your code:\n"
                
                edited = open_editor(template, f"mcdc_{step}.py")
                if edited and edited.strip() != template.strip():
                    session.script_lines.append(f"\n# --- {step.upper()} ---\n")
                    session.script_lines.append(edited)
                    session.record_choice(step, "define", "User defined")
                    print("✅ Saved!")
                else:
                    print("⚠️ No changes. Skipping.")
                    session.record_choice(step, "skip", "No definition")
                break
            
            elif choice == "3":
                print(f"⏭️ Skipping {step}")
                session.record_choice(step, "skip", "User skipped")
                break
            
            elif choice == "q":
                print("\n❌ Onboarding cancelled.")
                return session
            
            else:
                print("❌ Invalid choice. Enter 1, 2, 3, or q.")
        
        session.next_step()
    
    # Completion
    print("\n" + "="*70)
    print("🎉 ONBOARDING COMPLETE!")
    print("="*70)
    
    completed = len([c for c in session.user_choices if c['choice'] == 'define'])
    print(f"\nSteps completed: {completed}/7")
    print("\nYour path:")
    for choice in session.user_choices:
        print(f"  {choice['step']}: {choice['choice']}")
    
    if session.script_lines:
        save = input("\nSave script? (y/n): ").strip().lower()
        if save == 'y':
            filename = input("Filename [mcdc_simulation.py]: ").strip() or "mcdc_simulation.py"
            
            script = [
                "#!/usr/bin/env python3",
                f'"""MCDC Script - {user_goal}"""\n',
                "import mcdc",
                "import numpy as np\n",
                "# Review before running!",
            ] + session.script_lines
            
            Path(filename).write_text('\n'.join(script))
            print(f"✅ Saved to {filename}")
            
            # Syntax check
            try:
                subprocess.run(["ruff", "check", "--quiet", filename], check=True)
                print("✅ No syntax errors")
            except:
                print("⚠️ Syntax issues found")
    
    return session


def open_editor(template: str, filename: str = "temp.py") -> str:
    """Open temp file in user's editor."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(template)
        temp_path = f.name
    
    try:
        editor = os.environ.get('EDITOR', 'nano')
        print(f"\n📝 Opening {temp_path} with {editor}...")
        subprocess.run([editor, temp_path])
        
        with open(temp_path, 'r') as f:
            return f.read()
    
    except Exception as e:
        print(f"⚠️ Editor failed: {e}")
        return ""
    
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


def test_interactive_onboarding():
    """Verify session structure without manual input."""
    session = OnboardingSession()
    
    assert session.current_step_index == 0
    assert session.get_current_step() == "material"
    assert not session.is_complete()
    assert len(session.steps) == 7
    
    for i in range(7):
        session.next_step()
    
    assert session.is_complete()
    assert session.current_step_index == 7
    
    print("✅ Session structure test PASSED")
    return True


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_interactive_onboarding()
    else:
        interactive_onboarding()