from llm_agent.onboarding.concepts import CONCEPT_LESSONS
from llm_agent.utils import load_llm, load_retriever, create_rag_chain_with_prompt
from llm_agent.onboarding.script_builder import ScriptBuilder
from llm_agent.onboarding.tools import get_mcdc_tools
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MCDCTutor:
    """
    Interactive tutor for learning MCDC through a 7-step workflow.
    
    Each step:
    1. Teaches the concept (from hardcoded curriculum)
    2. Shows relevant examples (from RAG with step filtering)
    3. Answers questions (using step-specific RAG chain)
    4. Helps user create their version (using agent + tools)
    5. Validates and tracks state (using ScriptBuilder)
    """
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 3: Step-specific keywords for query expansion
    # Maps each workflow step to relevant technical terms that improve retrieval
    # ─────────────────────────────────────────────────────────────────────────────
    STEP_KEYWORDS = {
        "material": "MaterialMG Material capture scatter fission nuclide_composition density multi-group continuous-energy cross-section macroscopic",
        "surface": "Surface PlaneX PlaneY PlaneZ CylinderX CylinderY CylinderZ Sphere boundary_condition vacuum reflective interface geometry normal",
        "cell": "Cell region fill boolean operators intersection union & | translation rotation universe lattice root_universe",
        "source": "Source position energy direction isotropic white_direction time energy_group spectrum point_source volume_source uniform",
        "tally": "TallyMesh TallyCell TallySurface scores flux collision fission net-current mesh energy_bins detector mu_bins",
        "settings": "settings N_particle N_batch eigenmode census output population_control variance_reduction convergence active inactive",
        "run": "run execute simulate output h5 tally results k-effective convergence statistics batch cycle"
    }
    
    def __init__(self, llm, input_func=input):
        # Create our own builder instance
        self.builder = ScriptBuilder()
        self.input_func = input_func
        
        self.doc_filter = {
            "type": {
                # $nin means "Not In"
                "$nin": ["paper", "source_code", "internal_code"]
            }
        }

        # Set up retriever and RAG chain for Q&A
        self.retriever = load_retriever(k=5, search_filter=self.doc_filter)      
        
        rag_prompt = """You are MCDC-Tutor, explaining Monte Carlo particle transport concepts to beginners.

Question: {question}

Documentation: {context}

Provide a helpful answer that includes:
1. A simple code example (if relevant)
2. Clear explanation of key concepts (assume NO nuclear physics background)
3. Brief description of required parameters if applicable

Use friendly, educational tone. Keep it concise but informative.

Answer:"""
        self.rag_prompt = rag_prompt  
        
        self.rag_chain = create_rag_chain_with_prompt(llm, self.retriever, rag_prompt)
        
        try:
            self.tools = get_mcdc_tools(self.builder, self.retriever)

            self.llm = llm
            
            system_prompt = """You are MCDC-Tutor, an expert assistant for creating Monte Carlo particle transport simulations using the MCDC Python package.

---
## CRITICAL MATERIAL MODE INSTRUCTIONS (MG is DEFAULT)

**Multi-Group (MG) is the DEFAULT mode.** Continuous-Energy (CE) should ONLY be used if the user explicitly asks for "continuous energy" or "CE".

### PROTOCOL FOR CREATING MATERIALS (MG MODE)
When the user asks to create a material:

1.  **ANALYZE**: Did the user explicitly provide cross-section values (e.g., "capture=[1.0]")?
    * **YES (Explicit)**: Use **ONLY** the provided parameters. Do **NOT** add unrequested physics (e.g., if User says "capture=[1.0]", do NOT add Scatter).
    * **NO (Abstract)**: If the user only gives a name (e.g., "create water"), estimate reasonable physics values (Capture + Scatter).

2.  **PROPOSE**: Display the values you intend to use **BEFORE** calling any tools.
    * *Explicit Case*: "You specified Capture=[1.0]. I will create the material with just that. Does this look correct?"
    * *Abstract Case*: "For Water, I suggest Capture=[0.01], Scatter=[0.8]. Does this look correct?"

3.  **CONFIRM/EDIT**: Wait for the user to say "Yes" or provide different numbers.

4.  **EXECUTE**: Once confirmed, call `create_material_from_formula` using `mode="MG"` and the agreed-upon arrays.

### PROTOCOL FOR CE MODE (Only if requested)
1.  If the user explicitly asks for CE, you must calculate atomic composition.
2.  Remind them that `MCDC_XSLIB` is required.

---
## WORKFLOW FOR EVERY REQUEST

1.  **Check Status:** Call `get_current_script()` to check what entities already exist.
2.  **Search Docs (If Needed):** If the user asks "how to do X", use `search_docs(query)` before trying to call other tools.
3.  **Clarify:** If the request is ambiguous (e.g., "boundary condition"), ask a clarifying question.
4.  **Tool Call:** Call the appropriate tool.
5.  **Explain:** Explain what you created and why.

---
## TOOL PARAMETER FORMAT

-   All array parameters **MUST be strings**: e.g., `capture="[1.0]"` NOT `capture=[1.0]`.
-   2D arrays: e.g., `scatter="[[0.8, 0.1], [0.05, 0.85]]"`.
-   Surfaces params: e.g., `params="x=0.0"` or `params="center=[0.0, 0.0], radius=1.5"`.
"""
        
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt 
            )
        except Exception as e:
            print(f"{Colors.FAIL}Warning: Failed to initialize agent properly: {e}{Colors.ENDC}")
            print("Tutor may not function correctly. Check API keys and dependencies.")
            raise
    
    # Query expansion for better retrieval
    def expand_query(self, query: str, step: str) -> str:
        """
        Expand query with step-specific keywords for better retrieval.
        This helps find relevant documents even when user uses non-technical language.
        """
        # Start with step-specific context
        base_query = f"{step} {query}"
        
        # Append relevant keywords from our keyword mapping
        keywords = self.STEP_KEYWORDS.get(step, "")
        
        return f"{base_query} {keywords}"
    
    # Create step-filtered retriever so we only get examples relevant to the current step
    def get_step_retriever(self, step: str):
        """
        Create a retriever that filters by workflow step.
        This is crucial for showing users only relevant examples during each step.
        """
        try:
            # Access the underlying vectorstore from our existing retriever
            vectorstore = self.retriever.vectorstore
            
            step_filter = {
                "$and": [
                    self.doc_filter,     
                    {"section": step}    
                ]
            }

            return vectorstore.as_retriever(
                search_kwargs={
                    "k": 5, 
                    "filter": step_filter 
                }
            )

        except AttributeError:
            print(f"Debug: Could not access vectorstore for step filtering, using general retriever")
            return self.retriever
    
    def teach_concept(self, step: str) -> bool:
        """
        Run the 5-part mini-lesson for a given step.
        
        Returns:
            bool: True if user is ready to create, False if they want to skip
        """
        lesson = CONCEPT_LESSONS[step]
        
        print(f"\n{Colors.HEADER}{'='*60}")
        print(f"STEP: {step.upper()}")
        print(f"{'='*60}{Colors.ENDC}\n")
        print(f"{Colors.BOLD}Concept:{Colors.ENDC}\n{lesson['concept']}\n")
        print(f"{Colors.BOLD}Key parts:{Colors.ENDC}\n{lesson['parts']}\n")
        
        # Show step-specific examples from RAG
        try:
            step_retriever = self.get_step_retriever(step)
            expanded_query = self.expand_query("beginner simple example", step)
            examples = step_retriever.invoke(expanded_query)
            
            if examples:
                print(f"\n{Colors.CYAN}Example from regression tests:")
                print("-" * 60)
                # Show first 500 chars of first example
                print(examples[0].page_content[:500])
                if len(examples[0].page_content) > 500:
                    print("...")
                print("-" * 60 + f"{Colors.ENDC}")
        except Exception as e:
            print(f"\n(Could not load example: {e})")
        
        print(f"\n{Colors.YELLOW}Common questions about {step}:{Colors.ENDC}")
        for i, q in enumerate(lesson.get('key_questions', [])[:3], 1):
            print(f"  {i}. {q}")
        
        step_rag_chain = create_rag_chain_with_prompt(
            self.llm,
            self.get_step_retriever(step),
            self.rag_prompt
        )
        
        while True:
            q = self.input_func(f"\n{Colors.BOLD}Ask a question...").strip()
            if not q:
                break
            
            try:
                # Expand user query
                expanded_q = self.expand_query(q, step)
                
                print(f"\n{Colors.BLUE}TUTOR: ", end="", flush=True)
                
                for chunk in step_rag_chain.stream(expanded_q):
                    print(chunk, end="", flush=True)
                
                print(f"{Colors.ENDC}\n")
            except Exception as e:
                print(f"\n{Colors.FAIL}Error: {e}{Colors.ENDC}")
        
        # Check if ready to create
        ready = input(f"\nReady to create your {step}? [Y/n]: ").strip().lower()
        return ready != "n"
    
    def _parse_agent_response(self, response) -> str:
        """Handle all response formats: strings, dicts, AIMessage, content blocks."""
        if isinstance(response, str):
            return response
            
        if isinstance(response, dict):
            if 'output' in response:
                return response['output']
            if 'content' in response:
                return response['content']
            if 'messages' in response and response['messages']:
                return self._extract_message_content(response['messages'][-1])
                
        # Handle object with .content attribute (AIMessage)
        if hasattr(response, 'content'):
            return response.content
            
        # Handle list (e.g., list of messages or content blocks)
        if isinstance(response, list):
            # Try to extract text from blocks or join string representations
            try:
                text_parts = []
                for block in response:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif hasattr(block, 'content'):
                        text_parts.append(block.content)
                    else:
                        text_parts.append(str(block))
                return '\n'.join(text_parts)
            except Exception:
                return str(response)

        return str(response)

    def _extract_message_content(self, message):
        """Extract content from LangChain message objects or dicts."""
        content = None
        if isinstance(message, dict):
            content = message.get('content', str(message))
        elif hasattr(message, 'content'):
            content = message.content
        else:
            return str(message)

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                elif isinstance(block, str):
                    text_parts.append(block)
                # Failsafe for other unexpected block types
                else: 
                    text_parts.append(str(block))
            return '\n'.join(text_parts)
        
        # Handle simple string content
        if isinstance(content, str):
            return content
        
        # Fallback for all other types
        return str(content)

    def create_step(self, step: str) -> str:
        """
        Use agent to help user create their version of this step.
        Supports clarification loops with memory.
        """
        print(f"\n{Colors.HEADER}{'='*60}")
        print(f"CREATE YOUR {step.upper()}")
        print(f"{'='*60}{Colors.ENDC}\n")
        
        consecutive_errors = 0  
        max_consecutive_errors = 3 

        while True:
            # Get user's goal
            prompt_text = f"Describe the {step} you want to create"
            if step == "surface":
                prompt_text += " (e.g., 'sphere at origin radius 5', 'plane at x=10')"
            elif step == "material":
                prompt_text += " (e.g., 'water', 'UO2 fuel')"
            
            goal = self.input_func(f"{Colors.BOLD}{prompt_text}...").strip()   

            if not goal:
                print(f"{Colors.GREEN}Finished defining {step}s.{Colors.ENDC}")
                break
            
            print(f"\nGenerating {step}...\n")
            
            # FIX 1: Initialize conversation history so Agent remembers proposals
            messages = [
                {"role": "user", "content": f"Create a {step} based on this description: {goal}"}
            ]
            
            max_attempts = 5 
            
            for attempt in range(max_attempts):
                try:
                    # Pass full message history
                    response = self.agent.invoke({"messages": messages})
                    
                    output = self._parse_agent_response(response)
                    
                    # FIX 2: Check for SUCCESS first to ignore keywords in the final summary
                    # If the agent says "I have created" or "Defined material", stop.
                    if "created" in output.lower() or "defined" in output.lower():
                        print(f"\n{Colors.GREEN}" + "="*60)
                        print("AGENT RESPONSE:")
                        print("="*60)
                        print(output)
                        print("="*60 + f"{Colors.ENDC}\n")
                        success = True
                        consecutive_errors = 0
                        break

                    # FIX 3: Refined clarification phrases (removed 'capture=' to be safer)
                    clarification_phrases = [
                        "should that be", "what", "which", "natural or enriched",
                        "light water or heavy water", "h2o or d2o", "enrichment",
                        "boundary condition", "reflective or vacuum",
                        "i suggest", "i recommend", "does this look correct", 
                        "would you like", "do you want", "can you confirm",
                        "how about", "already exists", "already defined", 
                        "different name", "unable to", "cannot create", 
                        "please specify", "please provide", "?"
                    ]
                    
                    if any(phrase in output.lower() for phrase in clarification_phrases):
                        print(f"\n{Colors.CYAN}AGENT MESSAGE:")
                        print("="*60)
                        print(output)
                        print("="*60 + f"{Colors.ENDC}\n")
                        
                        clarification = self.input_func(f"{Colors.BOLD}Your answer: {Colors.ENDC}").strip()
                        if not clarification:
                            print("No clarification provided. Skipping...")
                            break
                        
                        # FIX 4: Append to history instead of overwriting context
                        messages.append({"role": "assistant", "content": output})
                        messages.append({"role": "user", "content": clarification})
                        continue 
                    
                    # Fallback success print if no triggers matched
                    print(f"\n{Colors.GREEN}" + "="*60)
                    print("AGENT RESPONSE:")
                    print("="*60)
                    print(output)
                    print("="*60 + f"{Colors.ENDC}\n")

                    success = True
                    consecutive_errors = 0 
                    break
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"\n{Colors.FAIL}An unexpected error occurred.")
                    print(f"   Error details: {type(e).__name__}: {str(e)}{Colors.ENDC}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\n{Colors.RED}Multiple errors. Skipping step.{Colors.ENDC}\n")
                        break

            # Show script state
            print(f"{len(self.builder.defined[step])} {step}(s) defined so far.")
            
        print("\n" + "="*60)
        print("CURRENT SCRIPT STATE:")
        print("="*60)
        print(self.builder.get_script())
        print("="*60 + "\n")
        
        return self.builder.get_script()
    
    def run_onboarding(self):
        """
        Full 7-step interactive curriculum.
        """
        print("\n" + "="*60)
        print(f"{Colors.HEADER}Welcome to MCDC Onboarding!{Colors.ENDC}")
        print("="*60)
        print("\nI'll guide you through building a complete MCDC simulation.")
        print("We'll follow a 7-step workflow used by all MCDC scripts.\n")
        
        # Map of step names to user-friendly descriptions
        steps = [
            ("material", "Materials (what things are made of)"),
            ("surface", "Surfaces (geometric boundaries)"),
            ("cell", "Cells (regions of space)"),
            ("source", "Source (where particles start)"),
            ("tally", "Tally (what to measure)"),
            ("settings", "Settings (simulation parameters)"),
        ]
        
        for step, description in steps:
            print(f"\n{Colors.HEADER}{'#'*60}")
            print(f"# Step: {description}")
            print(f"{'#'*60}{Colors.ENDC}")
            
            # Teach concept and check if user wants to create
            if self.teach_concept(step):
                self.create_step(step)
            else:
                print(f"Skipping {step}.")
                continue
        
        # Final script
        final_script = self.builder.get_script()
        
        print("\n" + "="*60)
        print(f"{Colors.GREEN}ONBOARDING COMPLETE!{Colors.ENDC}")
        print("="*60)
        print("\nYour final MCDC script:\n")
        print(final_script)
        print("="*60)
        
        # Offer to save
        save = input(f"\n{Colors.BOLD}Save this script? [Y/n]: {Colors.ENDC}").strip().lower()
        if save != "n":
            filename = input(f"{Colors.BOLD}Filename (e.g., my_simulation.py): {Colors.ENDC}").strip()
            if not filename:
                filename = "mcdc_simulation.py"
            if not filename.endswith(".py"):
                filename += ".py"
            
            try:
                Path(filename).write_text(final_script)
                print(f"\n{Colors.GREEN}Saved to {filename}{Colors.ENDC}")
                print(f"\nTo run: python {filename}")
            except Exception as e:
                print(f"\n{Colors.FAIL}Error saving file: {e}{Colors.ENDC}")
        
        print(f"\n{Colors.CYAN}Thanks for using MCDC Tutor!{Colors.ENDC}\n")


if __name__ == "__main__":
    """
    Run the tutor from command line.
    
    Usage:
        export GEMINI_API_KEY="your-key-here"
        python -m llm_agent.onboarding.tutor
    """
    try:

        llm = load_llm(temperature=0.1)
        tutor = MCDCTutor(llm)
        tutor.run_onboarding()
        
    except KeyboardInterrupt:
        print("\n\nExiting. Your progress was not saved.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()