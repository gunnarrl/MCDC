from llm_agent.onboarding.concepts import CONCEPT_LESSONS
from llm_agent.utils import load_llm, load_retriever, create_rag_chain_with_prompt
from llm_agent.onboarding.script_builder import ScriptBuilder
from llm_agent.onboarding.tools import get_mcdc_tools
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

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
    
    def __init__(self, llm):
        # Create our own builder instance (not a global)
        self.builder = ScriptBuilder()
        
        # Set up retriever and RAG chain for Q&A
        self.retriever = load_retriever(k=5)
        
        # FIX: Improved RAG prompt for educational, beginner-friendly responses
        rag_prompt = """You are MCDC-Tutor, explaining Monte Carlo particle transport concepts to beginners.

Question: {question}

Documentation: {context}

Provide a helpful answer that includes:
1. A simple code example (if relevant)
2. Clear explanation of key concepts (assume NO nuclear physics background)
3. Brief description of required parameters

Use friendly, educational tone. Keep it concise but informative.

Answer:"""
        self.rag_prompt = rag_prompt  # Store for reuse in step-specific chains
        
        self.rag_chain = create_rag_chain_with_prompt(llm, self.retriever, rag_prompt)
        
        try:
            self.tools = get_mcdc_tools(self.builder, self.retriever)

            self.llm = llm
            
            system_prompt = """You are MCDC-Tutor, an expert assistant for creating Monte Carlo particle transport simulations using the MCDC Python package.

---
## 💡 CRITICAL MATERIAL MODE INSTRUCTIONS (CE is now the default)

- **Continuous-Energy (CE)**: **Default mode.** Use this most of the time. It requires calculating the atomic composition and automatically adds a note about the required `MCDC_XSLIB` environment variable.
- **Multi-Group (MG)**: Use this mode **only for teaching basic concepts** or when the user asks or provides cross-section data (e.g., capture="[1.0]").

---
## ❓ WHEN TO ASK CLARIFYING QUESTIONS

Ask **ONE** clarifying question when a parameter is critical and unknown:
 DO ASK: "Should that be light water (H₂O) or heavy water (D₂O)?" (affects neutron physics significantly)
 DO ASK: "What enrichment for uranium fuel? PWR uses 3-5%, research reactors up to 20%." (critical safety parameter)
 DO ASK: "Natural uranium (0.72% U-235) or enriched?" (determines if material is fissile)
 DON'T ASK: "What density should I use?" (use standard values from MaterialCalculator)
 DON'T ASK: "What cross-section values?" (CE mode handles this; MG uses sensible defaults if needed)

---
## ⚙️ WORKFLOW FOR EVERY REQUEST

1.  **Check Status:** Call `get_current_script()` to check what entities already exist.
2.  **Search Docs (If Needed):** If the user asks "how to do X", "what are the parameters for Y", or "what's an example of Z", use `search_docs(query)` **before** trying to call other tools.
3.  **Clarify:** If the request is ambiguous (e.g., "uranium fuel"), ask **ONE** clarifying question, if needed.
4.  **Convert & Set Defaults:**
    * If the user names a **common material** (e.g., "water", "stainless steel"), **convert the name to its formula** (e.g., "H2O", "Fe0.7Cr0.2Ni0.1").
    * Look up and use the **default density** from `MaterialCalculator.COMMON_MATERIALS` for the formula.
5.  **Tool Call:** Call the appropriate tool with **ALL** required parameters, defaulting to `mode="CE"` unless MG is explicitly requested or required for teaching.
6.  **Explain:** Explain what you created and why.

---
## 📝 TOOL PARAMETER FORMAT

-   All array parameters **MUST be strings**: e.g., `capture="[1.0]"` NOT `capture=[1.0]`.
-   2D arrays: e.g., `scatter="[[0.8, 0.1], [0.05, 0.85]]"`.
-   Surfaces params: e.g., `params="x=0.0"` or `params="center=[0.0, 0.0], radius=1.5"`.

---
## EXAMPLE FLOW (CE Default)

**Scenario:** User wants 3% enriched fuel.

1.  User: "create uranium fuel"
2.  You: "Should that be natural uranium or enriched? PWR fuel is typically **3-5% U-235**." (Step 2)
3.  User: "3% enriched"
4.  You: [**Internal Conversion:** Name="fuel" → Formula="UO2", Density=10.5] (Step 3)
5.  You: [Call `create_material_from_formula`(**`"fuel"`**, **`"UO2"`**, **`10.5`**, **`mode="CE"`**, **`enrichment=0.03`**)] (Step 4)
6.  Response: "✓ Created **CE material** 'fuel': UO2 at 10.5 g/cm³ (enriched to 3.0% U-235). **NOTE:** CE mode requires MCDC\_XSLIB environment variable."""
        
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt 
            )
        except Exception as e:
            print(f"Warning: Failed to initialize agent properly: {e}")
            print("Tutor may not function correctly. Check API keys and dependencies.")
            raise
    
    # Query expansion for better retrieval
    # Combines user query with relevant technical terms to improve document recall
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
    
    # Create step-filtered retriever
    # Ensures we only retrieve examples/documentation relevant to current workflow step
    def get_step_retriever(self, step: str):
        """
        Create a retriever that filters by workflow step.
        This is crucial for showing users only relevant examples during each step.
        """
        try:
            # Access the underlying vectorstore from our existing retriever
            vectorstore = self.retriever.vectorstore
            
            # Create a new retriever with step-specific metadata filter
            return vectorstore.as_retriever(
                search_kwargs={
                    "k": 5, 
                    "filter": {"section": step} 
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
        
        print(f"\n{'='*60}")
        print(f"STEP: {step.upper()}")
        print(f"{'='*60}\n")
        print(f"Concept:\n{lesson['concept']}\n")
        print(f"Key parts:\n{lesson['parts']}\n")
        
        # Show step-specific examples from RAG
        # Uses query expansion + metadata filtering for maximum relevance
        try:

            step_retriever = self.get_step_retriever(step)
            expanded_query = self.expand_query("beginner simple example", step)
            examples = step_retriever.invoke(expanded_query)
            
            if examples:
                print(f"\nExample from regression tests:")
                print("-" * 60)
                # Show first 500 chars of first example
                print(examples[0].page_content[:500])
                if len(examples[0].page_content) > 500:
                    print("...")
                print("-" * 60)
        except Exception as e:
            print(f"\n(Could not load example: {e})")
        
        # Q&A loop - user can ask questions about the concept
        print(f"\n💡 Common questions about {step}:")
        for i, q in enumerate(lesson.get('key_questions', [])[:3], 1):
            print(f"  {i}. {q}")
        
        # Create step-specific RAG chain for Q&A
        step_rag_chain = create_rag_chain_with_prompt(
            self.llm,
            self.get_step_retriever(step),
            self.rag_prompt
        )
        
        while True:
            q = input(f"\nAsk a question about {step} (or press Enter to continue): ").strip()
            if not q:
                break
            
            try:
                # Expand user query
                expanded_q = self.expand_query(q, step)
                
                answer = step_rag_chain.invoke(expanded_q)
                print(f"\n🤖 {answer}\n")
            except Exception as e:
                print(f"\nError: {e}\n")
        
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
        Supports clarification loops for ambiguous requests.
        Allows creating multiple entities in a loop.
        """
        print(f"\n{'='*60}")
        print(f"CREATE YOUR {step.upper()}")
        print(f"{'='*60}\n")
        
         consecutive_errors = 0  
        max_consecutive_errors = 3 

        while True:
            # Get user's goal in natural language
            prompt_text = f"Describe the {step} you want to create"
            if step == "surface":
                prompt_text += " (e.g., 'sphere at origin radius 5', 'plane at x=10')"
            elif step == "material":
                prompt_text += " (e.g., 'water', 'UO2 fuel')"
            elif step == "cell":
                prompt_text += " (e.g., 'cell filled with fuel, bounded by s1 and s2')"
            elif step == "source":
                prompt_text += " (e.g., 'point source at origin with 1 MeV energy')"
            elif step == "tally":
                prompt_text += " (e.g., 'mesh tally from x=0 to 10 with 100 bins')"
            elif step == "settings":
                prompt_text += " (e.g., '1000 particles, 10 batches')"
                
                
            goal = input(f"{prompt_text} (or press Enter to finish this step): ").strip()
            
            if not goal:
                print(f"Finished defining {step}s.")
                break
            
            print(f"\nGenerating {step}...\n")
            
            # Allow for clarification loop (max 3 attempts)
            context = f"Create a {step} based on this description: {goal}"
            max_attempts = 3
            
            for attempt in range(max_attempts):
                try:
                    # Invoke agent
                    response = self.agent.invoke({
                        "messages": [{
                            "role": "user",
                            "content": context
                        }]
                    })
                    
                    # Parse response using robust handler
                    output = self._parse_agent_response(response)
                    
                    # Check if agent is asking a clarifying question
                    clarification_phrases = [
                        "should that be", "what", "which", "natural or enriched",
                        "light water or heavy water", "h2o or d2o", "enrichment",
                        "boundary condition", "reflective or vacuum"
                    ]
                    
                    if any(phrase in output.lower() for phrase in clarification_phrases):
                        print(f"\n🤖 AGENT QUESTION:")
                        print("="*60)
                        print(output)
                        print("="*60 + "\n")
                        
                        # Get user clarification
                        clarification = input("Your answer: ").strip()
                        if not clarification:
                            print("No clarification provided. Skipping...")
                            break
                        
                        # Update context with clarification
                        context = f"Based on user's clarification '{clarification}', create the {step}: {goal}"
                        continue  # Loop back to agent with clarification
                    
                    # show final response
                    print(f"\n" + "="*60)
                    print("🤖 AGENT RESPONSE:")
                    print("="*60)
                    print(output)
                    print("="*60 + "\n")

                    success = True
                    consecutive_errors = 0 
                    break
                    
                except Exception as e:
                    consecutive_errors += 1
                    print(f"\n❌ An unexpected error occurred while processing your request.")
                    print(f"   Error details: {type(e).__name__}: {str(e)}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"\nMultiple consecutive errors detected. Skipping this step to avoid repeated failures.")
                        print("   Please check your API key and network connection, then try again.")
                        print(f"   Let's try again. Please rephrase your request if possible.\n")
                        break

            if not success:
                continue
            
            # Show script state after each addition
            print(f"{len(self.builder.defined[step])} {step}(s) defined so far.")
            
        # Show the full current script state before exiting the step
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
        print("Welcome to MCDC Onboarding!")
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
            print(f"\n{'#'*60}")
            print(f"# Step: {description}")
            print(f"{'#'*60}")
            
            # Teach concept and check if user wants to create
            if self.teach_concept(step):
                self.create_step(step)
            else:
                print(f"⏭️  Skipping {step}.")
                continue
        
        # Final script
        final_script = self.builder.get_script()
        
        print("\n" + "="*60)
        print("ONBOARDING COMPLETE!")
        print("="*60)
        print("\nYour final MCDC script:\n")
        print(final_script)
        print("="*60)
        
        # Offer to save
        save = input("\nSave this script? [Y/n]: ").strip().lower()
        if save != "n":
            filename = input("Filename (e.g., my_simulation.py): ").strip()
            if not filename:
                filename = "mcdc_simulation.py"
            if not filename.endswith(".py"):
                filename += ".py"
            
            try:
                Path(filename).write_text(final_script)
                print(f"\nSaved to {filename}")
                print(f"\nTo run: python {filename}")
            except Exception as e:
                print(f"\nError saving file: {e}")
        
        print("\n👋 Thanks for using MCDC Tutor!\n")


if __name__ == "__main__":
    """
    Run the tutor from command line.
    
    Usage:
        export GEMINI_API_KEY="your-key-here"
        python -m llm_agent.onboarding.tutor
    """
    try:
        # Load LLM with low temperature for deterministic code generation
        llm = load_llm(temperature=0.1)
        
        # Create tutor instance
        tutor = MCDCTutor(llm)
        
        # Run the interactive onboarding
        tutor.run_onboarding()
        
    except KeyboardInterrupt:
        print("\n\n👋 Exiting. Your progress was not saved.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()