from onboarding.concepts import CONCEPT_LESSONS
from utils import load_llm, load_retriever, create_rag_chain
from onboarding.script_builder import ScriptBuilder
from onboarding.tools import get_mcdc_tools
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

class MCDCTutor:
    """
    Interactive tutor for learning MCDC through a 7-step workflow.
    
    Each step:
    1. Teaches the concept (from hardcoded curriculum)
    2. Shows a simple example (from RAG)
    3. Answers questions (using RAG chain)
    4. Helps user create their own version (using agent + tools)
    5. Validates and tracks state (using ScriptBuilder)
    """
    
    def __init__(self, llm):
        # Create our own builder instance (not a global)
        self.builder = ScriptBuilder()
        
        # Set up retriever and RAG chain for Q&A
        self.retriever = load_retriever(k=3)  # Increased from 2 to 3 for better coverage
        
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
        
        from utils import create_rag_chain_with_prompt
        self.rag_chain = create_rag_chain_with_prompt(llm, self.retriever, rag_prompt)
        
        # Pass builder to tool factory
        self.tools = get_mcdc_tools(self.builder)
        
        self.llm = llm
        
        # FIX: create_agent expects a PLAIN STRING prompt (not ChatPromptTemplate)
        # For create_agent, we only need system_prompt - no placeholders like {input} or {agent_scratchpad}
        # The agent framework handles message routing internally
        system_prompt = """You are MCDC-Tutor, an expert assistant for creating Monte Carlo particle transport simulations.

Your job: Help users create MCDC simulation scripts step-by-step using the available tools.

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:
1. NEVER ask the user for more information - make reasonable choices yourself
2. ALWAYS use tools to create entities - NEVER just describe what you would do
3. For names, choose sensible defaults like "material_1", "fuel", "absorber", "surface_1", etc.
4. Call get_current_script() first, then immediately call the creation tool

WORKFLOW FOR EVERY REQUEST:
Step 1: Call get_current_script() to check what exists
Step 2: Choose a good name based on the description (e.g., "pure absorber" → name it "absorber")
Step 3: Call the appropriate tool with ALL required parameters
Step 4: Explain what you created

TOOL PARAMETER FORMAT:
- All array parameters MUST be strings: capture="[1.0]" NOT capture=[1.0]
- 2D arrays: scatter="[[0.9]]" for 1-group, scatter="[[0.8, 0.1], [0.05, 0.85]]" for 2-group
- Surfaces: params="x=0.0" or params="center=[0.0, 0.0], radius=1.5"

EXAMPLE 1:
User: "Create a pure absorber material"
You: [Call get_current_script() → then call set_material_mg(name="absorber", capture="[1.0]", scatter="[[0.0]]")]
Response: "✅ Created 'absorber' - 100% absorption, no scattering"

EXAMPLE 2:
User: "Create a fissile material"
You: [Call get_current_script() → then call set_material_mg(name="fuel", capture="[0.45]", scatter="[[0.0]]", fission="[0.55]", nu_p="[2.5]")]
Response: "✅ Created 'fuel' - fissile material with 55% fission probability"

DO NOT ask follow-up questions. DO NOT say "I can help with that" - just DO IT."""
        
        # Create agent using create_agent with string prompt
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt  # Plain string, not ChatPromptTemplate
        )
    
    def teach_concept(self, step: str) -> bool:
        """
        Run the 5-part mini-lesson for a given step.
        
        Returns:
            bool: True if user is ready to create, False if they want to skip
        """
        lesson = CONCEPT_LESSONS[step]
        
        print(f"\n{'='*60}")
        print(f"📖 STEP: {step.upper()}")
        print(f"{'='*60}\n")
        print(f"Concept:\n{lesson['concept']}\n")
        print(f"Key parts:\n{lesson['parts']}\n")
        
        # Show simplest example from RAG
        # FIX: Add step-specific query expansion for better retrieval
        try:
            # Query expansion: add step-specific keywords
            query = f"beginner {step} example simple"
            examples = self.retriever.invoke(query)
            if examples:
                print(f"\n📄 Example from regression tests:")
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
        
        while True:
            q = input(f"\n❓ Ask a question about {step} (or press Enter to continue): ").strip()
            if not q:
                break
            
            try:
                # Use RAG chain to answer
                answer = self.rag_chain.invoke(q)
                print(f"\n🤖 {answer}\n")
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        # Check if ready to create
        ready = input(f"\n✅ Ready to create your {step}? [Y/n]: ").strip().lower()
        return ready != "n"
    
    def create_step(self, step: str) -> str:
        """
        Use agent to help user create their version of this step.
        
        Returns:
            str: The generated code snippet
        """
        print(f"\n{'='*60}")
        print(f"🎯 CREATE YOUR {step.upper()}")
        print(f"{'='*60}\n")
        
        # Get user's goal in natural language
        goal = input(f"Describe the {step} you want to create: ").strip()
        
        if not goal:
            print("Skipping (no description provided)")
            return ""
        
        print(f"\n🔧 Generating {step}...\n")
        
        # FIX: create_agent expects {"messages": [...]} format, not {"input": "..."}
        # Messages should be in LangChain format with role and content
        try:
            response = self.agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": f"Create a {step} based on this description: {goal}"
                }]
            })
            
            # FIX: Response from create_agent contains a "messages" list
            # The last message is the agent's final response
            messages = response.get("messages", [])
            if messages:
                last_message = messages[-1]
                # Handle different message formats
                if hasattr(last_message, 'content'):
                    # AIMessage object
                    output = last_message.content
                elif isinstance(last_message, dict):
                    # Dict format
                    output = last_message.get('content', str(last_message))
                elif isinstance(last_message, list):
                    # List of content blocks (Gemini format)
                    text_parts = [block.get('text', '') for block in last_message if isinstance(block, dict) and block.get('type') == 'text']
                    output = '\n'.join(text_parts) if text_parts else str(last_message)
                else:
                    output = str(last_message)
            else:
                output = "No response from agent"
            
            print(f"\n{'='*60}")
            print("🤖 AGENT RESPONSE:")
            print("="*60)
            print(output)
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error during agent execution: {e}")
            print("You can try again or skip this step.\n")
            import traceback
            traceback.print_exc()
            return ""
        
        # Show the current script state
        print("\n" + "="*60)
        print("📝 CURRENT SCRIPT:")
        print("="*60)
        print(self.builder.get_script())
        print("="*60 + "\n")
        
        return self.builder.get_script()
    
    def run_onboarding(self):
        """
        Full 7-step interactive curriculum.
        
        Workflow:
        1. Materials - Define what things are made of
        2. Surfaces - Define geometric boundaries
        3. Cells - Combine surfaces into regions filled with materials
        4. Source - Define where particles start
        5. Tally - Define what to measure
        6. Settings - Configure simulation parameters
        7. Run - Execute the simulation
        """
        print("\n" + "="*60)
        print("🎓 Welcome to MCDC Onboarding!")
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
        print("🎉 ONBOARDING COMPLETE!")
        print("="*60)
        print("\nYour final MCDC script:\n")
        print(final_script)
        print("="*60)
        
        # Offer to save
        save = input("\n💾 Save this script? [Y/n]: ").strip().lower()
        if save != "n":
            filename = input("Filename (e.g., my_simulation.py): ").strip()
            if not filename:
                filename = "mcdc_simulation.py"
            if not filename.endswith(".py"):
                filename += ".py"
            
            try:
                Path(filename).write_text(final_script)
                print(f"\n✅ Saved to {filename}")
                print(f"\nTo run: python {filename}")
            except Exception as e:
                print(f"\n❌ Error saving file: {e}")
        
        print("\n👋 Thanks for using MCDC Tutor!\n")


if __name__ == "__main__":
    """
    Run the tutor from command line.
    
    Usage:
        export GEMINI_API_KEY="your-key-here"
        python onboarding/tutor.py
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