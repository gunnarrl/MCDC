from onboarding.concepts import CONCEPT_LESSONS
from utils import load_llm, load_retriever, create_rag_chain
from onboarding.script_builder import ScriptBuilder
from onboarding.tools import get_mcdc_tools
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
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
        # FIX: Create our own builder instance (not a global)
        self.builder = ScriptBuilder()
        
        # Set up retriever and RAG chain for Q&A
        self.retriever = load_retriever(k=2)
        self.rag_chain = create_rag_chain(self.retriever)
        
        # FIX: Pass builder to tool factory
        self.tools = get_mcdc_tools(self.builder)
        
        self.llm = llm
        
        # FIX: Create agent using create_agent (LangChain 1.0+)
        # create_agent returns a runnable agent that can be invoked directly
        
        # Create prompt template for the agent
        # Must use ChatPromptTemplate for create_agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are MCDC-Tutor, helping users build Monte Carlo particle transport simulations.

You have access to tools that let you:
- Define materials (multi-group or continuous-energy)
- Create surfaces (planes, spheres, cylinders)
- Create cells (regions filled with materials)
- Set up sources (particle emission)
- Configure tallies (detectors)
- Set simulation settings

CRITICAL RULES:
1. ALWAYS call get_current_script FIRST to see what's already defined
2. Define entities in order: materials → surfaces → cells → source → tally → settings
3. If a tool returns ERROR, explain why and ask for clarification
4. When creating materials, ask if they want MG (multi-group) or CE (continuous-energy)
5. For MG materials, cross-sections are numpy arrays as strings: capture="[0.5]", scatter="[[0.9]]"
6. Surface names should be descriptive: s1, s2, sphere, cylinder, etc.
7. Region syntax: +surface means "positive side", -surface means "negative side", & is AND, | is OR
8. Always provide the exact tool parameters needed

Work step by step:
- First check what's defined
- Then create the requested entity
- Confirm success or explain errors"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # FIX: create_agent returns a runnable that can be invoked
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=prompt
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
        # FIX: Use .invoke() only (standardized on LangChain 1.0 API)
        try:
            examples = self.retriever.invoke(f"beginner {step} example")
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
        
        # FIX: create_agent returns a runnable that expects specific input format
        # The agent needs: input, chat_history (optional), agent_scratchpad (handled internally)
        try:
            response = self.agent.invoke({
                "input": f"Create a {step} based on this description: {goal}",
                "chat_history": [],  # Empty for now, can add conversation history later
            })
            
            # FIX: Response from create_agent is a dict with 'output' key
            # Format: {"input": "...", "output": "...", "intermediate_steps": [...]}
            output = response.get("output", "")
            
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