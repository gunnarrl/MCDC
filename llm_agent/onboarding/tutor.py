from llm_agent.onboarding.concepts import CONCEPT_LESSONS
from llm_agent.utils import load_llm, load_retriever, create_rag_chain_with_prompt
from llm_agent.onboarding.script_builder import ScriptBuilder
from llm_agent.onboarding.tools import get_mcdc_tools
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path

# UI Imports
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel
from rich.theme import Theme
from rich.status import Status
from rich.prompt import Confirm

# Define a custom theme for Rich
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "tutor": "bold blue",
    "user": "bold magenta",
    "step": "magenta bold reverse",
    "code": "bold white",
})

console = Console(theme=custom_theme)

class MCDCTutor:
    """
    Interactive tutor for learning MCDC through a 7-step workflow.
    """
    
    STEP_KEYWORDS = {
        "material": "MaterialMG Material capture scatter fission nuclide_composition density multi-group continuous-energy cross-section macroscopic",
        "surface": "Surface PlaneX PlaneY PlaneZ CylinderX CylinderY CylinderZ Sphere boundary_condition vacuum reflective interface geometry normal",
        "cell": "Cell region fill boolean operators intersection union & |",
        "hierarchy": "Universe Lattice fill translation rotation repetition grid 3D array hexagonal square", 
        "source": "Source position energy direction isotropic white_direction time energy_group spectrum point_source volume_source uniform",
        "tally": "TallyMesh TallyCell TallySurface scores flux collision fission net-current mesh energy_bins detector mu_bins",
        "settings": "settings N_particle N_batch eigenmode census output population_control variance_reduction convergence active inactive",
        "run": "run execute simulate output h5 tally results k-effective convergence statistics batch cycle"
    }
    
    def __init__(self, llm):
        self.builder = ScriptBuilder()
        
        # specific style for the input prompt cursor
        self.prompt_style = PromptStyle.from_dict({
            'prompt': '#00aa00 bold',  # Green bold prompt
        })
        self.session = PromptSession()

        self.doc_filter = {
            "type": {
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
### PROTOCOL FOR MODIFICATIONS
If the user asks to "delete", "change", "update", or "fix" an existing entity:
1.  Call `manage_script(action='delete', ...)` to remove the old version.
2.  If changing an entity, confirm the new plan with the user.
3.  Call the appropriate creation tool to define the new entity.

### PROTOCOL FOR COMPLEX TASKS
If the user requests a task that requires defining multiple entities (e.g., "finite cylinder" or "box"), or is otherwise complex:
1.  **PROPOSE**: Explicitly list the plan.
2.  **CONFIRM**: Ask "Does this plan look correct?"
3.  **EXECUTE**: Only after confirmation. Use `description` on the FIRST entity to label the group.

### TOOL USAGE GUIDELINES
**1. Creating Materials (`create_material`)**
* **Mode 'MG' (Multi-Group):** [DEFAULT] Pass arrays as lists: `{"capture": [0.1]}`.
* **Mode 'CE' (Continuous Energy):** [Use only if user explicitly asks] Use for specific isotopes.
    * Pass dictionary in `properties`: `{"nuclide_composition": {"U235": 0.7}}`.
* **Mode 'formula':** [Use if user asks for CE and provides a compound material like water, stainless steel, etc.] Use for chemical formulas (e.g., "H2O", "UO2").
    * Pass details in `properties`: `{"formula": "H2O", "density": 1.0}`.

**2. Creating Geometry (`create_surface`, `create_geometry`)**
* Use `create_surface` for Planes, Cylinders, Spheres.
* Use `create_geometry` for **Cells**, **Universes**, **Lattices**, or **Meshes**.
    * Cell Example: `type_='cell', params='{"region": "+s1 & -s2", "fill": "fuel"}'`.
* **np.linspace rule**: Use **N+1 points** for **N intervals**.

**3. Creating Tallies (`create_tally`)**
* Use `type_='mesh'` for TallyMesh. DO NOT use np.linspace for uniform mesh, just tuple (start, end, N).
* Use `type_='surface'` for TallySurface.

**4. Settings (`set_settings`)**
* Pass all settings in a single JSON object.

## WORKFLOW
1.  **Check Status:** Call `manage_script(action='get')`.
2.  **Search Docs:** If "how to", search first.
3.  **Clarify/Propose:** If ambiguous, ask.
4.  **Tool Call:** Call the appropriate tool.
5.  **Explain:** Briefly explain what you created.
"""
        
            self.agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=system_prompt 
            )
        except Exception as e:
            console.print(f"[error]Warning: Failed to initialize agent properly: {e}[/error]")
            raise
    
    def get_input(self, prompt_text=""):
        """
        Unified input handler. Prints the prompt text using Rich, 
        then gets input using prompt_toolkit.
        """
        if prompt_text:
            console.print(f"[user]{prompt_text}[/user]")
        
        # The tuple syntax is (style_class, text)
        return self.session.prompt([('class:prompt', '> ')], style=self.prompt_style).strip()

    def expand_query(self, query: str, step: str) -> str:
        base_query = f"{step} {query}"
        keywords = self.STEP_KEYWORDS.get(step, "")
        return f"{base_query} {keywords}"
    
    def get_step_retriever(self, step: str):
        try:
            vectorstore = self.retriever.vectorstore
            step_filter = {"$and": [self.doc_filter, {"section": step}]}
            return vectorstore.as_retriever(search_kwargs={"k": 5, "filter": step_filter})
        except AttributeError:
            return self.retriever
    
    def teach_concept(self, step: str) -> bool:
        lesson = CONCEPT_LESSONS[step]
        
        console.print("\n")
        console.rule(f"[step] STEP: {step.upper()} [/step]")
        
        # 1. Concept Block
        console.print(Panel(
            Markdown(lesson['concept']),
            title="Concept",
            border_style="magenta"
        ))
        
        # 2. Key Parts
        console.print("\n[bold]Key parts:[/bold]")
        console.print(Markdown(lesson['parts']))
        
        # Show step-specific examples
        try:
            step_retriever = self.get_step_retriever(step)
            expanded_query = self.expand_query("beginner simple example", step)
            examples = step_retriever.invoke(expanded_query)
            
            if examples:
                console.print("\n[dim]Reference Example:[/dim]")
                snip = Syntax(examples[0].page_content[:500], "python", theme="ansi_dark")
                console.print(snip)
        except Exception as e:
            console.print(f"\n[dim](Could not load example: {e})[/dim]")
        
        console.print(f"\n[warning]Common questions about {step}:[/warning]")
        for i, q in enumerate(lesson.get('key_questions', [])[:3], 1):
            console.print(f"  {i}. {q}")
        
        step_rag_chain = create_rag_chain_with_prompt(
            self.llm, self.get_step_retriever(step), self.rag_prompt
        )
        
        while True:
            q = self.get_input("Ask a question (press enter to continue):")
            if not q:
                break
            
            try:
                expanded_q = self.expand_query(q, step)
                console.print(f"\n[tutor]TUTOR:[/tutor] ", end="")
                
                # Use status spinner for RAG retrieval
                with console.status("[bold blue]Thinking...", spinner="dots"):
                    response = ""
                    for chunk in step_rag_chain.stream(expanded_q):
                        response += str(chunk)
                
                console.print(Markdown(response))
                console.print("")

            except Exception as e:
                console.print(f"\n[error]Error: {e}[/error]")
        
        console.print("")
        return Confirm.ask(f"[bold]Ready to create your {step}?[/bold]", default=True)
    
    def _parse_agent_response(self, response) -> str:
        if isinstance(response, str): return response
        if isinstance(response, dict):
            if 'output' in response: return response['output']
            if 'content' in response: return response['content']
            if 'messages' in response and response['messages']:
                return self._extract_message_content(response['messages'][-1])
        if hasattr(response, 'content'): return response.content
        if isinstance(response, list):
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
        content = None
        if isinstance(message, dict): content = message.get('content', str(message))
        elif hasattr(message, 'content'): content = message.content
        else: return str(message)

        if isinstance(content, list):
            text_parts = [b.get('text', '') if isinstance(b, dict) else str(b) for b in content]
            return '\n'.join(text_parts)
        return str(content)

    def _print_script(self):
        console.print("\n")
        console.rule("[bold cyan]CURRENT SCRIPT[/bold cyan]")
        script_content = self.builder.get_script()
        syntax = Syntax(script_content, "python", theme="monokai", line_numbers=True, word_wrap=True)
        console.print(syntax)
        console.rule("[bold cyan]END SCRIPT[/bold cyan]")
        console.print("\n")

    def _handle_view_mode(self):
        while True:
            self._print_script()
            console.print("[bold]View Mode:[/bold] Press [green]Enter[/green] to return to flow, or type an instruction to edit.")
            command = self.get_input()

            if not command:
                console.print("Returning to tutorial...")
                break
                
            messages = [{"role": "user", "content": command}]
            
            while True:
                try:
                    with console.status("[bold yellow]Processing edit...", spinner="dots"):
                        response = self.agent.invoke({"messages": messages})
                    
                    output = self._parse_agent_response(response)
                    console.print(Panel(Markdown(output), title="Agent", border_style="green"))
                    
                    clarification_phrases = [
                        "should that be", "what", "which", "correct", "confirm", "?",
                        "plan", "propose", "intend to", "clarify", "suggest"
                    ]
                    
                    if any(phrase in output.lower() for phrase in clarification_phrases):
                        user_reply = self.get_input("Response (or Enter to cancel):")
                        if not user_reply:
                            console.print("Edit cancelled.")
                            break
                        messages.append({"role": "assistant", "content": output})
                        messages.append({"role": "user", "content": user_reply})
                        continue 
                    break
                    
                except Exception as e:
                    console.print(f"[error]Error: {e}[/error]")
                    break

    def create_step(self, step: str) -> str:
        console.print("\n")
        console.rule(f"[bold]CREATE YOUR {step.upper()}[/bold]")
        
        consecutive_errors = 0  
        max_consecutive_errors = 3 

        while True:
            prompt_text = f"Describe {step}(s) to add, or type 'view' to see script/edit"
            if step == "surface": prompt_text += " (e.g., 'sphere radius 5')"
            elif step == "material": prompt_text += " (e.g., 'water')"
            
            goal = self.get_input(f"{prompt_text} (or Enter to finish):")

            if not goal:
                console.print(f"[success]Finished defining {step}s.[/success]")
                break
            
            if goal.lower() == "view":
                self._handle_view_mode()
                continue

            if goal.lower() == "undo":
                result = self.builder.undo_last()
                console.print(f"[warning]{result}[/warning]")
                continue

            start_count = len(self.builder.defined.get(step, set()))
            current_context = self.builder.get_script()
            
            messages = [{
                "role": "user", 
                "content": (
                    f"Here is the current MCDC script:\n```python\n{current_context}\n```\n\n"
                    f"TASK: Create a {step}(s) based on this description: {goal}\n"
                    f"IMPORTANT: Use existing variable names from the script where appropriate."
                )
            }]
            
            for attempt in range(5):
                try:
                    with console.status(f"[bold yellow]Drafting {step}...[/bold yellow]", spinner="dots"):
                        response = self.agent.invoke({"messages": messages})
                        output = self._parse_agent_response(response)

                    current_count = len(self.builder.defined.get(step, set()))
                    
                    if current_count > start_count:
                        console.print(Panel(output, title="[bold green]SUCCESS[/bold green]", border_style="green"))
                        new_code = self.builder.get_code_by_type(step)
                        console.print(f"\n[bold]Current {step.upper()} definitions:[/bold]")
                        console.print(Syntax(new_code, "python", theme="monokai"))
                        consecutive_errors = 0
                        break

                    # Clarification Logic
                    console.print(Panel(Markdown(output), title="[bold blue]TUTOR[/bold blue]", border_style="blue"))
                    
                    clarification_phrases = [
                        "should that be", "what", "which", "correct", "confirm", "?",
                        "plan", "propose", "intend to", "clarify", "suggest"
                    ]

                    if any(phrase in output.lower() for phrase in clarification_phrases):
                        clarification = self.get_input("Your answer:")
                        if not clarification: break
                        messages.append({"role": "assistant", "content": output})
                        messages.append({"role": "user", "content": clarification})
                        continue 
                    else:
                        break
                    
                except Exception as e:
                    consecutive_errors += 1
                    console.print(f"\n[error]An unexpected error occurred: {e}[/error]")
                    if consecutive_errors >= max_consecutive_errors: break

        self._print_script()
        return self.builder.get_script()
    
    def run_onboarding(self):
        console.print("\n")
        console.rule("[bold cyan]Welcome to MCDC Onboarding![/bold cyan]")
        console.print("\nI'll guide you through building a complete MCDC simulation.")
        console.print("We'll follow a 7-step workflow used by all MCDC scripts.\n")
        
        steps = [
            ("material", "Materials (what things are made of)"),
            ("surface", "Surfaces (geometric boundaries)"),
            ("cell", "Cells (regions of space)"),
            ("hierarchy", "Hierarchies (Universes & Lattices) [optional]"),
            ("source", "Source (where particles start)"),
            ("tally", "Tally (what to measure)"),
            ("settings", "Settings (simulation parameters)"),
        ]
        
        for step, description in steps:
            if self.teach_concept(step):
                self.create_step(step)
            else:
                console.print(f"[dim]Skipping {step}.[/dim]")
                continue
        
        final_script = self.builder.get_script()
        
        console.print("\n")
        console.rule("[bold green]ONBOARDING COMPLETE![/bold green]")
        console.print(Syntax(final_script, "python", theme="monokai"))
        
        if Confirm.ask("\n[bold]Save this script?[/bold]", default=True):
            filename = self.get_input("Filename (e.g., my_simulation.py):")
            if not filename: filename = "mcdc_simulation.py"
            if not filename.endswith(".py"): filename += ".py"
            
            try:
                Path(filename).write_text(final_script)
                console.print(f"\n[success]Saved to {filename}[/success]")
            except Exception as e:
                console.print(f"\n[error]Error saving file: {e}[/error]")
        
        console.print(f"\n[bold cyan]Thanks for using MCDC Tutor![/bold cyan]\n")

if __name__ == "__main__":
    try:
        llm = load_llm(temperature=0.1)
        tutor = MCDCTutor(llm)
        tutor.run_onboarding()
        
    except KeyboardInterrupt:
        console.print("\n\n[warning]Exiting. Your progress was not saved.[/warning]")
    except Exception as e:
        console.print(f"\n[error]Fatal error: {e}[/error]")
        import traceback
        traceback.print_exc()