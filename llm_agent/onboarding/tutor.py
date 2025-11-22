from llm_agent.onboarding.concepts import CONCEPT_LESSONS
from llm_agent.utils import load_llm, load_retriever, create_rag_chain_with_prompt
from llm_agent.onboarding.script_builder import ScriptBuilder
from llm_agent.onboarding.tools import get_mcdc_tools
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pathlib import Path
import subprocess

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
                
            current_context = self.builder.get_script()
            
            messages = [{
                "role": "user", 
                "content": (
                    f"Here is the current MCDC script:\n```python\n{current_context}\n```\n\n"
                    f"EDIT/ADD TASK: {command}\n"
                    f"IMPORTANT: You are editing/adding to an existing script. Use the variable names shown above."
                )
            }]
            
            while True:
                try:
                    with console.status("[bold yellow]Processing edit...", spinner="dots"):
                        response = self.agent.invoke({"messages": messages})
                    
                    output = self._parse_agent_response(response)
                    console.print(Panel(Markdown(output), title="Agent", border_style="green"))
                    
                    clarification_phrases = [
                        "should that be", "what", "which",
                        "i suggest", "i recommend", "does this look correct", 
                        "would you like", "do you want", "can you confirm",
                        "how about", "already exists", "already defined", 
                        "different name", "unable to", "cannot create", 
                        "please specify", "please provide", "?",
                        "plan", "propose", "intend to", "clarify", "confirm", "suggest", "recommend",
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

    def run_visualization(self, debug_mode: bool = True):
        """
        Generates a temporary script to visualize geometry in 3D.
        Decodes MCDC Quadric Coefficients (A..J) to reconstruct shapes.
        """
        console.print(f"\n[bold yellow]Generating 3D Geometry Preview...[/bold yellow]")

        base_script = self.builder.get_script(include_run=False)
        
        plot_code = f"""
# --- VISUALIZATION APPENDED BY TUTOR ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

# SETTINGS
BOUNDS = 15.0 
RESOLUTION = 20

def decode_and_plot(ax, obj, name, U, V):
    # Extract Quadric Coefficients (Default to 0.0 if missing)
    A = getattr(obj, 'A', 0.0)
    B = getattr(obj, 'B', 0.0)
    C = getattr(obj, 'C', 0.0)
    G = getattr(obj, 'G', 0.0)
    H = getattr(obj, 'H', 0.0)
    I = getattr(obj, 'I', 0.0)
    J = getattr(obj, 'J', 0.0)

    # --- CASE 1: PLANES (Linear terms only) ---
    # Check if quadratic terms (A,B,C) are basically zero
    if abs(A) < 1e-9 and abs(B) < 1e-9 and abs(C) < 1e-9:
        
        # Plane Z: Iz + J = 0  ->  z = -J/I
        if abs(I) > 1e-9:
            z_val = -J / I
            print(f"  -> Plotting {{name}} as PlaneZ (z={{z_val:.2f}})")
            ax.plot_surface(U, V, np.full_like(U, z_val), alpha=0.2, color='blue')
            return True
            
        # Plane X: Gx + J = 0  ->  x = -J/G
        elif abs(G) > 1e-9:
            x_val = -J / G
            print(f"  -> Plotting {{name}} as PlaneX (x={{x_val:.2f}})")
            ax.plot_surface(np.full_like(U, x_val), U, V, alpha=0.2, color='red')
            return True

        # Plane Y: Hy + J = 0  ->  y = -J/H
        elif abs(H) > 1e-9:
            y_val = -J / H
            print(f"  -> Plotting {{name}} as PlaneY (y={{y_val:.2f}})")
            ax.plot_surface(U, np.full_like(U, y_val), V, alpha=0.2, color='green')
            return True

    # --- CASE 2: CYLINDERS (One quadratic term is zero) ---
    # Cylinder Z: x^2 + y^2 + ... = 0  (A ~ B, C=0)
    elif abs(A - B) < 1e-5 and abs(A) > 1e-9 and abs(C) < 1e-9:
        # Center calculation: x0 = -G/2A, y0 = -H/2B
        x0 = -G / (2 * A)
        y0 = -H / (2 * B)
        # Radius calculation: r = sqrt(x0^2 + y0^2 - J/A)
        term = (x0**2 + y0**2) - (J / A)
        if term > 0:
            r = math.sqrt(term)
            print(f"  -> Plotting {{name}} as CylinderZ (r={{r:.2f}})")
            
            z = np.linspace(-BOUNDS, BOUNDS, RESOLUTION)
            theta = np.linspace(0, 2*np.pi, RESOLUTION)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = r * np.cos(theta_grid) + x0
            y_grid = r * np.sin(theta_grid) + y0
            
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.3, color='cyan')
            return True

    # --- CASE 3: SPHERES (A ~ B ~ C) ---
    elif abs(A - B) < 1e-5 and abs(A - C) < 1e-5 and abs(A) > 1e-9:
        x0 = -G / (2 * A)
        y0 = -H / (2 * B)
        z0 = -I / (2 * C)
        term = (x0**2 + y0**2 + z0**2) - (J / A)
        if term > 0:
            r = math.sqrt(term)
            print(f"  -> Plotting {{name}} as Sphere (r={{r:.2f}})")
            
            u = np.linspace(0, 2 * np.pi, RESOLUTION)
            v = np.linspace(0, np.pi, RESOLUTION)
            x = r * np.outer(np.cos(u), np.sin(v)) + x0
            y = r * np.outer(np.sin(u), np.sin(v)) + y0
            z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z0
            
            ax.plot_surface(x, y, z, alpha=0.3, color='magenta')
            return True

    return False

def plot_3d_debug(local_vars):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Geometry Debugger")
    ax.set_xlim(-BOUNDS, BOUNDS)
    ax.set_ylim(-BOUNDS, BOUNDS)
    ax.set_zlim(-BOUNDS, BOUNDS)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    grid = np.linspace(-BOUNDS, BOUNDS, RESOLUTION)
    U, V = np.meshgrid(grid, grid)
    
    count = 0
    print("-" * 40)
    print("DEBUG: Decoding Quadric Surfaces...")

    for name, obj in local_vars.items():
        if name.startswith('_') or name in ['np', 'plt', 'mcdc']: continue
        
        # Duck typing: does it have quadric coefficients?
        if hasattr(obj, 'A') and hasattr(obj, 'J'):
            try:
                if decode_and_plot(ax, obj, name, U, V):
                    count += 1
                else:
                    print(f"  ! Could not decode shape for {{name}} (Complex Quadric?)")
            except Exception as e:
                print(f"  ! Error plotting {{name}}: {{e}}")
        
    if count == 0:
        ax.text(0, 0, 0, "No Surfaces Found", color='black')
        print("WARNING: No standard shapes found.")
    
    plt.show()

try:
    plot_3d_debug(locals())
except Exception as e:
    print(f"Visualization Fatal Error: {{e}}")
"""
        
        viz_file = "temp_viz_script.py"
        full_script = base_script + plot_code
        Path(viz_file).write_text(full_script)
        
        try:
            subprocess.run(["python", viz_file], check=True)
            console.print("[success]Visualization closed.[/success]")
        except subprocess.CalledProcessError:
            console.print("[error]Visualization failed.[/error]")
        finally:
            Path(viz_file).unlink(missing_ok=True)


    def create_step(self, step: str) -> str:
        console.print("\n")
        console.rule(f"[bold]CREATE YOUR {step.upper()}[/bold]")
        
        consecutive_errors = 0  
        max_consecutive_errors = 3 

        while True:
            prompt_text = f"Describe {step}(s) to add, 'view' to edit, or 'viz' to plot"
            if step == "surface": prompt_text += " (e.g., 'sphere radius 5')"
            elif step == "material": prompt_text += " (e.g., 'water')"
            
            goal = self.get_input(f"{prompt_text} (or Enter to finish):")

            if not goal:
                console.print(f"[success]Finished defining {step}s.[/success]")
                break
            
            if goal.lower().startswith("viz"):
                debug_mode = "debug" in goal.lower()
                self.run_visualization(debug_mode=debug_mode)
                continue

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
            
            # GENERATION LOOP
            for attempt in range(5):
                try:
                    with console.status(f"[bold yellow]Drafting {step}...[/bold yellow]", spinner="dots"):
                        response = self.agent.invoke({"messages": messages})
                        output = self._parse_agent_response(response)

                    # Handle empty output edge case
                    if not output or not output.strip():
                        output = "(No text response provided by Agent. It may have executed a tool silently.)"

                    current_count = len(self.builder.defined.get(step, set()))
                    
                    # Entities were added
                    if current_count > start_count:
                        console.print(Panel(output, title="[bold green]SUCCESS[/bold green]", border_style="green"))
                        new_code = self.builder.get_code_by_type(step)
                        console.print(f"\n[bold]Current {step.upper()} definitions:[/bold]")
                        console.print(Syntax(new_code, "python", theme="monokai"))
                        consecutive_errors = 0
                        break # Break inner loop, return to user prompt

                    # Print the output so the user sees questions OR errors
                    border_color = "red" if "ERROR" in output else "blue"
                    title = "ERROR" if "ERROR" in output else "TUTOR"
                    console.print(Panel(Markdown(output), title=f"[bold {border_color}]{title}[/bold {border_color}]", border_style=border_color))
                
                    clarification_phrases = [
                        "should that be", "what", "which",
                        "i suggest", "i recommend", "does this look correct", 
                        "would you like", "do you want", "can you confirm",
                        "how about", "already exists", "already defined", 
                        "different name", "unable to", "cannot create", 
                        "please specify", "please provide", "?",
                        "plan", "propose", "intend to", "clarify", "confirm", "suggest", "recommend",
                    ]

                    # If it's an error or a question, let the user respond
                    is_question = any(phrase in output.lower() for phrase in clarification_phrases)
                    is_error = "ERROR" in output or "exception" in output.lower()

                    if is_question or is_error:
                        user_reply = self.get_input("Your answer (or Enter to cancel):")
                        if not user_reply: 
                            console.print("[dim]Cancelling current attempt...[/dim]")
                            break 
                        
                        messages.append({"role": "assistant", "content": output})
                        messages.append({"role": "user", "content": user_reply})
                        continue 
                    else:
                        # Agent said something that wasn't a success and wasn't a question.
                        console.print("[warning]Agent finished without creating entities. Try rephrasing?[/warning]")
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