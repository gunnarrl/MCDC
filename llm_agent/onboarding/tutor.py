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
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
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
    "markdown.code": "bold dark_green", 
    "code": "bold dark_green",
})

console = Console(theme=custom_theme)

class BackToMenu(Exception):
    """Raised when the user presses ESC to return to the main menu."""
    pass

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

        self.kb = KeyBindings()

        @self.kb.add(Keys.Escape)
        def _(event):
            event.app.exit(result="__ESCAPE__")

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
        Input handler
        """
        if prompt_text:
            console.print(f"[user]{prompt_text}[/user]")
        
        result = self.session.prompt(
            [('class:prompt', '> ')], 
            style=self.prompt_style,
            key_bindings=self.kb 
        )

        # Check if the user pressed Escape
        if result == "__ESCAPE__":
            raise BackToMenu()

        return result.strip()

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
        lesson = CONCEPT_LESSONS.get(step)
        if not lesson:
            console.print(f"[error]No lesson found for {step}[/error]")
            return False
        console.print("\n")
        console.rule(f"[step] STEP: {step.upper()} [/step]")
        
        # 1. Concept Block
        console.print(Panel(
            Markdown(lesson['concept']),
            title="Concept",
            border_style="magenta"
        ))

        # 2. Syntax
        if 'syntax' in lesson:
            console.print("\n[bold]Syntax Template:[/bold]")
            syntax_highlighted = Syntax(
                lesson['syntax'], 
                "python", 
                theme="monokai", 
                line_numbers=False, 
                word_wrap=True
            )
            console.print(syntax_highlighted)
        
        # 3. Key Parts
        console.print("\n[bold]Parameters:[/bold]")
        console.print(Markdown(lesson['parts']))

        # 4. Tips
        if 'tips' in lesson:
            console.print("\n[bold yellow]Tips & Common Mistakes:[/bold yellow]")
            for tip in lesson['tips']:
                console.print(Markdown(f"* {tip}"))

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

    def run_visualization(self, axis: str = 'z', position: float = 0.0):
        """
        Smart Visualizer:
        - Extracts 'region' strings directly from ScriptBuilder history.
        - Supports slicing along X, Y, or Z axis.
        """
        import re
        
        # 1. Extract Cell Logic Strings
        cell_logic_map = {}
        has_cells = False
        
        for entry in self.builder.entries:
            if entry['type'] == 'cell':
                has_cells = True
                match = re.search(r"region=(.+?)(?:,\s*\w+=|\))", entry['code'])
                if match:
                    cell_logic_map[entry['name']] = match.group(1).strip()

        mode_name = f"Slice Scanner ({axis.upper()}={position})" if has_cells else "3D Wireframe"
        console.print(f"\n[bold yellow]Generating Geometry Preview ({mode_name})...[/bold yellow]")
        
        base_script = self.builder.get_script(include_run=False)

        # ==============================================================================
        # MODE A: SLICE SCANNER (Dynamic Axis)
        # ==============================================================================
        slice_code = f"""
# --- SLICE VISUALIZATION APPENDED BY TUTOR ---
import matplotlib.pyplot as plt
import numpy as np

# INJECTED SETTINGS
CELL_REGIONS = {str(cell_logic_map)}
SLICE_AXIS = '{axis}'
SLICE_VAL = {position}
BOUNDS = 15.0
RES = 150

def run_slice_viz(local_vars):
    print(f"Scanning {{RES}}x{{RES}} pixels at {{SLICE_AXIS.upper()}}={{SLICE_VAL}}...")

    # 1. Identify Surfaces
    surfaces = {{}}
    for name, obj in local_vars.items():
        if hasattr(obj, 'A') and hasattr(obj, 'J'):
            surfaces[name] = obj

    # 2. Helper: Evaluate Surface Equation
    def eval_surf(s, x, y, z):
        return (getattr(s,'A',0)*x**2 + getattr(s,'B',0)*y**2 + getattr(s,'C',0)*z**2 +
                getattr(s,'D',0)*x*y  + getattr(s,'E',0)*y*z  + getattr(s,'F',0)*z*x +
                getattr(s,'G',0)*x    + getattr(s,'H',0)*y    + getattr(s,'I',0)*z + 
                getattr(s,'J',0))

    # 3. Setup Dynamic Grid based on Axis
    u = np.linspace(-BOUNDS, BOUNDS, RES)
    v = np.linspace(-BOUNDS, BOUNDS, RES)
    U, V = np.meshgrid(u, v)
    
    # Map 2D grid (U,V) to 3D coordinates (PX, PY, PZ)
    if SLICE_AXIS == 'z':
        PX, PY, PZ = U, V, np.full_like(U, SLICE_VAL)
        xlabel, ylabel = 'X [cm]', 'Y [cm]'
    elif SLICE_AXIS == 'y':
        PX, PY, PZ = U, np.full_like(U, SLICE_VAL), V
        xlabel, ylabel = 'X [cm]', 'Z [cm]'
    elif SLICE_AXIS == 'x':
        PX, PY, PZ = np.full_like(U, SLICE_VAL), U, V
        xlabel, ylabel = 'Y [cm]', 'Z [cm]'

    img = np.zeros((RES, RES)) - 1 
    sorted_surfs = sorted(surfaces.keys(), key=len, reverse=True)
    cell_names = list(CELL_REGIONS.keys())
    
    # 4. Scan Grid
    for i in range(RES):
        for j in range(RES):
            # Get real 3D coordinates for this pixel
            px, py, pz = PX[i,j], PY[i,j], PZ[i,j]
            
            for c_idx, c_name in enumerate(cell_names):
                logic = CELL_REGIONS[c_name]
                try:
                    for s_name in sorted_surfs:
                        if s_name in logic:
                            # Evaluate using the 3D coordinate for this pixel
                            val = eval_surf(surfaces[s_name], px, py, pz)
                            logic = logic.replace(f"+{{s_name}}", str(val > 0))
                            logic = logic.replace(f"-{{s_name}}", str(val < 0))
                    
                    logic = logic.replace("&", " and ").replace("|", " or ").replace("~", " not ")
                    if eval(logic):
                        img[i,j] = c_idx
                        break 
                except Exception:
                    pass

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8,8))
    
    if len(cell_names) > 0:
        cmap = plt.get_cmap('tab20', len(cell_names))
    else:
        cmap = plt.get_cmap('Greys')
        
    masked_img = np.ma.masked_where(img == -1, img)
    
    ax.imshow(masked_img, origin='lower', extent=[-BOUNDS, BOUNDS, -BOUNDS, BOUNDS], 
               cmap=cmap, vmin=0, vmax=len(cell_names)-1)
    
    # Legend
    from matplotlib.patches import Patch
    patches = [Patch(color=cmap(i), label=name) for i, name in enumerate(cell_names)]
    ax.legend(handles=patches, loc='upper right', title="Cells")
    
    ax.set_title(f"Cell Region Slice ({{SLICE_AXIS.upper()}}={{SLICE_VAL}})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle='--')
    plt.show()

try:
    run_slice_viz(locals())
except Exception as e:
    print(f"Slice Viz Error: {{e}}")
"""

        # MODE B: WIREFRAME (For Surfaces)
        wireframe_code = r"""
# --- WIREFRAME VISUALIZATION APPENDED BY TUTOR ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math

def run_wireframe_viz(local_vars):
    BOUNDS = 15.0
    RES = 20
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Surface Wireframe Preview")
    ax.set_xlim(-BOUNDS, BOUNDS); ax.set_ylim(-BOUNDS, BOUNDS); ax.set_zlim(-BOUNDS, BOUNDS)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    grid = np.linspace(-BOUNDS, BOUNDS, RES)
    U, V = np.meshgrid(grid, grid)
    found = 0

    print("-" * 40)
    print("Scanning Surfaces...")

    for name, obj in local_vars.items():
        if not (hasattr(obj, 'A') and hasattr(obj, 'J')): continue
        
        try:
            A, B, C = getattr(obj,'A',0), getattr(obj,'B',0), getattr(obj,'C',0)
            G, H, I, J = getattr(obj,'G',0), getattr(obj,'H',0), getattr(obj,'I',0), getattr(obj,'J',0)
            
            if abs(A)+abs(B)+abs(C) < 1e-9:
                if abs(I) > 1e-9:   # Plane Z
                    val = -J/I; ax.plot_surface(U, V, np.full_like(U, val), alpha=0.2, color='blue')
                elif abs(G) > 1e-9: # Plane X
                    val = -J/G; ax.plot_surface(np.full_like(U, val), U, V, alpha=0.2, color='red')
                elif abs(H) > 1e-9: # Plane Y
                    val = -J/H; ax.plot_surface(U, np.full_like(U, val), V, alpha=0.2, color='green')
                found += 1

            elif abs(C) < 1e-9 and abs(A-B) < 1e-5 and abs(A) > 1e-9: # Cylinder Z
                x0, y0 = -G/(2*A), -H/(2*B)
                r = math.sqrt(max(0, x0**2 + y0**2 - J/A))
                z = np.linspace(-BOUNDS, BOUNDS, RES)
                th = np.linspace(0, 2*np.pi, RES)
                TH, Z = np.meshgrid(th, z)
                Xg = r * np.cos(TH) + x0
                Yg = r * np.sin(TH) + y0
                ax.plot_surface(Xg, Yg, Z, alpha=0.3, color='cyan')
                found += 1

            elif abs(A-B) < 1e-5 and abs(A-C) < 1e-5 and abs(A) > 1e-9: # Sphere
                x0, y0, z0 = -G/(2*A), -H/(2*B), -I/(2*C)
                r = math.sqrt(max(0, x0**2 + y0**2 + z0**2 - J/A))
                u = np.linspace(0, 2*np.pi, RES)
                v = np.linspace(0, np.pi, RES)
                Xg = r * np.outer(np.cos(u), np.sin(v)) + x0
                Yg = r * np.outer(np.sin(u), np.sin(v)) + y0
                Zg = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z0
                ax.plot_surface(Xg, Yg, Zg, alpha=0.3, color='magenta')
                found += 1
                
        except Exception as e:
            print(f"  ! Error plotting {name}: {e}")

    if found == 0:
        ax.text(0,0,0, "No Surfaces", color='k')
    plt.show()

try:
    run_wireframe_viz(locals())
except Exception as e:
    print(f"Wireframe Error: {e}")
"""

        # Select and Inject
        plot_code = slice_code if has_cells else wireframe_code
        
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
                # Check if we are doing 2D Slicing (Cells exist) or 3D Wireframe
                has_cells = len(self.builder.defined.get('cell', set())) > 0
                
                # Default defaults
                viz_axis = 'z'
                viz_pos = 0.0

                if has_cells:
                    # Prompt user for slice details
                    slice_input = self.get_input("Enter slice (e.g., 'z=5', 'y=0') [Default: z=0]:")
                    
                    if slice_input:
                        import re
                        # Regex to capture 'x', 'y', or 'z' and the number
                        match = re.search(r"([xyz])\s*=?\s*([-\d.]+)", slice_input.lower())
                        if match:
                            viz_axis = match.group(1)
                            viz_pos = float(match.group(2))
                        else:
                            console.print("[warning]Could not parse input. Using default Z=0.[/warning]")

                # Pass these values to the visualizer
                self.run_visualization(axis=viz_axis, position=viz_pos)
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
        console.rule("[bold cyan]MCDC Onboarding[/bold cyan]")
        console.print("Welcome! Select a category to edit or add components.")
        console.print("MCDC scripts are built incrementally. Start with materials and surfaces, then move to cells, hierarchy, sources, tallies, and settings.\n")
        console.print("You can also view/edit the full script or save your progress.\n")
        console.print("[dim]Note: You can type 'undo' at any prompt to undo the last action.[/dim]")
        console.print("[dim]Tip: Use 'view' to see the current script and make edits.[/dim]")
        console.print("[dim]Tip: Use 'viz' to visualize your geometry after defining surfaces and cells.[/dim]")
        
        # Map Menu Options to (Internal Step Name, Display Label)
        menu_options = {
            "1": ("material", "Materials"),
            "2": ("surface", "Surfaces"),
            "3": ("cell", "Cells"),
            "4": ("hierarchy", "Universes & Lattices"),
            "5": ("source", "Sources"),
            "6": ("tally", "Tallies"),
            "7": ("settings", "Settings"),
        }

        while True:
            console.print("\n[bold]Main Menu:[/bold]")
            try:
                # Print dynamic menu with counts
                for key, (step_id, label) in menu_options.items():
                    # Check how many items exist for this step
                    # Special handling for hierarchy which covers multiple types
                    if step_id == "hierarchy":
                        count = len(self.builder.defined.get('universe', [])) + len(self.builder.defined.get('lattice', []))
                    else:
                        count = len(self.builder.defined.get(step_id, set()))
                    
                    status = f"[green]({count} defined)[/green]" if count > 0 else "[dim](empty)[/dim]"
                    console.print(f"  [{key}] {label} {status}")
                
                console.print("  \[v] View/Edit/Add Full Script")
                console.print("  \[l] Load Script from File")
                console.print("  \[s] Save & Exit")
                console.print("  \[q] Quit (No Save)")
                
                choice = self.get_input("Select option:")
                
                if choice in menu_options:
                    step_id, label = menu_options[choice]
                    
                    # First time visiting this step? Teach the concept.
                    # (We check if the builder is empty for this specific type)
                    is_empty = False
                    if step_id == "hierarchy":
                        is_empty = (len(self.builder.defined.get('universe', [])) + len(self.builder.defined.get('lattice', []))) == 0
                    else:
                        is_empty = len(self.builder.defined.get(step_id, set())) == 0

                    if is_empty:
                        # If user says "No" to "Ready to create?", we just go back to menu
                        if self.teach_concept(step_id):
                            self.create_step(step_id)
                    else:
                        # Already knows it, go straight to builder
                        self.create_step(step_id)
                        
                elif choice.lower() == 'v':
                    self._handle_view_mode()
                
                elif choice.lower() == 'l':
                    filepath = self.get_input("Enter path to python script (or drag into terminal):")
                    filepath = filepath.strip('"').strip("'")
                    
                    if filepath:
                        with console.status(f"[bold yellow]Parsing {filepath}...[/bold yellow]"):
                            result = self.builder.parse_and_load(filepath)
                        
                        if "Error" in result:
                            console.print(f"[bold red]{result}[/bold red]")
                        else:
                            console.print(f"[bold green]{result}[/bold green]")
                            # Show the user what we loaded
                            self._print_script()
                    
                elif choice.lower() == 's':
                    final_script = self.builder.get_script()
                    console.print("\n")
                    console.print(Syntax(final_script, "python", theme="monokai"))
                    
                    filename = self.get_input("Filename (e.g., simulation.py):")
                    if not filename: filename = "mcdc_simulation.py"
                    if not filename.endswith(".py"): filename += ".py"
                    
                    try:
                        Path(filename).write_text(final_script)
                        console.print(f"[success]Saved to {filename}[/success]")
                        break
                    except Exception as e:
                        console.print(f"[error]Error saving: {e}[/error]")
                        
                elif choice.lower() == 'q':
                    if Confirm.ask("Quit without saving?"):
                        console.print("[dim]Exiting...[/dim]")
                        break
                else:
                    console.print("[error]Invalid option[/error]")
            except BackToMenu:
                console.print("\n[bold yellow]Returning to Main Menu...[/bold yellow]")
                continue

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