from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from typing import Any
from prompt_toolkit import PromptSession
from langchain_core.runnables import Runnable
import subprocess
import sys
import os
from difflib import unified_diff

class DebugHandler:
    def __init__(self, agent: Runnable, builder: Any, session: PromptSession, console: Console, retriever: Any):
        self.agent = agent
        self.builder = builder
        self.session = session
        self.console = console
        self.retriever = retriever

    def _parse_agent_response(self, response: Any) -> str:
        """Helper to extract text from various LangChain response formats."""
        if isinstance(response, str): 
            return response
        if isinstance(response, dict):
            if 'output' in response: 
                return response['output']
            if 'content' in response: 
                return response['content']
            if 'messages' in response and response['messages']:
                return self._extract_message_content(response['messages'][-1])
        if hasattr(response, 'content'): 
            return response.content
        
        # Handle empty content gracefully
        return ""

    def _extract_message_content(self, message: Any) -> str:
        """Extract content from a single message object."""
        content = None
        if isinstance(message, dict): 
            content = message.get('content', str(message))
        elif hasattr(message, 'content'): 
            content = message.content
        else: 
            return str(message)
        
        if isinstance(content, list):
            # Handle multimodal content blocks - filter out empty ones
            text_parts = []
            for b in content:
                if isinstance(b, dict) and b.get('type') == 'text':
                    text = b.get('text', '').strip()
                    if text:  # Only add non-empty text
                        text_parts.append(text)
                elif isinstance(b, str) and b.strip():
                    text_parts.append(b)
            return '\n'.join(text_parts)
        
        return str(content) if content else ""

    def show_diff(self, old_code: str, new_code: str):
        """Visualizes changes made to the script using a unified diff."""
        diff = list(unified_diff(
            old_code.splitlines(),
            new_code.splitlines(),
            fromfile='Before',
            tofile='After',
            lineterm=''
        ))
        
        if not diff:
            self.console.print("[dim]No changes detected.[/dim]")
            return

        diff_text = "\n".join(diff)
        syntax = Syntax(diff_text, "diff", theme="monokai", word_wrap=True)
        self.console.print("\n")
        self.console.print(Panel(syntax, title="[bold yellow]Changes Applied[/bold yellow]", border_style="yellow"))

    def run_script(self) -> tuple[int, str]:
        """
        Run the current script and capture output.
        Returns (return_code, stderr_output)
        """
        temp_file = "debug_temp.py"
        try:
            with open(temp_file, "w") as f:
                # IMPORTANT: Use preserve_order=True for debugging
                f.write(self.builder.get_script(preserve_order=True))
            
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return result.returncode, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "ERROR: Script execution timed out (30s limit)"
        except Exception as e:
            return -1, f"ERROR: Failed to run script: {str(e)}"
        finally:
            if os.path.exists(temp_file): 
                os.remove(temp_file)

    def run(self):
        if not self.builder.entries:
            self.console.print("[warning]Script is empty.[/warning]")
            return

        self.console.print("\n")
        self.console.rule("[bold red]DEBUG MODE[/bold red]")
        self.console.print("[dim]The debugger will run your script and help fix errors.[/dim]")
        self.console.print("[dim]Commands: 'yes' to apply fix, 'run' to test again, 'no'/'exit' to quit[/dim]\n")

        # Enhanced system prompt for debugging context
        debug_context = """
=== DEBUG MODE INSTRUCTIONS ===

You are debugging an EXISTING, WORKING MCDC script that has an error.
Your ONLY job is to FIX THE SPECIFIC ERROR shown in the traceback.

**CRITICAL RULES**:
1. **NEVER suggest rewriting the entire script**
2. **NEVER suggest converting to different functions** - the script uses direct mcdc.Material(), mcdc.Surface.PlaneX() calls and that's CORRECT
3. **DO NOT create new entities** unless absolutely necessary to fix the error
4. **ONLY modify/delete/recreate the specific broken entity**

**COMMON ERRORS & FIXES**:

1. **NameError: 'X' is not defined**
   - Cause: Entity referenced but never created
   - Fix: Use `insert_entity()` to add the missing definition BEFORE the entity that uses it
   - Example: If `Cell(region=+plane_x)` fails, insert: `insert_entity(code='plane_x = mcdc.Surface.PlaneX(x=5.0)', entity_type='surface', name='plane_x', before='fuel_cell')`

2. **NameError due to wrong order**
   - Cause: Entity used before it's defined (both exist but wrong order)
   - Fix: Use `replace_entity()` won't help here. Instead: delete the one that comes first, then recreate it after dependencies

3. **IndexError in mesh tally** (index -XXX out of bounds)
   - Cause: Mesh dimension has single grid point (degenerate dimension)
   - Fix: Use `replace_entity()` to recreate mesh WITHOUT the degenerate dimension
   - Example: If `MeshStructured(x=..., y=np.array([0.0]), z=...)` fails, replace with `MeshStructured(x=..., z=...)` (omit y entirely for 2D XZ slice)
   - **KEY**: For 2D meshes, just don't pass the constant dimension - MCDC will handle it

4. **TypeError: scatter must be 2D**
   - Fix: Use `replace_entity()` wrapping in extra brackets: `scatter=np.array([[0.95]])` not `np.array([0.95])`

5. **Lost particle / geometry errors**
   - Check cell region boolean logic: use `&` not `and`, `|` not `or`

**YOUR WORKFLOW**:
1. Call `manage_script(action='get')` to see current state
2. Identify the SPECIFIC broken entity from the error traceback
3. Choose the right tool:
   - **Missing entity**: `insert_entity(code, type, name, before='entity_that_uses_it')`
   - **Broken entity**: `replace_entity(type, name, new_code)` (preserves position)
   - **Wrong order**: `delete_entity()` then `create_*()` (moves to end)
4. Explain briefly what you fixed

**DO NOT**:
- Suggest rewriting materials, surfaces, or other working entities
- Mention "helper functions" or "tool conversions"
- Create duplicate entities
- Make changes unrelated to the error
"""

        # --- MAIN DEBUG LOOP ---
        iteration = 0
        max_iterations = 5
        
        while iteration < max_iterations:
            iteration += 1
            self.console.print(f"\n[bold cyan]--- Debug Iteration {iteration}/{max_iterations} ---[/bold cyan]")
            
            # --- STEP 1: RUN AND CAPTURE ERROR ---
            with self.console.status("[bold red]Running simulation...[/bold red]"):
                return_code, error_msg = self.run_script()
            
            if return_code == 0:
                self.console.print("[bold green]✓ Script executed successfully! All errors fixed.[/bold green]")
                return
            
            self.console.print(Panel(Syntax(error_msg, "text"), title="Error Traceback", border_style="red"))

            # --- STEP 2: RETRIEVE DOCS ---
            with self.console.status("[bold blue]Searching documentation...[/bold blue]"):
                # Extract key error terms for better search
                error_lines = error_msg.split('\n')
                search_terms = []
                for line in error_lines:
                    if 'Error:' in line or 'Exception:' in line:
                        search_terms.append(line)
                search_query = ' '.join(search_terms[-3:]) if search_terms else error_msg[-500:]
                
                docs = self.retriever.invoke(search_query)
                
                doc_context = ""
                for i, doc in enumerate(docs[:3]):
                    source = doc.metadata.get("source", "Unknown")
                    doc_context += f"[Doc {i+1} - {source}]\n{doc.page_content}\n\n"

            # --- STEP 3: BUILD DEBUG PROMPT ---
            raw_script = self.builder.get_script(preserve_order=True)
            script_lines = raw_script.split('\n')
            numbered_script = "\n".join([f"{i+1:03d} | {line}" for i, line in enumerate(script_lines)])

            # Track state before agent acts
            script_before = raw_script
            entities_before = {k: set(v) for k, v in self.builder.defined.items()}

            messages = [{
                "role": "user",
                "content": f"""{debug_context}

### CURRENT SCRIPT STATE:
```python
{numbered_script}
```

### ERROR TRACEBACK:
```
{error_msg}
```

### RELEVANT DOCUMENTATION:
{doc_context}

### YOUR TASK:
Fix ONLY the specific error shown above. Use manage_script, insert_entity, and create_* tools as needed.
"""
            }]
            
            # --- STEP 4: AGENT PROCESSES ---
            with self.console.status("[bold red]Analyzing error and proposing fix...[/bold red]"):
                try:
                    response = self.agent.invoke({"messages": messages})
                    output = self._parse_agent_response(response)
                except Exception as e:
                    self.console.print(f"[error]Agent error: {e}[/error]")
                    break
            
            # Handle case where agent only called tools without text explanation
            if not output or not output.strip():
                script_after = self.builder.get_script(preserve_order=True)
                if script_before != script_after:
                    output = "I've applied the fix using the available tools."
                else:
                    output = "I couldn't determine a fix for this error."
            
            # Display agent's explanation
            if output:
                self.console.print(Panel(Markdown(output), title="Debugger Analysis", border_style="red"))
            
            # Show what changed
            script_after = self.builder.get_script(preserve_order=True)
            if script_before != script_after:
                self.show_diff(script_before, script_after)
                
                # Show entity changes
                for etype, names in self.builder.defined.items():
                    added = names - entities_before.get(etype, set())
                    removed = entities_before.get(etype, set()) - names
                    if added:
                        self.console.print(f"[green]+ Added {etype}(s): {', '.join(added)}[/green]")
                    if removed:
                        self.console.print(f"[red]- Removed {etype}(s): {', '.join(removed)}[/red]")
            
            # --- STEP 5: USER DECISION ---
            # Check if agent made changes
            changes_made = script_before != script_after
            
            if changes_made:
                prompt_text = "Accept these changes? ('yes' to test, 'no' to revert, 'exit'): "
            else:
                prompt_text = "No changes made. ('run' to retry, 'exit' to quit): "
            
            user_cmd = self.session.prompt(prompt_text).strip().lower()
            
            if user_cmd in ['exit', 'quit', 'q']:
                break
            
            elif user_cmd in ['no', 'n', 'revert']:
                if changes_made:
                    # Revert by reloading from script_before
                    self.console.print("[yellow]Reverting changes...[/yellow]")
                    # We'd need a way to restore state - for now just warn
                    self.console.print("[warning]Cannot auto-revert. Use 'undo' from main menu if needed.[/warning]")
                break
            
            elif user_cmd in ['yes', 'y', 'apply', 'run', '']:
                # Loop back to test the fix
                continue
            
            else:
                self.console.print(f"[warning]Unknown command: {user_cmd}[/warning]")
        
        if iteration >= max_iterations:
            self.console.print(f"[warning]Reached maximum iterations ({max_iterations}). Exiting debug mode.[/warning]")
            self.console.print("[dim]You can continue editing from the main menu.[/dim]")