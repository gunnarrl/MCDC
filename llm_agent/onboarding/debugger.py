from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.status import Status
from typing import Any, List, Dict, Union
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
        if isinstance(response, str): return response
        if isinstance(response, dict):
            if 'output' in response: return response['output']
            if 'content' in response: return response['content']
            if 'messages' in response and response['messages']:
                return self._extract_message_content(response['messages'][-1])
        if hasattr(response, 'content'): return response.content
        return str(response)

    def _extract_message_content(self, message: Any) -> str:
        """Extract content from a single message object."""
        content = None
        if isinstance(message, dict): content = message.get('content', str(message))
        elif hasattr(message, 'content'): content = message.content
        else: return str(message)
        
        if isinstance(content, list):
            # Handle multimodal content blocks
            text_parts = [b.get('text', '') if isinstance(b, dict) else str(b) for b in content]
            return '\n'.join(text_parts)
        return str(content)

    def show_diff(self, old_code: str, new_code: str):
        """Visualizes changes made to the script using a unified diff."""
        diff = list(unified_diff(
            old_code.splitlines(),
            new_code.splitlines(),
            fromfile='Current Script',
            tofile='Updated Script',
            lineterm=''
        ))
        
        if not diff:
            return

        diff_text = "\n".join(diff)
        syntax = Syntax(diff_text, "diff", theme="monokai", word_wrap=True)
        self.console.print("\n")
        self.console.print(Panel(syntax, title="[bold yellow]Applied Changes[/bold yellow]", border_style="yellow"))

    def run(self):
        if not self.builder.entries:
            self.console.print("[warning]Script is empty.[/warning]")
            return

        self.console.print("\n")
        self.console.rule("[bold red]DEBUG MODE[/bold red]")

        # --- STEP 1: CAPTURE ERROR ---
        error_msg = ""
        use_auto_run = self.session.prompt("Run script now to capture error? (y/n): ")
        
        if use_auto_run.strip().lower() == 'y':
            with self.console.status("[bold red]Running simulation...[/bold red]"):
                temp_file = "debug_temp.py"
                try:
                    # Write current state to temp file
                    with open(temp_file, "w") as f:
                        f.write(self.builder.get_script())
                    
                    # Run via subprocess to capture stderr safely
                    result = subprocess.run(
                        [sys.executable, temp_file],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        error_msg = result.stderr
                        self.console.print(Panel(Syntax(error_msg, "text"), title="Captured Traceback", border_style="red"))
                    else:
                        self.console.print("[green]Script ran successfully! No errors found.[/green]")
                        return 
                except Exception as e:
                    error_msg = str(e)
                finally:
                    if os.path.exists(temp_file): 
                        os.remove(temp_file)
        
        # Fallback: Manual Entry
        if not error_msg:
            self.console.print("Paste the error message (type END on new line to finish):")
            lines = []
            while True:
                line = self.session.prompt("")
                if line.strip().upper() == 'END': break
                lines.append(line)
            error_msg = "\n".join(lines)

        if not error_msg.strip(): 
            return

        # --- STEP 2: RETRIEVE CONTEXT ---
        with self.console.status("[bold blue]Retrieving documentation...[/bold blue]"):
            # Search using the error message (tail end usually has the specific exception)
            search_query = error_msg[-300:] 
            docs = self.retriever.invoke(search_query)
            
            doc_context = ""
            for i, doc in enumerate(docs[:3]):
                doc_context += f"[Excerpt {i+1}]\n{doc.page_content}\n"

        # --- STEP 3: PREPARE PROMPT ---
        raw_script = self.builder.get_script()
        script_lines = raw_script.split('\n')
        # Add line numbers for the LLM
        numbered_script = "\n".join([f"{i+1:03d} | {line}" for i, line in enumerate(script_lines)])

        system_rules = """
        You are an expert MCDC (Monte Carlo Dynamic Code) Debugger. 
        Your goal is to fix the broken script by comparing it and the error message against the Documentation and your Tool Definitions.

        ### DIAGNOSTIC HEURISTICS (Apply in order)
        1. **Geometry Overlaps/Gaps:** If error mentions "lost particle" or "overlap", check boolean logic in Cell `region`.
        2. **Material definitions:** If error mentions "cross-section", verify material names match the library.
        3. **Source/Geometry Mismatch:** If particles die immediately, check if Source `position` is inside a Cell.
        4. **Parameter Types:** Ensure lists and 2d arrays are used where MCDC expects them.
        5. **Other Errors:** Many other errors/issues can occur, if the issue doesn't match any of these heuristics, take a close look at all avaliable documentation to correctly diagnose the issue and propose a fix.

        ### RESOURCE PRIORITY
        1. **Tool Definitions:** Check your available tools (e.g., `create_surface`, `create_material`) to confirm correct parameter names and types.
        2. **Retrieved Docs:** Use the provided excerpts for conceptual rules.
        3. **Error Message & Script:** Use the traceback and script to locate the exact line and failure type.

        **If you cannot accurately identify a solution, do not propose one**
        
        ### OUTPUT FORMAT
        1. **The Diagnosis:** A 1-sentence explanation of *what* broke.
        2. **The Fix:** The exact Python code block to replace the broken part.
        3. **The Lesson:** A brief tip on how to avoid this.

        ### SAFETY PROTOCOL
        * **PROPOSE FIRST:** Do not execute any tools (like `manage_script` or `create_...`) in your first response.
        * **NO IMPORTS:** The script already imports `mcdc` and `numpy as np`. Do not include import statements in your "Fix" code blocks.
        * **WAIT FOR CONFIRMATION:** Only execute the fix after the user confirms the plan.
        """

        messages = [{
            "role": "user",
            "content": (
                f"### BROKEN SCRIPT (Line Numbers Added)\n```python\n{numbered_script}\n```\n\n"
                f"### ERROR TRACEBACK\n```\n{error_msg}\n```\n\n"
                f"### RETRIEVED DOCUMENTATION\n{doc_context}\n\n"
                f"### INSTRUCTIONS\n{system_rules}\n"
            )
        }]

        # --- STEP 4: RUN AGENT LOOP ---
        while True:
            # Snapshot script state before agent acts
            script_before = self.builder.get_script()
            
            with self.console.status("[bold red]Analyzing...[/bold red]"):
                response = self.agent.invoke({"messages": messages})
                output = self._parse_agent_response(response)
            
            # Print Agent's textual response
            self.console.print(Panel(Markdown(output), title="Debugger", border_style="red"))
            messages.append({"role": "assistant", "content": output})

            # Check if Agent executed a tool that changed the script
            script_after = self.builder.get_script()
            if script_before != script_after:
                self.show_diff(script_before, script_after)

            user_reply = self.session.prompt("Reply (or 'exit'): ")
            if user_reply.lower() in ['no', 'exit', 'quit']: break
            
            messages.append({"role": "user", "content": user_reply})