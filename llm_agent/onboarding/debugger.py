from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from typing import Any, List, Dict, Union
from prompt_toolkit import PromptSession
from langchain_core.runnables import Runnable

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

    def run(self):
        if not self.builder.entries:
            self.console.print("[warning]Script is empty.[/warning]")
            return

        self.console.print("\n")
        self.console.rule("[bold red]DEBUG MODE[/bold red]")
        self.console.print("Paste the error message (type END on new line to finish):")

        lines = []
        while True:
            line = self.session.prompt("")
            if line.strip().upper() == 'END': break
            lines.append(line)
        
        error_msg = "\n".join(lines)
        if not error_msg.strip(): return

        with self.console.status("[bold blue]Retrieving documentation...[/bold blue]"):
            search_query = error_msg[-200:] 
            docs = self.retriever.invoke(search_query)
            
            doc_context = ""
            for i, doc in enumerate(docs[:3]):
                doc_context += f"[Excerpt {i+1}]\n{doc.page_content}\n"

        current_script = self.builder.get_script()
        
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
        * **WAIT FOR CONFIRMATION:** Only execute the fix after the user confirms the plan.
        """

        messages = [{
            "role": "user",
            "content": (
                f"### BROKEN SCRIPT\n```python\n{current_script}\n```\n\n"
                f"### ERROR TRACEBACK\n```\n{error_msg}\n```\n\n"
                f"### RETRIEVED DOCUMENTATION\n{doc_context}\n\n"
                f"### INSTRUCTIONS\n{system_rules}\n"
            )
        }]

        # --- STEP 3: RUN AGENT LOOP ---
        while True:
            with self.console.status("[bold red]Analyzing...[/bold red]"):
                response = self.agent.invoke({"messages": messages})
                output = self._parse_agent_response(response)
            
            self.console.print(Panel(Markdown(output), title="Debugger", border_style="red"))
            messages.append({"role": "assistant", "content": output})

            user_reply = self.session.prompt("Reply (or 'exit'): ")
            if user_reply.lower() in ['no', 'exit', 'quit']: break
            
            messages.append({"role": "user", "content": user_reply})