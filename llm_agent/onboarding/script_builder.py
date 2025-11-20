from typing import List, Dict, Set

class ScriptBuilder:
    """
    Tracks the state of the MCDC script being built.
    Now supports Undo and Delete operations.
    """
    def __init__(self):
        self.imports = ["import mcdc", "import numpy as np", ""]
        # Store entries as dictionaries to enable targeted deletion
        # Format: {'code': str, 'type': str, 'name': str}
        self.entries: List[Dict[str, str]] = []
        
        self.defined: Dict[str, Set[str]] = {
            "material": set(),
            "surface": set(),
            "cell": set(),
            "universe": set(),
            "lattice": set(),
            "mesh": set(),
            "source": set(),
            "tally": set(),
            "settings": set()
        }
    
    def add_line(self, code: str, entity_type: str, name: str):
        """Add a line of code and register the entity."""
        self.entries.append({
            'code': code,
            'type': entity_type, 
            'name': name
        })
        
        # Auto-create key if it doesn't exist to prevent KeyErrors
        if entity_type not in self.defined:
            self.defined[entity_type] = set()
            
        self.defined[entity_type].add(name)
    
    def get_script(self) -> str:
        """Reconstruct the script from imports and active entries."""
        script_lines = list(self.imports)
        for entry in self.entries:
            script_lines.append(entry['code'])
        script_lines.append("\nmcdc.run()\n")
        return "\n".join(script_lines)
    
    def has_entity(self, entity_type: str, name: str) -> bool:
        return name in self.defined.get(entity_type, set())

    def undo_last(self) -> str:
        """Remove the most recently added entity."""
        if not self.entries:
            return "Nothing to undo."
        
        last_entry = self.entries.pop()
        # Use discard to avoid errors if key missing
        if last_entry['type'] in self.defined:
            self.defined[last_entry['type']].discard(last_entry['name'])
            
        return f"Undid creation of {last_entry['type']} '{last_entry['name']}'."

    def delete_entity(self, entity_type: str, name: str) -> bool:
        """Delete a specific entity by name and type."""
        if not self.has_entity(entity_type, name):
            return False
        
        # Filter out the entry with matching type and name
        initial_count = len(self.entries)
        self.entries = [
            e for e in self.entries 
            if not (e['type'] == entity_type and e['name'] == name)
        ]
        
        if len(self.entries) < initial_count:
            if entity_type in self.defined:
                self.defined[entity_type].discard(name)
            return True
        return False

    def reset(self):
        self.__init__()