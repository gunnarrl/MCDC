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
    
    def get_script(self, include_run: bool = True) -> str:
        """
        Reconstructs the script with strict MCDC ordering:
        Materials -> Surfaces -> Cells -> Universes -> Lattices -> Sources -> Tallies -> Settings
        """
        
        order = [
            "material", 
            "surface", 
            "cell", 
            "universe", 
            "lattice", 
            "source", 
            "tally", 
            "settings" 
        ]
        
        lines = list(self.imports)
        lines.append("") 
        
        # Collect entries by type
        for section in order:
            section_entries = [e for e in self.entries if e['type'] == section]
            
            if section_entries:
                lines.append(f"# === {section.upper()} DEFINITIONS ===")
                for entry in section_entries:
                    lines.append(entry['code'])
                lines.append("") 

        # everything else
        others = [e for e in self.entries if e['type'] not in order]
        if others:
            lines.append("# === OTHER ===")
            for entry in others:
                lines.append(entry['code'])

        if include_run:
            lines.append("# === RUN ===")
            lines.append("mcdc.run()")
            
        return "\n".join(lines)
    
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
    
    def get_code_by_type(self, entity_type: str) -> str:
        """Get the code for all entities of a specific type."""
        lines = [e['code'] for e in self.entries if e['type'] == entity_type]
        if not lines:
            return "No entries defined."
        return "\n".join(lines)

    def reset(self):
        self.__init__()