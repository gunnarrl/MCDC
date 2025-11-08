from typing import List, Dict, Set

class ScriptBuilder:
    """
    Tracks the state of the MCDC script being built.
    
    Maintains:
    - Lines of code in order
    - Set of defined entities by type (materials, surfaces, etc.)
    """
    def __init__(self):
        self.lines: List[str] = ["import mcdc", "import numpy as np", ""]
        self.defined: Dict[str, Set[str]] = {
            "material": set(),
            "surface": set(),
            "cell": set(),
            "source": set(),
            "tally": set(),
            "settings": set()
        }
    
    def add_line(self, code: str, entity_type: str, name: str):
        """Add a line of code and register the entity."""
        self.lines.append(code)
        self.defined[entity_type].add(name)
    
    def get_script(self) -> str:
        """Return complete script with mcdc.run() at the end."""
        return "\n".join(self.lines) + "\nmcdc.run()\n"
    
    def has_entity(self, entity_type: str, name: str) -> bool:
        """Check if an entity has been defined."""
        return name in self.defined.get(entity_type, set())
    
    def reset(self):
        """Reset the builder to initial state."""
        self.__init__()
