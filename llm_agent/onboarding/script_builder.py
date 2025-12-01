from typing import List, Dict, Set
from pathlib import Path
import ast
from collections import defaultdict, deque

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
    
    def get_script(self, include_run: bool = True, preserve_order: bool = True) -> str:
        """
        Reconstructs the script.
        
        Args:
            include_run: Whether to append mcdc.run() at the end
            preserve_order: If True, outputs entries in insertion order (best for complex scripts).
                           If False, groups by type (materials, surfaces, cells, etc.)
        """
        lines = list(self.imports)
        lines.append("") 
        
        if preserve_order:
            # INSERTION ORDER - respects dependencies
            for entry in self.entries:
                lines.append(entry['code'])
        else:
            # TYPE-BASED GROUPING - may break dependencies
            order = [
                "material", 
                "surface", 
                "cell", 
                "universe", 
                "lattice",
                "mesh",  # MUST come before tallies that reference them
                "source", 
                "tally", 
                "settings" 
            ]
            
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
                lines.append("")

        if include_run:
            lines.append("# === RUN ===")
            lines.append("mcdc.run()")
            
        return "\n".join(lines)
    
    def reorder(self):
        """
        Topologically sorts the script entries based on variable dependencies.
        Ensures that if 'A' uses 'B', 'B' is defined before 'A'.
        """
        # 1. Map Names to Indices
        name_to_idx = {entry['name']: i for i, entry in enumerate(self.entries)}
        
        # 2. Build Dependency Graph
        # Graph: A -> B means A must come BEFORE B
        adj = defaultdict(set)
        in_degree = defaultdict(int)
        
        # Initialize in_degree for all indices
        for i in range(len(self.entries)):
            in_degree[i] = 0

        for i, entry in enumerate(self.entries):
            # Parse the code to find what variables it references
            try:
                tree = ast.parse(entry['code'])
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        referenced_var = node.id
                        
                        # If this entry references another entity we track
                        if referenced_var in name_to_idx:
                            dep_idx = name_to_idx[referenced_var]
                            
                            # Self-references don't count (e.g. x = x + 1)
                            if dep_idx != i:
                                # dep_idx MUST precede i. Edge: dep -> i
                                if i not in adj[dep_idx]:
                                    adj[dep_idx].add(i)
                                    in_degree[i] += 1
            except Exception:
                continue

        # 3. Kahn's Algorithm for Topological Sort
        queue = deque()
        # Find all nodes with no dependencies (can be first)
        for i in range(len(self.entries)):
            if in_degree[i] == 0:
                queue.append(i)
        
        sorted_indices = []
        while queue:
            # Pop from left to preserve original stability for independent items
            u = queue.popleft()
            sorted_indices.append(u)
            
            # Use sorted iteration to ensure deterministic output
            for v in sorted(list(adj[u])):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # 4. Cycle Handling / Fallback
        # If we didn't visit everything, there is a cycle or graph error.
        # Just append the missing items in their original order.
        if len(sorted_indices) < len(self.entries):
            seen = set(sorted_indices)
            for i in range(len(self.entries)):
                if i not in seen:
                    sorted_indices.append(i)

        # 5. Apply Reordering
        self.entries = [self.entries[i] for i in sorted_indices]
        return True
        
    def parse_and_load(self, filepath: str) -> str:
        """
        Parses an existing Python file and populates the ScriptBuilder state.
        Uses AST to identify MCDC entities and variable names.
        """
        path = Path(filepath)
        if not path.exists():
            return f"Error: File {filepath} not found."

        try:
            source_code = path.read_text()
            tree = ast.parse(source_code)
        except Exception as e:
            return f"Error parsing syntax: {e}"

        # Reset current state
        self.reset()
        
        # Helper to extract source code segment from a node
        def get_segment(node):
            return ast.get_source_segment(source_code, node)

        # MCDC Class to Type Mapping
        type_map = {
            'Material': 'material', 'MaterialMG': 'material',
            'Surface': 'surface',
            'Cell': 'cell',
            'Universe': 'universe', 'Lattice': 'lattice',
            'MeshUniform': 'mesh', 'MeshStructured': 'mesh', 'Mesh': 'mesh',
            'Source': 'source',
            'Tally': 'tally', # Covers TallyGlobal, TallySurface, etc.
        }

        count = 0
        
        for node in tree.body:
            # 1. Handle Imports (Keep standard ones, ignore duplicates)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                code = get_segment(node)
                if code not in self.imports:
                    self.imports.append(code)
                continue

            # 2. Handle Assignments (e.g., m1 = mcdc.Material(...))
            if isinstance(node, ast.Assign):
                # We assume single assignment for MCDC entities (m1 = ...)
                target = node.targets[0]
                value = node.value
                
                # Check if it's an MCDC call
                if isinstance(value, ast.Call):
                    # Resolve function name (handle mcdc.Material and mcdc.Surface.PlaneX)
                    func_name = ""
                    if isinstance(value.func, ast.Attribute):
                        if isinstance(value.func.value, ast.Name) and value.func.value.id == 'mcdc':
                            func_name = value.func.attr # e.g. 'Material'
                        elif isinstance(value.func.value, ast.Attribute): # e.g. mcdc.Surface.PlaneX
                            func_name = value.func.value.attr # 'Surface'
                    
                    # Determine Type
                    entity_type = "other"
                    for key, val in type_map.items():
                        if func_name.startswith(key):
                            entity_type = val
                            break
                    
                    # Extract Name
                    var_name = target.id if isinstance(target, ast.Name) else "unknown"
                    
                    self.add_line(get_segment(node), entity_type, var_name)
                    count += 1
                    continue

                # Handle Settings (mcdc.settings.x = y)
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute):
                    if target.value.attr == 'settings':
                        self.add_line(get_segment(node), "settings", target.attr)
                        count += 1
                        continue

            # 3. Handle Standalone Expressions (e.g., mcdc.Source(...) without assignment)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                # Similar logic to assignments, but name is generic
                if isinstance(node.value.func, ast.Attribute):
                    attr_name = node.value.func.attr
                    if "Source" in attr_name:
                        self.add_line(get_segment(node), "source", f"source_{len(self.defined['source'])}")
                        count += 1
                        continue
                    if "run" in attr_name:
                        continue

            # 4. Fallback: Add everything else as 'other' (comments, math, etc)
            code_segment = get_segment(node)
            if code_segment and "mcdc.run" not in code_segment:
                self.add_line(code_segment, "other", "generic_code")

        return f"Successfully loaded {count} entities from {path.name}."

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
    
    def replace_code(self, old_code: str, new_code: str) -> bool:
        """
        Replace a specific code segment with new code.
        Returns True if replacement was successful.
        """
        for entry in self.entries:
            if entry['code'].strip() == old_code.strip():
                entry['code'] = new_code
                return True
        return False
    
    def replace_entity(self, entity_type: str, name: str, new_code: str) -> bool:
        """
        Replace an entity's code while preserving its position in the script.
        This is better than delete+recreate for debugging.
        Returns True if replacement was successful.
        """
        for entry in self.entries:
            if entry['type'] == entity_type and entry['name'] == name:
                entry['code'] = new_code
                return True
        return False
    
    def insert_entity(self, code: str, entity_type: str, name: str, 
                     before: str = None, after: str = None, position: int = None) -> bool:
        """
        Insert a new entity at a specific position in the script.
        
        Args:
            code: The Python code to insert
            entity_type: Type of entity ('material', 'surface', 'cell', etc.)
            name: Variable name for the entity
            before: Name of entity to insert before (optional)
            after: Name of entity to insert after (optional)
            position: Exact index to insert at (optional)
        
        Priority: position > before > after > append to end
        Returns True if successful
        """
        # Check if entity already exists
        if self.has_entity(entity_type, name):
            return False
        
        new_entry = {
            'code': code,
            'type': entity_type,
            'name': name
        }
        
        # Determine insertion point
        insert_idx = None
        
        if position is not None:
            # Direct index insertion
            insert_idx = max(0, min(position, len(self.entries)))
        
        elif before:
            # Find the entity to insert before
            for idx, entry in enumerate(self.entries):
                if entry['name'] == before:
                    insert_idx = idx
                    break
        
        elif after:
            # Find the entity to insert after
            for idx, entry in enumerate(self.entries):
                if entry['name'] == after:
                    insert_idx = idx + 1
                    break
        
        # Insert or append
        if insert_idx is not None:
            self.entries.insert(insert_idx, new_entry)
        else:
            self.entries.append(new_entry)
        
        # Register in defined set
        if entity_type not in self.defined:
            self.defined[entity_type] = set()
        self.defined[entity_type].add(name)
        
        return True
    
    def find_entity_index(self, entity_type: str, name: str) -> int:
        """
        Find the index of an entity in the entries list.
        Returns -1 if not found.
        """
        for idx, entry in enumerate(self.entries):
            if entry['type'] == entity_type and entry['name'] == name:
                return idx
        return -1
    
    def get_code_by_type(self, entity_type: str) -> str:
        """Get the code for all entities of a specific type."""
        lines = [e['code'] for e in self.entries if e['type'] == entity_type]
        if not lines:
            return "No entries defined."
        return "\n".join(lines)

    def reset(self):
        self.__init__()