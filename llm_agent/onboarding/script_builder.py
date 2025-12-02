from typing import List, Dict, Set
from pathlib import Path
import ast
import re
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
        
        # Auto-sort to fix dependencies and grouping
        self.reorder()
    
    def reorder(self):
        """
        Topologically sorts the script entries based on variable dependencies.
        
        Updates:
        1. Ensures 'settings' always appear at the bottom.
        2. Parses 'region' strings to find hidden dependencies.
        3. Bubbles up commented entries to the top of their type-block.
        """
        # Separate 'settings' from the rest
        free_entries = []
        graph_entries = []
        
        for entry in self.entries:
            if entry['type'] == 'settings' or entry['type'] == 'source':
                free_entries.append(entry)
            else:
                graph_entries.append(entry)

        # Map Names to Indices (relative to graph_entries)
        name_to_idx = {entry['name']: i for i, entry in enumerate(graph_entries)}
        
        # Build Dependency Graph
        adj = defaultdict(set)
        in_degree = defaultdict(int)
        
        def add_dependency(u, v):
            """u depends on v (v must come before u)"""
            if v != u and u not in adj[v]:
                adj[v].add(u)
                in_degree[u] += 1

        for i, entry in enumerate(graph_entries):
            # AST parsing for direct variable usage
            try:
                tree = ast.parse(entry['code'])
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        ref = node.id
                        if ref in name_to_idx:
                            add_dependency(i, name_to_idx[ref])
            except Exception:
                continue

        # Kahn's Algorithm
        queue = deque()
        for i in range(len(graph_entries)):
            if in_degree[i] == 0:
                queue.append(i)
        
        sorted_indices = []
        while queue:
            u = queue.popleft()
            sorted_indices.append(u)
            
            # Sort neighbors for deterministic output
            for v in sorted(list(adj[u])):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # Cycle/Error handling
        if len(sorted_indices) < len(graph_entries):
            seen = set(sorted_indices)
            for i in range(len(graph_entries)):
                if i not in seen:
                    sorted_indices.append(i)

        # Construct Preliminary List
        new_entries = [graph_entries[i] for i in sorted_indices]
        
        # Post-Processing: Bubble Up Commented Entries
        # Re-map names to NEW indices for fast dependency checking
        new_name_to_idx = {e['name']: i for i, e in enumerate(new_entries)}
        
        def depends(idx_a, idx_b):
            """Returns True if entry at new index A depends on entry at new index B"""
            # Check original graph using names
            name_a = new_entries[idx_a]['name']
            name_b = new_entries[idx_b]['name']
            
            # Map back to original indices to check 'adj'
            orig_a = name_to_idx[name_a]
            orig_b = name_to_idx[name_b]
            
            # Since we only swap adjacent items, we just need to check if A depends on B directly or indirectly.
            # But 'adj' stores direct edges: adj[b] contains a if a depends on b.
            return orig_a in adj[orig_b]

        for i in range(len(new_entries)):
            # Check if this entry has a comment (header)
            if new_entries[i]['code'].strip().startswith("#"):
                
                # Bubble up
                curr = i
                while curr > 0:
                    prev = curr - 1
                    curr_ent = new_entries[curr]
                    prev_ent = new_entries[prev]
                    
                    # Stop if different type
                    if curr_ent['type'] != prev_ent['type']:
                        break
                        
                    # Stop if previous one also has a comment (don't reorder headers)
                    if prev_ent['code'].strip().startswith("#"):
                        break
                        
                    # Stop if dependency exists (Current depends on Previous)
                    if depends(curr, prev):
                        break
                        
                    # SWAP
                    new_entries[prev], new_entries[curr] = new_entries[curr], new_entries[prev]
                    
                    # Update map for next iteration (swapped indices)
                    # (Actually we don't need to update map if we look up by name every time)
                    curr -= 1

        # append remaining
        new_entries.extend(free_entries)
        
        self.entries = new_entries
        return True
    
    def get_script(self, include_run: bool = True, preserve_order: bool = True) -> str:
        """
        Reconstructs the script.
        """
        lines = list(self.imports)
        lines.append("") 
        
        last_type = None

        if preserve_order:
            # INSERTION ORDER
            for entry in self.entries:
                # Add a blank line if switching types
                current_type = entry['type']
                if last_type and current_type != last_type:
                    geo_types = ['surface', 'cell', 'universe', 'lattice']
                    if not (current_type in geo_types and last_type in geo_types):
                        lines.append("") 
                
                lines.append(entry['code'])
                last_type = current_type
        else:
            # TYPE-BASED GROUPING (Legacy/Fallback)
            order = [
                "material", "surface", "cell", "universe", 
                "lattice", "mesh", "source", "tally", "settings" 
            ]
            for section in order:
                section_entries = [e for e in self.entries if e['type'] == section]
                if section_entries:
                    lines.append(f"# === {section.upper()} DEFINITIONS ===")
                    for entry in section_entries:
                        lines.append(entry['code'])
                    lines.append("") 
            
            others = [e for e in self.entries if e['type'] not in order]
            if others:
                lines.append("# === OTHER ===")
                for entry in others:
                    lines.append(entry['code'])

        if include_run:
            lines.append("")
            lines.append("# === RUN ===")
            lines.append("mcdc.run()")
            
        return "\n".join(lines)
    
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
            # Handle Imports 
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                code = get_segment(node)
                if code not in self.imports:
                    self.imports.append(code)
                continue

            # Handle Assignments (m1 = mcdc.Material(...))
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

            # Handle Standalone Expressions (e.g., mcdc.Source(...) without assignment)
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

            # Add everything else as 'other' (comments, math, etc)
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
                self.reorder()
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
                self.reorder()
                return True
        return False
    
    def insert_entity(self, code: str, entity_type: str, name: str, 
                     before: str = None, after: str = None, position: int = None) -> bool:
        """
        Insert a new entity at a specific position in the script.
        """
        if self.has_entity(entity_type, name):
            return False
        
        new_entry = {
            'code': code,
            'type': entity_type,
            'name': name
        }
        
        insert_idx = None
        if position is not None:
            insert_idx = max(0, min(position, len(self.entries)))
        elif before:
            for idx, entry in enumerate(self.entries):
                if entry['name'] == before:
                    insert_idx = idx
                    break
        elif after:
            for idx, entry in enumerate(self.entries):
                if entry['name'] == after:
                    insert_idx = idx + 1
                    break
        
        if insert_idx is not None:
            self.entries.insert(insert_idx, new_entry)
        else:
            self.entries.append(new_entry)
        
        if entity_type not in self.defined:
            self.defined[entity_type] = set()
        self.defined[entity_type].add(name)
        
        self.reorder()
        return True
    
    def find_entity_index(self, entity_type: str, name: str) -> int:
        for idx, entry in enumerate(self.entries):
            if entry['type'] == entity_type and entry['name'] == name:
                return idx
        return -1
    
    def get_code_by_type(self, entity_type: str) -> str:
        lines = [e['code'] for e in self.entries if e['type'] == entity_type]
        if not lines:
            return "No entries defined."
        return "\n".join(lines)

    def reset(self):
        self.__init__()