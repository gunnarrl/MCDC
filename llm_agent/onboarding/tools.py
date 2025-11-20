import json
import numpy as np
import pandas as pd
import re
from langchain.tools import tool
from .script_builder import ScriptBuilder
from typing import Dict, List, Set, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path


ONBOARDING_DIR = Path(__file__).parent

class MaterialCalculator:
    """
    Public-domain nuclear data. Calculates atomic compositions only.
    Loads data from nuclear_data.csv
    SHOULD NOT USE FOR REAL SIMULATIONS - MAY BE INACCURATE OR INCOMPLETE
    """
    
    try:
        # Load data from the CSV file
        _data_df = pd.read_csv(ONBOARDING_DIR / "nuclear_data.csv")
        _data_df.set_index('Isotope', inplace=True)
        
        # Recreate the old dictionary structure for compatibility
        NUCLEAR_DATA = {}
        for elem in _data_df['Element'].unique():
            elem_df = _data_df[_data_df['Element'] == elem]
            NUCLEAR_DATA[elem] = {
                iso: (row['Mass_u'], row['Abundance'])
                for iso, row in elem_df.iterrows()
            }
            
        # Create a set of dominant isotopes for filtering
        DOMINANT_ISOTOPES = set(_data_df[_data_df['Dominant'] == 1].index)

    except FileNotFoundError:
        print("FATAL ERROR: nuclear_data.csv not found.")
    
    try:
        _materials_df = pd.read_csv(ONBOARDING_DIR / "material_properties.csv")
        # Create a searchable dictionary from this dataframe
        COMMON_MATERIALS = {}
        for _, row in _materials_df.iterrows():
            key = row['name']
            COMMON_MATERIALS[key] = {
                'formula': row['formula'],
                'density': row['density']
            }
            # Add all aliases
            if pd.notna(row['aliases']):
                for alias in row['aliases'].split(';'):
                    COMMON_MATERIALS[alias.strip()] = {
                        'formula': row['formula'],
                        'density': row['density']
                    }
    except FileNotFoundError:
        print("ERROR: material_properties.csv not found.")
        COMMON_MATERIALS = {}
    
    AVOGADRO = 6.02214076e23
    
    @staticmethod
    def parse_formula(formula: str) -> Dict[str, float]:
        """Parse chemical formulas with element counts: H2O, UO2, B4C, Fe0.7Cr0.2Ni0.1"""
        parsed = {}
        # Pattern: Element symbol (1-2 letters) followed by optional number
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        for element, count_str in matches:
            count = float(count_str) if count_str else 1.0
            parsed[element] = parsed.get(element, 0) + count
        
        return parsed
    
    @staticmethod
    def calculate_composition(formula: str, density: float, enrichment: float = None) -> Dict[str, float]:
        """
        Calculate atoms/barn-cm for each isotope from chemical formula.
        
        Process:
        1. Parse formula -> element counts
        2. Calculate molecular weight using NIST isotopic masses
        3. molecules/cm³ = (density / mol_weight) × Avogadro
        4. For each element, split into isotopes using natural abundances
        5. For uranium: apply enrichment weight fraction -> atom fraction conversion
        6. Convert atoms/cm³ -> atoms/barn-cm (×1e-24)
        """
        element_counts = MaterialCalculator.parse_formula(formula)
        
        # Calculate molecular weight accounting for isotopic abundances
        molecular_weight = 0.0
        for element, count in element_counts.items():
            if element not in MaterialCalculator.NUCLEAR_DATA:
                raise ValueError(f"Unknown element: {element}")
            
            element_weight = 0.0
            for isotope, (mass, abundance) in MaterialCalculator.NUCLEAR_DATA[element].items():
                element_weight += mass * abundance
            
            molecular_weight += element_weight * count
        
        if molecular_weight == 0:
            raise ValueError("Could not calculate molecular weight")
        
        # Convert density to molecules/cm³
        molecules_per_cm3 = (density / molecular_weight) * MaterialCalculator.AVOGADRO
        
        # Calculate isotopic composition
        composition = {}
        for element, count in element_counts.items():
            if element == 'U' and enrichment is not None:
                # Special uranium enrichment handling
                u235_mass = MaterialCalculator.NUCLEAR_DATA['U']['U235'][0]
                u238_mass = MaterialCalculator.NUCLEAR_DATA['U']['U238'][0]
                
                # Atom fraction of U235 = (W235/M235) / (W235/M235 + W238/M238)
                # where W235 = enrichment, W238 = 1 - enrichment
                atom_frac_u235 = (enrichment/u235_mass) / (enrichment/u235_mass + (1-enrichment)/u238_mass)
                
                # Apply to all uranium isotopes, scaling natural abundances
                for isotope, (mass, abundance) in MaterialCalculator.NUCLEAR_DATA['U'].items():
                    if isotope == 'U235':
                        isotope_abundance = atom_frac_u235
                    elif isotope == 'U238':
                        isotope_abundance = 1.0 - atom_frac_u235
                    else:
                        # U234 and others scaled proportionally
                        isotope_abundance = abundance * (1e-6)  # Very small for simplicity
                    
                    atoms_per_barn_cm = molecules_per_cm3 * count * isotope_abundance * 1e-24
                    composition[isotope] = atoms_per_barn_cm
            else:
                # Normal isotopic splitting by natural abundance
                for isotope, (mass, abundance) in MaterialCalculator.NUCLEAR_DATA[element].items():
                    atoms_per_barn_cm = molecules_per_cm3 * count * abundance * 1e-24
                    if atoms_per_barn_cm > 1e-30:  # Filter out negligible isotopes
                        composition[isotope] = atoms_per_barn_cm
        
        return composition


class CreateMaterialArgs(BaseModel):
    name: str = Field(..., description="Variable name for the material")
    mode: str = Field(..., description="'MG' (Multi-Group), 'CE' (Continuous Energy), or 'formula'")
    properties: str = Field(..., description="JSON string of properties. MG keys: capture, scatter, fission, nu_p, speed (lists). CE: nuclide_composition (dict). Formula: formula (str), density (float), enrichment (float).")
    description: Optional[str] = Field(None, description="Optional comment/description to place above this line")

class CreateSurfaceArgs(BaseModel):
    name: str = Field(..., description="Variable name for the surface")
    type_: str = Field(..., description="Surface type: 'PlaneX', 'PlaneY', 'PlaneZ', 'CylinderZ', 'Sphere', etc.")
    params: str = Field(..., description="JSON string of parameters (e.g. {'x': 5.0} or {'center': [0,0,0], 'radius': 5.0})")
    boundary_condition: str = Field("interface", description="'interface', 'vacuum', or 'reflective'")
    description: Optional[str] = Field(None, description="Optional comment/description")

class CreateGeometryArgs(BaseModel):
    type_: str = Field(..., description="'cell', 'universe', 'lattice', or 'mesh'")
    name: str = Field(..., description="Variable name")
    params: str = Field(..., description="JSON string of parameters specific to the entity type")
    description: Optional[str] = Field(None, description="Optional comment/description")

class CreateSourceArgs(BaseModel):
    name: str = Field("source", description="Variable name (usually just 'source' unless multiple)")
    params: str = Field(..., description="JSON string of parameters: position, direction, energy, time, probability, etc.")
    description: Optional[str] = Field(None, description="Optional comment/description")

class CreateTallyArgs(BaseModel):
    type_: str = Field(..., description="'global', 'surface', 'cell', or 'mesh'")
    name: str = Field(..., description="Variable name")
    params: str = Field(..., description="JSON string. Must include 'scores'. For surface/cell tallies, include 'surface'/'cell' name. For mesh, include 'mesh' name.")
    description: Optional[str] = Field(None, description="Optional comment/description")

class SetSettingsArgs(BaseModel):
    params: str = Field(..., description="JSON string of settings: N_particle, N_batch, rng_seed, etc.")
    description: Optional[str] = Field(None, description="Optional comment/description")

class ManageScriptArgs(BaseModel):
    action: str = Field(..., description="'get' to view script, 'delete' to remove entity")
    entity_type: Optional[str] = Field(None, description="Required for delete: 'material', 'surface', 'cell', etc.")
    name: Optional[str] = Field(None, description="Required for delete: variable name")

class SearchDocsArgs(BaseModel):
    query: str = Field(..., description="Search query for MCDC documentation")

# --- TOOL FACTORY ---

def get_mcdc_tools(builder: ScriptBuilder, retriever: Any):

    def _add_code(code: str, type_: str, name: str, description: Optional[str]):
        """Helper to format code with comments"""
        full_code = f"# {description}\n{code}" if description else code
        builder.add_line(full_code, type_, name)

    @tool(args_schema=CreateMaterialArgs)
    def create_material(name: str, mode: str, properties: str, description: Optional[str] = None) -> str:
        """Create a material (MG, CE, or Formula).
        
        KEYS for 'properties' JSON based on 'mode':
        
        1. mode='MG' (Multi-Group):
           - capture (array): Capture cross-section.
           - scatter (array): Scatter cross-section (G x G).
           - fission (array): Fission cross-section.
           - nu_p (array): Prompt neutrons per fission.
           - nu_d (array): Delayed neutrons per fission.
           - chi_p (array): Prompt fission spectrum.
           - chi_d (array): Delayed fission spectrum.
           - speed (array): Particle speed.
           - decay_rate (array): Precursor decay rate.
           
        2. mode='CE' (Continuous Energy):
           - nuclide_composition (dict): {'isotope': density, ...} e.g. {'U235': 1.0}
           
        3. mode='formula' (Helper):
           - formula (str): Chemical formula (e.g. "H2O", "UO2")
           - density (float): Density in g/cm^3.
           - enrichment (float): Optional U-235 enrichment (0.0 to 1.0).
           """
        if builder.has_entity("material", name): return f"ERROR: Material '{name}' already defined."
        
        try:
            props = json.loads(properties)
            code = ""
            
            if mode.lower() == 'formula':
                formula = props.get('formula', name)
                density = props.get('density')
                enrichment = props.get('enrichment')
                
                # Auto-detect density if missing
                if density is None and formula in MaterialCalculator.COMMON_MATERIALS:
                    density = MaterialCalculator.COMMON_MATERIALS[formula]['density']
                
                if density is None: return "ERROR: Density required for formula mode."
                
                comp = MaterialCalculator.calculate_composition(formula, density, enrichment)
                # Format dict nicely
                comp_str = "{\n" + ",\n".join([f"        '{k}': {v:.6e}" for k,v in comp.items()]) + "\n    }"
                code = f"{name} = mcdc.Material(nuclide_composition={comp_str})"
                
            elif mode.lower() == 'ce':
                comp = props.get('nuclide_composition', {})
                code = f"{name} = mcdc.Material(nuclide_composition={comp})"
                
            elif mode.lower() == 'mg':
                # Build MG arguments
                lines = []
                for key in ['capture', 'scatter', 'fission', 'nu_p', 'nu_d', 'speed', 'decay_rate']:
                    if key in props:
                        val = props[key]
                        # Ensure lists become numpy arrays in string
                        val_str = f"np.array({val})" if isinstance(val, list) else str(val)
                        lines.append(f"    {key}={val_str}")
                code = f"{name} = mcdc.MaterialMG(\n" + ",\n".join(lines) + "\n)"
            
            _add_code(code, "material", name, description)
            return f"Created {mode} material '{name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @tool(args_schema=CreateSurfaceArgs)
    def create_surface(name: str, type_: str, params: str, boundary_condition: str = "interface", description: Optional[str] = None) -> str:
        """Create a geometric surface.
        
        VALID TYPES and their PARAMS (in 'params' JSON):
        - PlaneX: {'x': float}
        - PlaneY: {'y': float}
        - PlaneZ: {'z': float}
        - Plane: {'A': float, 'B': float, 'C': float, 'D': float}
        - CylinderX: {'center': [y, z], 'radius': float}
        - CylinderY: {'center': [x, z], 'radius': float}
        - CylinderZ: {'center': [x, y], 'radius': float}
        - Sphere: {'center': [x, y, z], 'radius': float}
        - Quadric: {'A':.., 'B':.., ... 'J':..}
        
        Boundary Condition (bc): 'interface' (default), 'vacuum', or 'reflective'.
        """
        if builder.has_entity("surface", name): return f"ERROR: Surface '{name}' already defined."
        
        try:
            p = json.loads(params)
            # Build args string
            args = []
            for k, v in p.items():
                val = f"np.array({v})" if isinstance(v, list) else str(v)
                args.append(f"{k}={val}")
            
            if boundary_condition and boundary_condition != "interface":
                args.append(f"boundary_condition='{boundary_condition}'")
                
            code = f"{name} = mcdc.Surface.{type_}({', '.join(args)})"
            _add_code(code, "surface", name, description)
            return f"Created Surface '{name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @tool(args_schema=CreateGeometryArgs)
    def create_geometry(type_: str, name: str, params: str, description: Optional[str] = None) -> str:
        """Create complex geometry entities.
        
        VALID TYPES and PARAMS:
        
        1. type_='cell':
           - region (str): Boolean CSG expression (e.g. "+s1 & -s2").
           - fill (str): Material name, Universe name, or Lattice name.
           - translation (list): [x, y, z] (Optional).
           - rotation (list): [x, y, z] (Optional).
           
        2. type_='universe':
           - cells (list): List of cell objects/names.
           - root (bool): True if this is the root universe.
           
        3. type_='lattice':
           - x, y, z (tuple): Grid def (start, end, num_cells).
           - universes (list): 2D/3D list of Universe objects.
           
        4. type_='mesh' (or 'mesh_structured' / 'mesh_uniform'):
           - x, y, z (tuple/array): 
             For Uniform: Use list of 3 floats [start, end, N].
             For Structured: Use string "np.linspace(...)" or list of grid points.
        """
        if builder.has_entity(type_, name): return f"ERROR: {type_} '{name}' already defined."
        
        try:
            p = json.loads(params)
            args = []
            
            # Detect Mesh Type
            is_mesh = "mesh" in type_.lower()
            # Allow explicit override via type name (e.g. "mesh_structured")
            force_structured = "structured" in type_.lower()
            
            for k, v in p.items():
                val = str(v)
                
                # --- MESH HANDLING ---
                if is_mesh and (k in ['x', 'y', 'z', 't']):
                    # Case A: User provided a numpy string (e.g. "np.linspace(0,10,5)")
                    if "np." in val or "numpy" in val:
                        force_structured = True
                        # val is already the correct string
                        
                    # Case B: User provided a list
                    elif isinstance(v, list):
                        # If explicit structured requested OR list length is not 3
                        # (Uniform meshes MUST be length 3: start, stop, N)
                        if force_structured or len(v) != 3:
                            force_structured = True
                            val = f"np.array({v})"
                        else:
                            # It is a uniform mesh tuple (0.0, 10.0, 5)
                            val = f"({v[0]}, {v[1]}, {v[2]})"

                # Convert lists to numpy arrays for specific non-mesh fields
                elif isinstance(v, list) and k not in ['fill', 'region', 'cells', 'universes']: 
                     val = f"np.array({v})"
                
                # Clean up reference lists (remove quotes around variable names)
                if k in ['cells', 'universes'] and isinstance(v, list):
                    val = str(v).replace("'", "").replace('"', "")

                args.append(f"{k}={val}")
            
            # Determine class name
            class_map = {
                "cell": "Cell", 
                "universe": "Universe", 
                "lattice": "Lattice", 
                "mesh": "MeshUniform" # Default
            }
            
            # Clean type_ to basic key for map lookup
            base_type = "mesh" if "mesh" in type_.lower() else type_.lower()
            cls_name = class_map.get(base_type, "MeshUniform") # Fallback to MeshUniform
            
            if is_mesh and force_structured:
                cls_name = "MeshStructured"

            code = f"{name} = mcdc.{cls_name}({', '.join(args)})"
            _add_code(code, type_, name, description)
            return f"Created {cls_name} '{name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @tool(args_schema=CreateSourceArgs)
    def create_source(name: str, params: str, description: Optional[str] = None) -> str:
        """Create a particle source.
        
        VALID KEYS for 'params':
        - position (list): [x,y,z] point.
        - x, y, z (list): [min, max] distribution ranges.
        - direction (list): [u,v,w] vector.
        - white_direction (list): [nx,ny,nz] surface normal.
        - isotropic (bool): Default True.
        - energy (float/array): Energy values (eV).
        - energy_group (int): MG group index.
        - time (float/list): Time range.
        - probability (float): Source weight.
        """
        try:
            p = json.loads(params)
            args = []
            for k, v in p.items():
                val = f"np.array({v})" if isinstance(v, list) else str(v)
                args.append(f"{k}={val}")
            
            code = f"{name} = mcdc.Source({', '.join(args)})"
            # Source is unique in builder as it doesn't block duplicates usually, but let's track it
            unique_name = name if not builder.has_entity("source", name) else f"{name}_{len(builder.defined['source'])}"
            
            _add_code(code, "source", unique_name, description)
            return f"Created Source '{unique_name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @tool(args_schema=CreateTallyArgs)
    def create_tally(type_: str, name: str, params: str, description: Optional[str] = None) -> str:
        """Create a tally.
        
        VALID TYPES and REQUIRED PARAMS:
        - global: params={'scores': ['flux', ...]}
        - surface: params={'scores': [...], 'surface': 'surface_name'}
        - cell: params={'scores': [...], 'cell': 'cell_name'}
        - mesh: params={'scores': [...], 'mesh': 'mesh_name'}
        
        Common Params:
        - scores (list): ['flux', 'absorption', 'fission', 'net-current', ...]
        - mu, energy, time (array/string): e.g., "np.linspace(0,1,10)" OR [0.0, 1.0, 2.0]
        """
        if builder.has_entity("tally", name): return f"ERROR: Tally '{name}' already defined."
        
        try:
            p = json.loads(params)
            args = []
            for k, v in p.items():
                val = str(v)
                
                # Handle numeric arrays (time, energy, mu)
                # If it's a list, wrap in np.array. If it's a string (np.linspace), leave it.
                if k in ['time', 'energy', 'mu'] and isinstance(v, list):
                    val = f"np.array({v})"
                
                # Handle variable references (mesh, surface, cell)
                args.append(f"{k}={val}")
                
            class_map = {
                "global": "TallyGlobal",
                "surface": "TallySurface",
                "cell": "TallyCell",
                "mesh": "TallyMesh"
            }
            cls_name = class_map.get(type_.lower())
            if not cls_name: return f"ERROR: Unknown tally type {type_}"
            
            code = f"{name} = mcdc.{cls_name}({', '.join(args)})"
            _add_code(code, "tally", name, description)
            return f"Created {type_} tally '{name}'"
        except Exception as e:
            return f"ERROR: {e}"

    @tool(args_schema=SetSettingsArgs)
    def set_settings(params: str, description: Optional[str] = None) -> str:
        """Set global simulation settings.
        
        Valid keys for 'params' JSON:
        - N_particle (int): Number of histories to run.
        - N_batch (int): Number of batches.
        - rng_seed (int): Random number seed.
        - time_boundary (float): Time edge after which particles are killed.
        - progress_bar (bool): Display progress bar (default True).
        - output_name (str): Output filename (default "output.h5").
        - save_input_deck (bool): Save input to output file (default False).
        - k_eff (bool): Run k-eigenvalue problem (requires Eigenmode card).
        - source_file (str): Path to source file.
        - IC_file (str): Path to initial condition file.
        - active_bank_buff (int): Size of active bank buffer.
        - census_bank_buff (int): Size of census bank buffer (multiples of N_particle).
        - source_bank_buff (int): Size of source bank buffer.
        - future_bank_buff (int): Size of future bank buffer.
        """
        try:
            p = json.loads(params)
            lines = []
            if description: lines.append(f"# {description}")
            
            for k, v in p.items():
                lines.append(f"mcdc.settings.{k} = {v}")
            
            code = "\n".join(lines)
            builder.add_line(code, "settings", "global_settings")
            return f"Updated settings: {', '.join(p.keys())}"
        except Exception as e:
            return f"ERROR: {e}"

    @tool(args_schema=ManageScriptArgs)
    def manage_script(action: str, entity_type: Optional[str] = None, name: Optional[str] = None) -> str:
        """Get current script state OR delete an entity."""
        if action == "delete":
            if not entity_type or not name: return "ERROR: Delete requires entity_type and name."
            if builder.delete_entity(entity_type, name):
                return f"Deleted {entity_type} '{name}'"
            return f"ERROR: {entity_type} '{name}' not found."
            
        # Default to "get"
        summary = {k: list(v) for k, v in builder.defined.items()}
        return f"CURRENT SCRIPT:\n{builder.get_script()}\n\nDEFINED:\n{json.dumps(summary, indent=2)}"

    @tool(args_schema=SearchDocsArgs)
    def search_docs(query: str) -> str:
        """Search MCDC documentation."""
        try:
            docs = retriever.invoke(query)
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown").replace("llm_agent/corpus/", "")
                formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted) if formatted else "No results found."
        except Exception as e:
            return f"Error searching docs: {e}"

    return [
        create_material,
        create_surface,
        create_geometry,
        create_source,
        create_tally,
        set_settings,
        manage_script,
        search_docs
    ]