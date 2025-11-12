import json
import numpy as np
import pandas as pd
import re
from langchain.tools import tool
from onboarding.script_builder import ScriptBuilder
from typing import Dict, List, Set

class MaterialCalculator:
    """
    Public-domain nuclear data. Calculates atomic compositions only.
    Loads data from nuclear_data.csv
    SHOULD NOT USE FOR REAL SIMULATIONS - MAY BE INACCURATE OR INCOMPLETE
    """
    
    try:
        # Load data from the CSV file
        _data_df = pd.read_csv("onboarding/nuclear_data.csv")
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
    
    COMMON_MATERIALS = {
        'water': {'formula': 'H2O', 'density': 1.0, 'aliases': ['h2o', 'light water', 'lw']},
        'heavy_water': {'formula': 'D2O', 'density': 1.1, 'aliases': ['d2o', 'heavy water', 'hw']},
        'uranium_dioxide': {'formula': 'UO2', 'density': 10.5, 'aliases': ['uo2', 'uranium oxide', 'fuel']},
        'zirconium': {'formula': 'Zr', 'density': 6.52, 'aliases': ['zr', 'zirc', 'zircaloy']},
        'boron_carbide': {'formula': 'B4C', 'density': 2.52, 'aliases': ['b4c', 'boron carbide', 'control rod']},
        'graphite': {'formula': 'C', 'density': 1.7, 'aliases': ['c', 'graphite', 'moderator']},
        'sodium': {'formula': 'Na', 'density': 0.97, 'aliases': ['na', 'sodium', 'coolant']},
        'lead': {'formula': 'Pb', 'density': 11.34, 'aliases': ['pb', 'lead', 'shielding']},
        'iron': {'formula': 'Fe', 'density': 7.87, 'aliases': ['fe', 'iron', 'steel']},
        'stainless_steel': {'formula': 'Fe0.7Cr0.2Ni0.1', 'density': 8.0, 'aliases': ['ss', 'stainless steel', 'steel']},
    }
    
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

# TOOL FUNCTION

def get_mcdc_tools(builder: ScriptBuilder):
    """
    Factory function that creates tools bound to a specific ScriptBuilder instance.
    """
    
    @tool
    def create_material_from_formula(
        name: str,
        formula: str = None,
        density: float = None,
        mode: str = "CE",
        capture: str = None,
        scatter: str = None,
        fission: str = None,
        nu_p: str = None,
        enrichment: float = None
    ) -> str:
        """
        Create material from chemical formula (MG or CE mode).
        
        MG Mode: Self-contained, uses macroscopic cross-sections
        CE Mode (default): Requires MCDC_XSLIB environment variable. Calculates a simplified
                 nuclide composition using only dominant isotopes.
        
        Examples:
        - MG water: create_material_from_formula("water", "H2O", 1.0, mode="MG", 
                                                capture="[0.02]", scatter="[[0.08]]")
        - CE water: create_material_from_formula("water", "H2O", 1.0, mode="CE")
        - CE 3% fuel: create_material_from_formula("fuel", "UO2", 10.5, mode="CE", enrichment=0.03)
        """
        if builder.has_entity("material", name):
            return f"ERROR: Material '{name}' already defined."
        
        formula_clean = formula.strip() if formula else ""
        
        # Auto-detect common materials
        detected_formula = None
        if not formula_clean:
            for mat_key, info in MaterialCalculator.COMMON_MATERIALS.items():
                if name.lower() in info['aliases'] or info['formula'].lower() == name.lower():
                    formula_clean = info['formula']
                    if density is None:
                        density = info['density']
                    detected_formula = info['formula']
                    break
        
        # If still no formula, try to use the name as the formula
        if not formula_clean:
            formula_clean = name
            
        # If density is still unknown, error out (it's required for CE)
        if mode.upper() == "CE" and density is None:
             # Try one last time to get density from common materials
            if formula_clean.upper() in [v['formula'] for v in MaterialCalculator.COMMON_MATERIALS.values()]:
                 for k, v in MaterialCalculator.COMMON_MATERIALS.items():
                     if v['formula'] == formula_clean.upper():
                         density = v['density']
                         break
            else:
                return f"ERROR: Density is required for CE material '{name}' and was not provided or found."

        try:
            if mode.upper() == "CE":

                # 1. Calculate the full, precise composition
                full_composition = MaterialCalculator.calculate_composition(
                    formula_clean, density, enrichment
                )
                
                # 2. Filter for dominant isotopes
                filtered_composition = {}
                for isotope, value in full_composition.items():
                    # Uranium is a special case: always keep U235 and U238
                    if isotope.startswith("U"):
                        if enrichment is not None and isotope in ["U235", "U238"]:
                             filtered_composition[isotope] = value
                        elif enrichment is None and isotope in MaterialCalculator.DOMINANT_ISOTOPES:
                             filtered_composition[isotope] = value # Keep natural U
                    # For all other elements, check the dominant list
                    elif isotope in MaterialCalculator.DOMINANT_ISOTOPES:
                        filtered_composition[isotope] = value

                # 3. Build the code string
                comp_lines = [
                    f"        '{iso}': {val:.16e}," 
                    for iso, val in filtered_composition.items()
                ]
                comp_str = "\n".join(comp_lines)
                
                code = f"""{name} = mcdc.Material(
                    nuclide_composition={{
                {comp_str}
                    }}
                )
                # NOTE: CE mode requires MCDC_XSLIB environment variable
                # export MCDC_XSLIB="/path/to/your/nuclear/data"
                """
                
                builder.add_line(code, "material", name)
                enrichment_str = f" (enriched to {enrichment*100:.1f}% U-235)" if enrichment else ""
                return f"✓ Created CE material '{name}': {formula_clean} at {density} g/cm³{enrichment_str}"
            
            else:
                # Multi-Group mode - use provided cross-sections
                code_parts = [f"{name} = mcdc.MaterialMG("]
                
                # Use defaults or provided values
                cap = capture if capture else "[0.1]"  # Default to some absorption
                code_parts.append(f"    capture=np.array({cap})")
                
                if scatter:
                    code_parts.append(f",\n    scatter=np.array({scatter})")
                if fission:
                    code_parts.append(f",\n    fission=np.array({fission})")
                    if nu_p:
                        code_parts.append(f",\n    nu_p=np.array({nu_p})")
                
                code_parts.append("\n)")
                code = "".join(code_parts)
                
                builder.add_line(code, "material", name)
                formula_str = f"{formula_clean} at {density} g/cm³" if formula_clean else ""
                return f"✓ Created MG material '{name}': {formula_str} (no external data needed)"
                
        except Exception as e:
            return f"ERROR: {str(e)}"
        
    @tool
    def set_material_mg(
        name: str, 
        capture: str, 
        scatter: str = None, 
        fission: str = None,
        nu_p: str = None,
        speed: str = None
    ) -> str:
        """
        Define a multi-group (MG) material with cross-sections.
        
        Parameters are numpy array strings:
        - capture: "[0.5]" for 1-group, "[0.5, 0.3]" for 2-group
        - scatter: "[[0.9]]" for 1-group, "[[0.8, 0.1], [0.05, 0.85]]" for 2-group
        - fission: "[0.1]" (optional)
        - nu_p: "[2.5]" (optional, neutrons per fission)
        - speed: "[200000.0]" (optional, cm/s)
        
        Example: set_material_mg("fuel", "[0.45]", "[[0.0]]", "[0.55]", "[2.5]")
        """
        if builder.has_entity("material", name):
            return f"ERROR: Material '{name}' already defined."
        
        try:
            # Parse string inputs to numpy arrays
            cap_arr = eval(f"np.array({capture})")
            
            # Build the code string
            code_parts = [f"{name} = mcdc.MaterialMG("]
            code_parts.append(f"    capture=np.array({capture})")
            
            if scatter:
                code_parts.append(f",\n    scatter=np.array({scatter})")
            if fission:
                code_parts.append(f",\n    fission=np.array({fission})")
            if nu_p:
                code_parts.append(f",\n    nu_p=np.array({nu_p})")
            if speed:
                code_parts.append(f",\n    speed=np.array({speed})")
            
            code_parts.append("\n)")
            code = "".join(code_parts)
            
            builder.add_line(code, "material", name)
            return f"✓ Defined MG material: {name}"
            
        except Exception as e:
            return f"ERROR: Failed to parse parameters: {str(e)}"
    
    @tool
    def set_material_ce(name: str, nuclide_composition: str) -> str:
        """
        Define a continuous-energy (CE) material with nuclide composition.
        
        Parameters:
        - name: Variable name for the material
        - nuclide_composition: Dictionary string, e.g. "{'U235': 0.0005, 'U238': 0.022, 'O16': 0.046}"
        
        Example: set_material_ce("fuel", "{'U235': 0.0005, 'U238': 0.022}")
        """
        if builder.has_entity("material", name):
            return f"ERROR: Material '{name}' already defined."
        
        try:
            # Validate it's a dict-like string
            comp_dict = eval(nuclide_composition)
            if not isinstance(comp_dict, dict):
                return "ERROR: nuclide_composition must be a dictionary"
            
            code = f"{name} = mcdc.Material(\n    nuclide_composition={nuclide_composition}\n)"
            builder.add_line(code, "material", name)
            return f"✓ Defined CE material: {name}"
            
        except Exception as e:
            return f"ERROR: Failed to parse composition: {str(e)}"
    
    @tool
    def create_surface(
        name: str, 
        surface_type: str, 
        params: str,
        boundary_condition: str = None
    ) -> str:
        """
        Create a geometric surface.
        
        Parameters:
        - name: Variable name for the surface
        - surface_type: One of: PlaneX, PlaneY, PlaneZ, Sphere, CylinderX, CylinderY, CylinderZ
        - params: Parameter string depending on type:
            * PlaneX/Y/Z: "x=0.0" or "y=5.0" or "z=-2.0"
            * Sphere: "center=[0.0, 0.0, 0.0], radius=1.5"
            * CylinderX/Y/Z: "center=[0.0, 0.0], radius=1.0"
        - boundary_condition: Optional, one of: "vacuum", "reflective", "interface" (default)
        
        Example: create_surface("s1", "PlaneX", "x=0.0", "vacuum")
        """
        if builder.has_entity("surface", name):
            return f"ERROR: Surface '{name}' already defined."
        
        # Validate surface type
        valid_types = ["PlaneX", "PlaneY", "PlaneZ", "Sphere", 
                       "CylinderX", "CylinderY", "CylinderZ"]
        if surface_type not in valid_types:
            return f"ERROR: Invalid surface_type. Must be one of: {valid_types}"
        
        try:
            # Build the code
            bc_str = f', boundary_condition="{boundary_condition}"' if boundary_condition else ""
            code = f"{name} = mcdc.Surface.{surface_type}({params}{bc_str})"
            
            builder.add_line(code, "surface", name)
            return f"✓ Defined surface: {name} ({surface_type})"
            
        except Exception as e:
            return f"ERROR: Failed to create surface: {str(e)}"
    
    @tool
    def create_cell(name: str, region: str, fill: str) -> str:
        """
        Create a cell with a region and fill.
        
        Parameters:
        - name: Variable name for the cell (can be "" for anonymous cells)
        - region: Boolean expression of surfaces, e.g. "+s1 & -s2" or "-sphere"
        - fill: Name of material or universe to fill the cell
        
        Example: create_cell("fuel_cell", "+s1 & -s2 & -cy", "fuel")
        """

        # Check fill exists
        if not builder.has_entity("material", fill):
            # Could also be a universe, but we're not tracking those yet
            return f"ERROR: Fill '{fill}' not defined. Define the material first."
        
        try:
            # If name is provided, assign to variable
            if name:
                code = f"{name} = mcdc.Cell(region={region}, fill={fill})"
                builder.add_line(code, "cell", name)
                return f"✓ Defined cell: {name}"
            else:
                # Anonymous cell
                code = f"mcdc.Cell(region={region}, fill={fill})"
                builder.add_line(code, "cell", f"_anon_cell_{len(builder.defined['cell'])}")
                return f"✓ Defined anonymous cell with fill={fill}"
                
        except Exception as e:
            return f"ERROR: Failed to create cell: {str(e)}"
    
    @tool
    def create_source(
        x: str = None,
        y: str = None,
        z: str = None,
        energy: str = None,
        isotropic: bool = True,
        energy_group: str = None
    ) -> str:
        """
        Create a particle source.
        
        Parameters:
        - x: x-range as "[xmin, xmax]" or "xval" for point
        - y: y-range as "[ymin, ymax]" or "yval" for point
        - z: z-range as "[zmin, zmax]" or "zval" for point
        - energy: Energy in eV (for CE) as float string, e.g. "1e6"
        - energy_group: Group index for MG, e.g. "0"
        - isotropic: True for isotropic direction, False for beam
        
        Example: create_source("[0.0, 10.0]", "[0.0, 10.0]", "[0.0, 10.0]", energy_group="0")
        """
        try:
            code_parts = ["mcdc.Source("]
            params = []
            
            if x:
                params.append(f"x={x}")
            if y:
                params.append(f"y={y}")
            if z:
                params.append(f"z={z}")
            if energy:
                params.append(f"energy={energy}")
            if energy_group is not None:
                params.append(f"energy_group={energy_group}")
            
            params.append(f"isotropic={isotropic}")
            
            code_parts.append(", ".join(params))
            code_parts.append(")")
            code = "".join(code_parts)
            
            builder.add_line(code, "source", f"_source_{len(builder.defined['source'])}")
            return f"✓ Defined source"
            
        except Exception as e:
            return f"ERROR: Failed to create source: {str(e)}"
    
    @tool
    def create_tally_mesh(
        mesh_type: str,
        mesh_params: str,
        scores: str
    ) -> str:
        """
        Create a mesh tally.
        
        Parameters:
        - mesh_type: "MeshUniform" or "MeshStructured"
        - mesh_params: Parameters for mesh, e.g. "x=(0.0, 10.0, 100)" for uniform
        - scores: List of scores as string, e.g. "['flux', 'fission']"
        
        Example: create_tally_mesh("MeshUniform", "x=(0.0, 10.0, 100)", "['flux']")
        NEED TO ADD OTHER FORMS OF TALLY
        """
        try:
            mesh_code = f"mesh = mcdc.{mesh_type}({mesh_params})"
            tally_code = f"mcdc.TallyMesh(mesh=mesh, scores={scores})"
            
            builder.add_line(mesh_code, "tally", f"_mesh_{len(builder.defined['tally'])}")
            builder.add_line(tally_code, "tally", f"_tally_{len(builder.defined['tally'])}")
            return f"✓ Defined mesh tally"
            
        except Exception as e:
            return f"ERROR: Failed to create tally: {str(e)}"
    
    @tool
    def set_settings(n_particle: int, n_batch: int) -> str:
        """
        Set simulation settings.
        
        Parameters:
        - n_particle: Number of particles per batch
        - n_batch: Number of batches
        
        Example: set_settings(1000, 10)
        NEED TO COMPLETE SETTING LIST
        """
        try:
            code = f"mcdc.settings.N_particle = {n_particle}\nmcdc.settings.N_batch = {n_batch}"
            builder.add_line(code, "settings", "_settings")
            return f"✓ Set N_particle={n_particle}, N_batch={n_batch}"
            
        except Exception as e:
            return f"ERROR: Failed to set settings: {str(e)}"
    
    @tool
    def search_docs(query: str) -> str:
        """
        Search MCDC documentation for examples and API details.
        
        Use this tool to find:
        - How to create specific materials, surfaces, cells
        - Example code snippets
        - Parameter requirements
        
        Always search docs before generating code!
        """
        # FIX: Can't access retriever from here in closure
        # This is a limitation - need to pass retriever too or handle differently
        return "Doc search not available in tool context. Ask the tutor for examples."
    
    @tool
    def get_current_script() -> str:
        """
        Get the current state of the script being built.
        
        Returns:
        - Complete script text
        - Summary of defined entities
        
        Use this to check what's already defined before creating new entities.
        """
        summary = {
            "materials": list(builder.defined["material"]),
            "surfaces": list(builder.defined["surface"]),
            "cells": list(builder.defined["cell"]),
            "sources": list(builder.defined["source"]),
            "tallies": list(builder.defined["tally"]),
            "settings_configured": bool(builder.defined["settings"])
        }
        
        script = builder.get_script()
        return f"""
CURRENT SCRIPT:
{script}

DEFINED ENTITIES:
{json.dumps(summary, indent=2)}
"""
    
    # Return all tools as a list
    return [
        create_material_from_formula,
        set_material_mg,
        set_material_ce,
        create_surface,
        create_cell,
        create_source,
        create_tally_mesh,
        set_settings,
        get_current_script,
        # search_docs  # needs retriever access
    ]