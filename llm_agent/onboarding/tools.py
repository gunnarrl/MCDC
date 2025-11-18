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


# PYDANTIC MODELS

class CreateMaterialFormulaArgs(BaseModel):
    name: str = Field(..., description="Variable name for the material")
    formula: Optional[str] = Field(None, description="Chemical formula (e.g., 'H2O', 'UO2'). If None, tries to use 'name'.")
    density: Optional[float] = Field(None, description="Density in g/cm³. Required for CE mode. Optional for MG.")
    mode: str = Field("MG", description="Mode, 'MG' (default) or 'CE'.")
    capture: Optional[str] = Field(None, description="MG capture cross-section, e.g., '[0.02]'.")
    scatter: Optional[str] = Field(None, description="MG scatter cross-section, e.g., '[[0.08]]'.")
    fission: Optional[str] = Field(None, description="MG fission cross-section, e.g., '[0.1]'.")
    nu_p: Optional[str] = Field(None, description="MG neutrons per fission, e.g., '[2.5]'.")
    enrichment: Optional[float] = Field(None, description="U-235 enrichment fraction (e.g., 0.03 for 3%).")

class SetMaterialMGArgs(BaseModel):
    name: str = Field(..., description="Variable name for the MG material")
    capture: str = Field(..., description="Numpy array string for capture cross-section, e.g., '[0.5]'.")
    scatter: Optional[str] = Field(None, description="Numpy array string for scatter cross-section, e.g., '[[0.9]]'.")
    fission: Optional[str] = Field(None, description="Numpy array string for fission cross-section, e.g., '[0.1]'.")
    nu_p: Optional[str] = Field(None, description="Numpy array string for neutrons per fission, e.g., '[2.5]'.")
    speed: Optional[str] = Field(None, description="Numpy array string for particle speed, e.g., '[200000.0]'.")

class SetMaterialCEArgs(BaseModel):
    name: str = Field(..., description="Variable name for the CE material")
    nuclide_composition: str = Field(..., description="Dictionary string of nuclide composition, e.g., \"{'U235': 0.0005, 'U238': 0.022}\".")

class CreateSurfaceArgs(BaseModel):
    name: str = Field(..., description="Variable name for the surface")
    surface_type: str = Field(..., description="One of: PlaneX, PlaneY, PlaneZ, Sphere, CylinderX, CylinderY, CylinderZ")
    params: str = Field(..., description="Parameter string, e.g., 'x=0.0' or 'center=[0.0, 0.0], radius=1.5'")
    boundary_condition: Optional[str] = Field(None, description="Optional: 'vacuum', 'reflective', or 'interface' (default)")

class CreateCellArgs(BaseModel):
    name: str = Field(..., description="Variable name for the cell (can be '' for anonymous cells)")
    region: str = Field(..., description="Boolean expression of surfaces, e.g., '+s1 & -s2' or '-sphere'")
    fill: str = Field(..., description="Name of the material or universe to fill the cell")

class CreateSourceArgs(BaseModel):
    x: Optional[str] = Field(None, description="x-range as '[xmin, xmax]' or 'xval' for point")
    y: Optional[str] = Field(None, description="y-range as '[ymin, ymax]' or 'yval' for point")
    z: Optional[str] = Field(None, description="z-range as '[zmin, zmax]' or 'zval' for point")
    energy: Optional[str] = Field(None, description="Energy in eV (for CE) as float string, e.g., '1e6'")
    energy_group: Optional[str] = Field(None, description="Group index for MG, e.g., '0'")
    isotropic: bool = Field(True, description="True for isotropic direction (default), False for beam")

class CreateTallyMeshArgs(BaseModel):
    mesh_type: str = Field(..., description="Type of mesh, e.g., 'MeshUniform' or 'MeshStructured'")
    mesh_params: str = Field(..., description="Parameters for mesh, e.g., 'x=(0.0, 10.0, 100)'")
    scores: str = Field(..., description="List of scores as string, e.g., \"['flux', 'fission']\"")

class CreateTallySurfaceArgs(BaseModel):
    surface: str = Field(..., description="Name of the surface to tally on")
    scores: str = Field(..., description="List of scores, e.g. \"['net-current', 'flux']\"")

class SetSettingsArgs(BaseModel):
    n_particle: int = Field(..., description="Number of particles per batch")
    n_batch: int = Field(..., description="Number of batches")

class SearchDocsArgs(BaseModel):
    query: str = Field(..., description="Search query for MCDC documentation")

class DeleteEntityArgs(BaseModel):
    entity_type: str = Field(..., description="Type: 'material', 'surface', 'cell', 'source', 'tally'")
    name: str = Field(..., description="The variable name to delete, e.g., 'm1'")



# TOOL FACTORY 

def get_mcdc_tools(builder: ScriptBuilder, retriever: Any):
    """
    Factory function that creates tools bound to a specific ScriptBuilder instance
    and a document retriever.
    """
    
    @tool(args_schema=CreateMaterialFormulaArgs)
    def create_material_from_formula(
        name: str,
        formula: Optional[str] = None,
        density: Optional[float] = None,
        mode: str = "MG",
        capture: Optional[str] = None,
        scatter: Optional[str] = None,
        fission: Optional[str] = None,
        nu_p: Optional[str] = None,
        enrichment: Optional[float] = None
    ) -> str:
        """
        Create material from chemical formula (MG or CE mode).
        MG is default. If MG, captures/scatter must be provided.
        """
        try: 
            if builder.has_entity("material", name):
                return f"ERROR: Material '{name}' already defined."
            
            formula_clean = formula.strip() if formula else ""
            local_density = density 
            
            # Auto-detect common materials
            detected_formula = None
            if not formula_clean:
                for mat_key, info in MaterialCalculator.COMMON_MATERIALS.items():
                    if name.lower() in info['aliases'] or info['formula'].lower() == name.lower():
                        formula_clean = info['formula']
                        if local_density is None:
                            local_density = info['density']
                        detected_formula = info['formula']
                        break
            
            if not formula_clean:
                formula_clean = name
                
            # Only enforce density requirement for CE mode
            if mode.upper() == "CE" and local_density is None:
                if formula_clean.upper() in [v['formula'] for v in MaterialCalculator.COMMON_MATERIALS.values()]:
                    for k, v in MaterialCalculator.COMMON_MATERIALS.items():
                        if v['formula'] == formula_clean.upper():
                            local_density = v['density']
                            break
                else:
                    return f"ERROR: Density is required for CE material '{name}' and was not provided or found."

            if mode.upper() == "CE":
                # Calculate the full, precise composition
                full_composition = MaterialCalculator.calculate_composition(
                    formula_clean, local_density, enrichment
                )
                
                # Filter for dominant isotopes
                filtered_composition = {}
                for isotope, value in full_composition.items():
                    if isotope.startswith("U"):
                        if enrichment is not None and isotope in ["U235", "U238"]:
                            filtered_composition[isotope] = value
                        elif enrichment is None and isotope in MaterialCalculator.DOMINANT_ISOTOPES:
                            filtered_composition[isotope] = value 
                    elif isotope in MaterialCalculator.DOMINANT_ISOTOPES:
                        filtered_composition[isotope] = value

                # Build the code string
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
                return f"Created CE material '{name}': {formula_clean} at {local_density} g/cm³{enrichment_str}"
            
            else:
                # Multi-Group mode - same as set_material_mg
                code_parts = [f"{name} = mcdc.MaterialMG("]
                
                # Use defaults if not provided
                cap = capture if capture else "[0.1]" 
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
                return f"Created MG material '{name}'"
        except Exception as e:
            return f"ERROR: Could not create material. Please check your inputs and try again."

            
        
    @tool(args_schema=SetMaterialMGArgs)
    def set_material_mg(
        name: str, 
        capture: str, 
        scatter: Optional[str] = None, 
        fission: Optional[str] = None,
        nu_p: Optional[str] = None,
        speed: Optional[str] = None
    ) -> str:
        """
        Define a multi-group (MG) material with cross-sections
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
            return f"Defined MG material: {name}"
            
        except Exception as e:
            return f"ERROR: Failed to parse parameters: {str(e)}"
    
    @tool(args_schema=SetMaterialCEArgs)
    def set_material_ce(name: str, nuclide_composition: str) -> str:
        """
        Define a continuous-energy (CE) material with nuclide composition
        """
        if builder.has_entity("material", name):
            return f"ERROR: Material '{name}' already defined."
        
        try:
            comp_dict = eval(nuclide_composition)
            if not isinstance(comp_dict, dict):
                return "ERROR: nuclide_composition must be a dictionary"
            
            code = f"{name} = mcdc.Material(\n    nuclide_composition={nuclide_composition}\n)"
            builder.add_line(code, "material", name)
            return f"Defined CE material: {name}"
            
        except Exception as e:
            return f"ERROR: Failed to parse composition: {str(e)}"
    
    @tool(args_schema=CreateSurfaceArgs)
    def create_surface(
        name: str, 
        surface_type: str, 
        params: str,
        boundary_condition: Optional[str] = None
    ) -> str:
        """
        Create a geometric surface
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
            return f"Defined surface: {name} ({surface_type})"
            
        except Exception as e:
            return f"ERROR: Failed to create surface: {str(e)}"
    
    @tool(args_schema=CreateCellArgs)
    def create_cell(name: str, region: str, fill: str) -> str:
        """
        Create a cell with a region and fill.
        ...
        """

        # Check fill exists
        if not builder.has_entity("material", fill):
            # Could also be a universe, but not yet implemented
            return f"ERROR: Fill '{fill}' not defined. Define the material first."
        
        try:
            # If name is provided, assign to variable
            if name:
                code = f"{name} = mcdc.Cell(region={region}, fill={fill})"
                builder.add_line(code, "cell", name)
                return f"Defined cell: {name}"
            else:
                # Anonymous cell
                code = f"mcdc.Cell(region={region}, fill={fill})"
                builder.add_line(code, "cell", f"_anon_cell_{len(builder.defined['cell'])}")
                return f"Defined anonymous cell with fill={fill}"
                
        except Exception as e:
            return f"ERROR: Failed to create cell: {str(e)}"
    
    @tool(args_schema=CreateSourceArgs)
    def create_source(
        x: Optional[str] = None,
        y: Optional[str] = None,
        z: Optional[str] = None,
        energy: Optional[str] = None,
        energy_group: Optional[str] = None,
        isotropic: bool = True
    ) -> str:
        """
        Create a particle source
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
            return f"Defined source"
            
        except Exception as e:
            return f"ERROR: Failed to create source: {str(e)}"
    
    @tool(args_schema=CreateTallyMeshArgs)
    def create_tally_mesh(
        mesh_type: str,
        mesh_params: str,
        scores: str
    ) -> str:
        """
        Create a mesh tally
        """
        try:
            mesh_code = f"mesh = mcdc.{mesh_type}({mesh_params})"
            tally_code = f"mcdc.TallyMesh(mesh=mesh, scores={scores})"
            
            builder.add_line(mesh_code, "tally", f"_mesh_{len(builder.defined['tally'])}")
            builder.add_line(tally_code, "tally", f"_tally_{len(builder.defined['tally'])}")
            return f"Defined mesh tally"
            
        except Exception as e:
            return f"ERROR: Failed to create tally: {str(e)}"
    
    @tool(args_schema=SetSettingsArgs)
    def set_settings(n_particle: int, n_batch: int) -> str:
        """
        Set simulation settings
        """
        try:
            code = f"mcdc.settings.N_particle = {n_particle}\nmcdc.settings.N_batch = {n_batch}"
            builder.add_line(code, "settings", "_settings")
            return f"Set N_particle={n_particle}, N_batch={n_batch}"
            
        except Exception as e:
            return f"ERROR: Failed to set settings: {str(e)}"
    
    @tool(args_schema=CreateTallySurfaceArgs)
    def create_tally_surface(surface: str, scores: str) -> str:
        """
        Create a surface tally.
        """
        try:
            # Check surface exists
            if not builder.has_entity("surface", surface):
                return f"ERROR: Surface '{surface}' not defined."

            code = f"mcdc.TallySurface(surface={surface}, scores={scores})"
            builder.add_line(code, "tally", f"_tally_surf_{len(builder.defined['tally'])}")
            return f"Defined surface tally on {surface}"
        except Exception as e:
            return f"ERROR: Failed to create surface tally: {str(e)}"

    @tool(args_schema=SearchDocsArgs)
    def search_docs(query: str) -> str:
        """
        Search MCDC documentation for examples and API details.
        """
        try:
            # 1. Use the passed-in retriever
            docs = retriever.invoke(query)
            
            # 2. Format the results (logic copied from utils.py/format_docs)
            formatted = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "Unknown")
                source = source.replace("llm_agent/corpus/", "")
                content = doc.page_content
                formatted.append(f"[Source {i}: {source}]\n{content}")
            
            if not formatted:
                return "No relevant documents found for that query."
                
            return "\n\n---\n\n".join(formatted)
            
        except Exception as e:
            return f"ERROR during document search: {str(e)}"
    
    @tool
    def get_current_script() -> str:
        """
        Get the current state of the script being built
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
    @tool(args_schema=DeleteEntityArgs)
    def delete_entity(entity_type: str, name: str) -> str:
        """
        Delete an existing entity from the script. 
        Use this to remove mistakes or before re-creating a modified version.
        """
        success = builder.delete_entity(entity_type, name)
        if success:
            return f"Deleted {entity_type} '{name}'."
        else:
            return f"ERROR: Could not find {entity_type} '{name}' to delete."
    
    # Return all tools as a list
    return [
        create_material_from_formula,
        set_material_mg,
        set_material_ce,
        create_surface,
        create_cell,
        create_source,
        create_tally_mesh,
        create_tally_surface,
        set_settings,
        get_current_script,
        search_docs,
        delete_entity
    ]