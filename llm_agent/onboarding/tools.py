import json
import numpy as np
from langchain.tools import tool
from onboarding.script_builder import ScriptBuilder

# FIX: Import json for get_current_script


def get_mcdc_tools(builder: ScriptBuilder):
    """
    Factory function that creates tools bound to a specific ScriptBuilder instance.
    
    This uses closures to "bake in" the builder instance that the tools will operate on.
    Each MCDCTutor instance creates its own set of tools with its own builder.
    
    FIX: Now accepts builder parameter instead of using global instance.
    """
    
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
        
        Parameters are numpy array strings, e.g.:
        - capture: "[0.5]" for 1-group, "[0.5, 0.3]" for 2-group
        - scatter: "[[0.9]]" for 1-group, "[[0.8, 0.1], [0.05, 0.85]]" for 2-group
        - fission: "[0.1]" (optional)
        - nu_p: "[2.5]" (optional, neutrons per fission)
        - speed: "[200000.0]" (optional, cm/s)
        
        Example: set_material_mg("fuel", "[0.45]", "[[0.0]]", "[0.55]", "[2.5]")
        
        FIX: 
        - Changed from mcdc.material() to mcdc.MaterialMG()
        - Parameters now match actual MCDC API
        - Accept strings and parse them (LLM outputs strings, not Python objects)
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
        
        FIX: Added CE material support (different from MG)
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
        
        FIX:
        - Changed from mcdc.surface.Type to mcdc.Surface.Type (capital S)
        - Accept string params instead of list (easier for LLM)
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
        
        FIX:
        - Removed 'name=' parameter (not in MCDC API)
        - If name is empty, don't assign to variable
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
        
        FIX: Added proper parameter handling for MCDC Source API
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
        
        FIX: Added tally support
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
        
        FIX: Added settings support
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
        
        # FIX: Added json import at top of file
        return f"""
CURRENT SCRIPT:
{script}

DEFINED ENTITIES:
{json.dumps(summary, indent=2)}
"""
    
    # Return all tools as a list
    return [
        set_material_mg,
        set_material_ce,
        create_surface,
        create_cell,
        create_source,
        create_tally_mesh,
        set_settings,
        get_current_script,
        # search_docs  # Commented out - needs retriever access
    ]