CONCEPT_LESSONS = {
    "material": {
        "concept": (
            "A *material* defines what your geometry is made of—its isotopic composition, "
            "mass density, and macroscopic nuclear cross-sections. Every cell must contain "
            "exactly one material, and materials must be defined before they can be assigned."
        ),
        "parts": (
            "composition (dict mapping nuclide to density), density (float, g/cm³), "
            "optional: capture/scatter/fission cross-section arrays for MG mode"
        ),
        "key_questions": [
            "What is the difference between mcdc.material (continuous) and mcdc.MaterialMG (multigroup)?",
            "How do I define a common material like water, air, or uranium?",
            "What are the units for density and composition?",
            "When should I use a nuclide card vs. direct macroscopic constants?"
        ]
    },
    
    "surface": {
        "concept": (
            "A *surface* is a geometric boundary (plane, sphere, cylinder, etc.) that defines "
            "the inside vs. outside regions of your model. Surfaces are combined with boolean "
            "operators (& for intersection, | for union) to create cell regions. The +/- sign "
            "indicates which side of the surface is considered 'inside'."
        ),
        "parts": (
            "surface_type (str: 'PlaneX', 'PlaneY', 'PlaneZ', 'Sphere', 'Cylinder', etc.), "
            "parameters (list: position, radius, etc.), boundary_condition (optional: 'vacuum', 'reflective')"
        ),
        "key_questions": [
            "What surface types are available and what parameters do they need?",
            "How do I create a simple box, sphere, or cylinder?",
            "What does the + or - sign before a surface mean?",
            "How do I combine multiple surfaces to make a complex cell shape?"
        ]
    },
    
    "cell": {
        "concept": (
            "A *cell* is a region of space (defined by surfaces) filled with a material "
            "or another universe. Cells are the building blocks of your geometry. "
            "Each cell must specify a region (boolean expression of surfaces) and a fill "
            "(material or sub-geometry). Cells can be translated or rotated."
        ),
        "parts": (
            "region (Region object: e.g., +s1 & -s2), fill (Material or Cell), "
            "optional: translation (list), rotation (list)"
        ),
        "key_questions": [
            "How do I define a cell that is the intersection of two surfaces?",
            "What is the difference between fill=m1 and fill=another_cell?",
            "What is a universe and when do I need one?",
            "Can a cell contain multiple materials?"
        ]
    },

    "hierarchy": {
        "concept": (
            "Hierarchies allow you to build complex, repeating structures without redefining geometry "
            "thousands of times. Think of a 'Universe' as a container or a blueprint that groups cells together. "
            "You can then place that Universe inside another Cell using the 'fill' parameter. "
            "Lattices take this further by creating a grid where each voxel is filled by a specific Universe."
        ),
        "parts": (
            "1. Universe: A named collection of cells. It has no boundaries itself; it extends infinitely "
            "until 'clipped' by the cell that contains it.\n"
            "2. Lattice: A structured grid (like a checkerboard). You define the grid lines (x, y, z) "
            "and map each grid element (i, j, k) to a specific Universe.\n"
            "3. Fill: The link between levels. A Cell in the main world can be 'filled' with a "
            "Universe or Lattice instead of a Material."
        ),
        "key_questions": [
            "How do I put a Universe inside a Cell? (Use the 'fill' parameter in create_cell)",
            "What is the 'root' universe? (The top-level universe where particles start)",
            "What are the steps to create a lattice? (Define grid lines, create universes, map them to the lattice)"
        ]
    },

    "source": {
        "concept": (
            "A *source* defines where, when, and with what energy particles are born. "
            "Sources can be point, line, surface, or volume distributions. "
            "Energy can be mono-energetic, spectrum, or multigroup. "
            "Direction can be isotropic (uniform in all directions) or a fixed vector."
        ),
        "parts": (
            "position (list or bounds), energy (float or array), "
            "direction (list or 'isotropic'), time (optional), group (for MG)"
        ),
        "key_questions": [
            "How do I place a point source at the origin?",
            "How do I make a uniform volume source in a box?",
            "What does 'isotropic' mean and when should I use it?",
            "How do I set the source energy to 1 MeV?"
        ]
    },
    
    "tally": {
        "concept": (
            "A *tally* is a detector that records what you want to measure: "
            "particle flux, reaction rates, currents through surfaces, or mesh-binned data. "
            "Tallies can be placed on surfaces, in cells, or on a structured mesh. "
            "Multiple scores (e.g., flux, collision, net-current) can be recorded simultaneously."
        ),
        "parts": (
            "tally_type ('Surface', 'Mesh', 'Cell'), geometry (surface, mesh, or cell), "
            "scores (list of strings), energy_bins (optional), mu_bins (optional)"
        ),
        "key_questions": [
            "What is the difference between TallySurface, TallyMesh, and TallyCell?",
            "What scores are available (flux, collision, net-current)?",
            "How do I bin my tally in energy or angle?",
            "When should I use a mesh tally vs. a surface tally?"
        ]
    },
    
    "settings": {
        "concept": (
            "*Settings* control the simulation: number of particles, batches, "
            "convergence criteria, variance reduction, and output verbosity. "
            "Settings are global and affect the entire simulation. "
            "Critical simulations (eigenmode) require different settings than fixed-source problems."
        ),
        "parts": (
            "N_particle (int), N_batch (int), N_cycle (optional for eigenmode), "
            "output (str: 'stdout', 'h5'), population_control (optional)"
        ),
        "key_questions": [
            "How many particles do I need for a good statistics?",
            "What is the difference between N_particle and N_batch?",
            "How do I control output files and verbosity?",
            "What settings are needed for a criticality (k-eigenvalue) calculation?"
        ]
    },
    
    "run": {
        "concept": (
            "`mcdc.run()` executes the simulation with all definitions you've provided. "
            "It validates the input, initializes the particle bank, runs batches, "
            "and writes tallies to the specified output file. "
            "After running, you can inspect tallies to see flux, reaction rates, and k-effective (if eigenmode)."
        ),
        "parts": (
            "No parameters – all configuration comes from preceding definitions. "
            "Output: HDF5 file (.h5) with tallies, runtime info, and convergence data."
        ),
        "key_questions": [
            "What happens when I call mcdc.run()?",
            "How do I know if the simulation finished successfully?",
            "Where are the results stored?",
            "What should I do if I get an error during the run?"
        ]
    }
}