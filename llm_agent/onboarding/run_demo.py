# script is supposed to automatically run the slab_absorbium test

import time
from llm_agent.utils import load_llm
from llm_agent.onboarding.tutor import MCDCTutor, Colors

INPUT_SCRIPT = [
    # --- MATERIAL STEP ---
    "", "y", 
    "Create MG material m1 with capture=[1.0]", "Yes",
    "Create MG material m2 with capture=[1.5]", "Yes", 
    "Create MG material m3 with capture=[2.0]", "Yes",
    "", # Finish Step

    # --- SURFACE STEP ---
    "", "y",
    "Create PlaneZ at z=0.0 named s1 boundary vacuum", "Yes",
    "Create PlaneZ at z=2.0 named s2", "Yes",
    "Create PlaneZ at z=4.0 named s3", "Yes",
    "Create PlaneZ at z=6.0 named s4 boundary vacuum", "Yes",
    "", # Finish Step

    # --- CELL STEP ---
    "", "y",
    "Create cell filled with m2 region +s1 & -s2",
    "Create cell filled with m3 region +s2 & -s3",
    "Create cell filled with m1 region +s3 & -s4",
    "", # Finish Step

    # --- SOURCE STEP ---
    "", "y",
    "Create isotropic source from z=0.0 to 6.0 energy_group=0", 
    "", # Finish Step

    # --- TALLY STEP ---
    "", "y",
    "Create TallySurface on s4 scores net-current",
    "Create MeshStructured from z=0.0 to 6.0 with 61 points, scores flux and collision",
    "", # Finish Step

    # --- SETTINGS STEP ---
    "", "y",
    "Set N_particle=100 and N_batch=2",
    "", # Finish Step

    # save
    "y", 
    "automated_slab.py"
]

class AutoInput:
    def __init__(self, script):
        self.script = iter(script)
        self.count = 0

    def __call__(self, prompt=""):
        # Print the prompt so we see what the tutor is asking
        print(prompt, end="")
        try:
            response = next(self.script)
            time.sleep(1.5)
            print(f"{Colors.CYAN}{response}{Colors.ENDC}")
            return response
        except StopIteration:
            return "n"

if __name__ == "__main__":
    try:
        llm = load_llm(temperature=0.0)
        
        # Initialize AutoInput with our script
        auto_input = AutoInput(INPUT_SCRIPT)
        
        # Pass auto_input to the tutor
        tutor = MCDCTutor(llm, input_func=auto_input)
        
        tutor.run_onboarding()
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()