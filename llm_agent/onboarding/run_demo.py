# script is supposed to automatically run the slab_absorbium test

import time
from llm_agent.utils import load_llm
from llm_agent.onboarding.tutor import MCDCTutor, Colors

INPUT_SCRIPT = [
    # --- MATERIAL STEP ---
    "", "Y", 
    "Create 3 materials named m1, m2, and m3 that have capture = 1, 1.5, 2",
    "", # Finish Step

    # --- SURFACE STEP ---
    "", "Y",
    "create 4 surfaces named s1-4 that are planes with z=0, 2, 4, 6. The first and last surfaces should be vacuums",
    "", # Finish Step

    # --- CELL STEP ---
    "", "Y",
    "fill the area between s1 and s2 with m2, between s2 and s3 with m3, and between s3 and s4 with m1",
    "", # Finish Step

    # --- SOURCE STEP ---
    "", "Y",
    "create the source between z = 0-6.0, it should be isotropic and energy group 0", 
    "", # Finish Step

    # --- TALLY STEP ---
    "", "Y",
    "create a tally surface on s4 that tracks the flow of particles over that surface",
    "create a mesh tally that uses a structured mesh from z=0-6 with 60 points. It should track flux and particle collisions, track their direction using 32 bins",
    "", # Finish Step

    # --- SETTINGS STEP ---
    "", "Y",
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
        llm = load_llm(temperature=0.1)
        
        # Initialize AutoInput with our script
        auto_input = AutoInput(INPUT_SCRIPT)
        
        # Pass auto_input to the tutor
        tutor = MCDCTutor(llm, input_func=auto_input)
        
        tutor.run_onboarding()
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()