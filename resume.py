import json
from crews.compilation_crew import run_compilation_crew
from engine.middleware import BoundaryForgeMiddleware
from engine.metrics import run_validation

def resume_compilation():
    print("=== BOUNDARY FORGE RESUME SCRIPT ===")
    print("\n[1] Loading previously extracted signals...")
    with open("data/boundaries.json", "r") as f:
        boundaries = json.load(f)
        
    print(f"Loaded {len(boundaries)} boundaries.")
    
    print("\n[2] CrewAI Hierarchical Compilation...")
    # K-Means clustering inside the crew now handles intelligent selection & token limits
    rules = run_compilation_crew(boundaries)
    
    print("\n[3] Validating Contract via Middleware...")
    middleware = BoundaryForgeMiddleware()
    
    # We validate the contract against the known vulnerabilities it was built to protect against
    if boundaries:
        test_slice = boundaries[:min(25, len(boundaries))]
        run_validation(test_slice, middleware)
    else:
        print("No boundaries found to validate.")
        
    print("\n=== RESUME COMPLETE ===")

if __name__ == "__main__":
    resume_compilation()
