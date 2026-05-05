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
    # Cap at top 8 to avoid 4096 context length limits!
    rules = run_compilation_crew(boundaries[:8])
    
    print("\n[3] Validating Contract via Middleware...")
    middleware = BoundaryForgeMiddleware()
    
    # We validate on a small slice of the test probes
    with open("data/results.json", "r") as f:
        results = json.load(f).get("results", [])
    
    if results:
        test_slice = results[:min(25, len(results))]
        run_validation(test_slice, middleware)
    else:
        print("No test probes found to validate.")
        
    print("\n=== RESUME COMPLETE ===")

if __name__ == "__main__":
    resume_compilation()
