import json
import random
from crews.generation_crew import generate_probes
from engine.batch_runner import run_all_probes
from engine.signal_extractor import extract_boundaries
from crews.compilation_crew import run_compilation_crew
from engine.middleware import BoundaryForgeMiddleware
from engine.metrics import run_validation
from config import PROBE_COUNT

def run_boundary_forge():
    print("=== BOUNDARY FORGE INITIALIZED ===")
    
    print("\n[1] CrewAI Generating Probes...")
    probes = generate_probes(total=PROBE_COUNT)
    random.shuffle(probes)
    train, test = probes[:max(1, len(probes)//2)], probes[max(1, len(probes)//2):] # dynamic split for testing
    
    print("\n[2] AMD MI300X Batch Inference...")
    results = run_all_probes(train)
    
    print("\n[3] Extracting Signals...")
    boundaries = extract_boundaries(results)
    
    print("\n[4] CrewAI Hierarchical Compilation...")
    rules = run_compilation_crew(boundaries)
    
    print("\n[5] Validating Contract via Middleware...")
    middleware = BoundaryForgeMiddleware()
    metrics = run_validation(test, middleware)
    
    print(f"\n✅ SUCCESS. Failure rate dropped from {metrics['baseline_failure_rate']}% to {metrics['contract_failure_rate']}%")

if __name__ == "__main__":
    run_boundary_forge()
