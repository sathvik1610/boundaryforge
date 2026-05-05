"""
validate_only.py — Re-run Stage 5 (Middleware Validation) only.

Use this when:
- You already have data/contract.json and data/boundaries.json
- You want to quickly test a contract change without re-running the full pipeline
- You are tuning middleware or contract rules before a demo

Usage:
    python validate_only.py
"""

import json
from engine.middleware import BoundaryForgeMiddleware
from engine.metrics import run_validation


def run_validate_only():
    print("=== BOUNDARY FORGE — VALIDATE ONLY ===")

    print("\n[1] Loading boundaries...")
    with open("data/boundaries.json") as f:
        boundaries = json.load(f)
    print(f"    Loaded {len(boundaries)} known boundary failures.")

    print("\n[2] Loading contract...")
    with open("data/contract.json") as f:
        contract = json.load(f)
    print(f"    Loaded {len(contract['rules'])} rules from contract.json.")

    print("\n[3] Running middleware validation against known failures...")
    middleware = BoundaryForgeMiddleware()
    test_slice = boundaries[:min(25, len(boundaries))]
    metrics = run_validation(test_slice, middleware)

    print("\n=== VALIDATION COMPLETE ===")
    print(f"\nFinal metrics saved to data/final_metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_validate_only()
