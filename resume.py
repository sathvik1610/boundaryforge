"""
resume.py — Resume BoundaryForge from Stage 4 (Compilation + Validation)

Use when Stages 1-3 are done (data/boundaries.json populated) and you want
to recompile the contract and re-validate without re-running inference.

Safe to run multiple times — does not touch probes.json or results.json.
"""

import json
import time
from datetime import datetime, timezone

from config import TOP_RULES, MAX_COMPILER_INPUT
from crews.compilation_crew import run_compilation_crew
from engine.middleware import BoundaryForgeMiddleware
from engine.metrics import run_validation


def resume_compilation():
    print("=== BOUNDARY FORGE RESUME SCRIPT ===")
    run_start = time.time()
    run_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("\n[1] Loading previously extracted boundary signals...")
    with open("data/boundaries.json", "r", encoding="utf-8") as f:
        boundaries = json.load(f)
    print(f"    Loaded {len(boundaries)} boundaries.")

    if not boundaries:
        raise ValueError(
            "CRITICAL FAILURE: data/boundaries.json is empty. "
            "Re-run the full pipeline (python main.py --production)."
        )

    # M1/M4: Use MAX_COMPILER_INPUT, not just TOP_RULES
    compiler_input = boundaries[:MAX_COMPILER_INPUT]
    print(f"\n[2] CrewAI Compilation "
          f"({len(compiler_input)} high-risk cases → target {TOP_RULES} rules)...")
    rules, comp_timing = run_compilation_crew(compiler_input)
    print(f"    ✅ {comp_timing['rules_compiled']} rules in {comp_timing['compilation_seconds']}s")

    print(f"\n[3] Validating Contract against all {len(boundaries)} high-risk boundary cases...")
    t_val = time.time()
    middleware = BoundaryForgeMiddleware()
    metrics    = run_validation(boundaries, middleware)
    val_s      = round(time.time() - t_val, 2)

    total_wall = round(time.time() - run_start, 2)

    # Append to run_metrics.json
    try:
        from main import _save_run_metrics
        _save_run_metrics({
            "run_timestamp":           run_ts,
            "run_mode":                "resume",
            "total_wall_clock_seconds": total_wall,
            "stage_4_compilation":     comp_timing,
            "stage_5_validation": {
                "validation_seconds":    val_s,
                "interception_rate":     metrics.get("interception_rate"),
                "contract_failure_rate": metrics.get("contract_failure_rate"),
                "false_positive_rate":   metrics.get("false_positive_rate"),
                "boundaries_tested":     len(boundaries),
            },
        })
    except Exception as e:
        print(f"    ⚠️  Could not save run_metrics.json: {e}")

    print("\n=== RESUME COMPLETE ===")
    print(f"  Rules compiled    : {comp_timing['rules_compiled']}")
    print(f"  Risk interception : {metrics.get('risk_interception_rate', metrics.get('interception_rate', 'N/A'))}%")
    print(f"  Miss rate         : {metrics.get('contract_failure_rate', 'N/A')}%")
    print(f"  False positive    : {metrics.get('false_positive_rate', 'N/A')}%")
    print(f"  Total time        : {total_wall}s")


if __name__ == "__main__":
    resume_compilation()
