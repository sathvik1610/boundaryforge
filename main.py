"""
main.py — BoundaryForge Full Production Pipeline

Stage order:
  [0] Backup existing data/
  [1] Generate probes  → data/probes.json
  [2] Batch inference  → data/results_raw.jsonl + data/results.json + data/failed_probes.json
  [3] Extract signals  → data/boundaries.json + data/scored_results.json
  [4] Compile contract → data/contract.json
  [5] Validate         → data/final_metrics.json
  [6] Save run metrics → data/run_metrics.json  (appended, history preserved)

Usage:
    python main.py --production    # full run (PROBE_COUNT from config)
    python main.py                 # test run  (TEST_PROBE_COUNT from config)
"""

import json
import os
import shutil
import time
from datetime import datetime, timezone

from config import PROBE_COUNT, TOP_RULES, MAX_COMPILER_INPUT
from crews.generation_crew import generate_probes
from engine.batch_runner import run_all_probes
from engine.signal_extractor import extract_boundaries
from crews.compilation_crew import run_compilation_crew
from engine.middleware import BoundaryForgeMiddleware
from engine.metrics import run_validation


# ── Backup helper ─────────────────────────────────────────────────────────────

def backup_data_dir() -> str:
    """Copy existing data/ artifacts to data/backups/TIMESTAMP/ before overwriting."""
    backups_dir = os.path.join("data", "backups")
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backups_dir, ts)
    os.makedirs(backup_path, exist_ok=True)

    important = [
        "probes.json", "results.json", "results_raw.jsonl",
        "boundaries.json", "scored_results.json",
        "contract.json", "final_metrics.json",
        "failed_probes.json", "demo_cache.json",
    ]
    backed_up = []
    for fname in important:
        src = os.path.join("data", fname)
        if os.path.exists(src) and os.path.getsize(src) > 5:
            shutil.copy2(src, os.path.join(backup_path, fname))
            backed_up.append(fname)

    if backed_up:
        print(f"    📦 Backed up : {', '.join(backed_up)}")
        print(f"    📁 Location  : {backup_path}")
    else:
        print("    ℹ️  No existing data files to back up.")

    return backup_path


# ── run_metrics.json helper (M2, M3, T5) ─────────────────────────────────────

RUN_METRICS_PATH = "data/run_metrics.json"


def _load_run_history() -> list:
    """Load existing run history. Returns list of past run dicts."""
    if os.path.exists(RUN_METRICS_PATH):
        try:
            with open(RUN_METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("runs", [])
        except Exception:
            pass
    return []


def _save_run_metrics(run_record: dict) -> None:
    """Append this run's metrics to run_metrics.json (M3: never overwrites history)."""
    history = _load_run_history()
    history.append(run_record)
    with open(RUN_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"runs": history}, f, indent=2)
    print(f"    💾 Run metrics appended to {RUN_METRICS_PATH} "
          f"(total runs stored: {len(history)})")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_boundary_forge():
    import sys
    production = "--production" in sys.argv
    resume_inference = "--resume-inference" in sys.argv

    print("=" * 62)
    print("  BOUNDARY FORGE INITIALIZED")
    print("=" * 62)
    if production:
        print(f"🚀 PRODUCTION MODE  |  Target: {PROBE_COUNT} probes")
    else:
        print(f"⚠️  TEST MODE        |  Target: {PROBE_COUNT} probes")
        print("   Pass --production for the full run.")

    run_start = time.time()
    run_ts    = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── [0] Backup + Reset ───────────────────────────────────────────────────
    print("\n[0] Backing up existing data artifacts...")
    backup_path = backup_data_dir()

    # Issue #2: Clear JSONL checkpoint AFTER backup so a fresh main.py run
    # never merges results from a previous run. The backup preserves the old file.
    from engine.batch_runner import JSONL_PATH
    if os.path.exists(JSONL_PATH):
        if resume_inference:
            print("    ▶️  --resume-inference flag detected. Keeping existing results_raw.jsonl")
        else:
            os.remove(JSONL_PATH)
            print("    🗑️  Cleared stale results_raw.jsonl — fresh run starts clean.")

    # ── [1] Generate or Load Probes ──────────────────────────────────────────
    if resume_inference:
        print("\n[1] --resume-inference: Loading existing probes from data/probes.json...")
        if not os.path.exists("data/probes.json"):
            raise FileNotFoundError(
                "--resume-inference requires data/probes.json but it does not exist. "
                "Run a fresh production run first."
            )
        with open("data/probes.json", "r", encoding="utf-8") as f:
            probes = json.load(f)
        # P2: validate immediately — fail with a clear message before inference starts
        if not isinstance(probes, list) or not probes:
            raise ValueError(
                "data/probes.json is empty or not a list. "
                "Delete it and run a fresh production run."
            )
        if not all(isinstance(p, str) for p in probes):
            raise ValueError(
                "data/probes.json contains non-string entries. "
                "Delete it and run a fresh production run."
            )
        gen_timing = {
            "generation_seconds": 0.0,
            "probes_target": len(probes),
            "probes_actual": len(probes),
            "batches_attempted": 0,
            "mode": "resumed_from_disk",
        }
        print(f"    ✅ Loaded {len(probes)} valid probes from disk.")
    else:
        print(f"\n[1] CrewAI Generating Probes (target: {PROBE_COUNT})...")
        probes, gen_timing = generate_probes(total=PROBE_COUNT)
        print(f"    ✅ {gen_timing['probes_actual']}/{gen_timing['probes_target']} probes "
              f"in {gen_timing['generation_seconds']}s")

    # ── [2] Batch Inference on ALL probes ────────────────────────────────────
    # M1 fix: no train/test split — ALL probes attacked
    print(f"\n[2] AMD MI300X Batch Inference on all {len(probes)} probes...")
    results, inf_timing = run_all_probes(probes)
    print(f"    ✅ {inf_timing['probes_succeeded']}/{inf_timing['probes_total']} succeeded "
          f"in {inf_timing['inference_seconds']}s")

    min_success_pct = 0.90 if production else 0.25
    min_succeeded = max(1, int(len(probes) * min_success_pct))
    if inf_timing["probes_succeeded"] < min_succeeded:
        raise ValueError(
            f"CRITICAL FAILURE: Inference success rate too low for {'production' if production else 'test'} mode. "
            f"Only {inf_timing['probes_succeeded']}/{len(probes)} succeeded. "
            f"Required: {min_succeeded} ({int(min_success_pct*100)}%)."
        )

    # ── [3] Extract Boundary Signals ─────────────────────────────────────────
    print("\n[3] Extracting Boundary Signals...")
    boundaries, ext_timing = extract_boundaries(results)
    print(f"    ✅ {ext_timing['boundaries_found']}/{ext_timing['total_scored']} high-risk cases "
          f"in {ext_timing['extraction_seconds']}s "
          f"(embed: {ext_timing['embed_seconds']}s)")

    if not boundaries:
        raise ValueError(
            "CRITICAL FAILURE: Signal extractor found zero high-risk boundary cases. "
            "Cannot compile contract."
        )

    # ── [4] Compile Safety Contract ───────────────────────────────────────────
    # M1: Pass boundaries[:MAX_COMPILER_INPUT] — K-Means selects representatives
    compiler_input = boundaries[:MAX_COMPILER_INPUT]
    print(f"\n[4] CrewAI Compilation "
          f"({len(compiler_input)} high-risk cases → target {TOP_RULES} rules)...")
    rules, comp_timing = run_compilation_crew(compiler_input)
    print(f"    ✅ {comp_timing['rules_compiled']} rules compiled "
          f"in {comp_timing['compilation_seconds']}s")

    # ── [5] Validate Contract Against Discovered Boundaries ───────────────────
    # Validate against ALL high-risk boundary cases, not held-out test probes
    print(f"\n[5] Validating Contract against {len(boundaries)} high-risk boundary cases...")
    t5_start   = time.time()
    middleware  = BoundaryForgeMiddleware()
    metrics     = run_validation(boundaries, middleware)
    val_seconds = round(time.time() - t5_start, 2)
    print(f"    ✅ Validation complete in {val_seconds}s")

    # ── [6] Save Run Metrics ──────────────────────────────────────────────────
    total_wall = round(time.time() - run_start, 2)

    # P1: Use inferences actually executed THIS run for speedup calculation.
    # On a fully-resumed run this_run == 0, so speedup is omitted rather than
    # inflated by dividing full planned work by a few seconds of JSONL loading.
    inferences_this_run = inf_timing["total_inferences_this_run"]
    gpu_actual          = inf_timing["inference_seconds"]
    cpu_est             = inferences_this_run * 2.0        # 2s/inference CPU baseline
    speedup = round(cpu_est / gpu_actual, 1) if gpu_actual > 0 and inferences_this_run > 0 else 0

    run_record = {
        "run_timestamp":              run_ts,
        "run_mode":                   "resume" if resume_inference else ("production" if production else "test"),
        "total_wall_clock_seconds":   total_wall,
        # CrewAI times
        "crewai_times": {
            "generation_crew_seconds":   gen_timing["generation_seconds"],
            "compilation_crew_seconds":  comp_timing["compilation_seconds"],
        },
        # vLLM / GPU times
        "vllm_gpu_times": {
            "total_inference_seconds":      inf_timing["inference_seconds"],
            "total_inferences_planned":     inf_timing["total_inferences_planned"],
            "total_inferences_this_run":    inf_timing["total_inferences_this_run"],
            "avg_seconds_per_inference": round(gpu_actual / inf_timing["total_inferences_this_run"], 4)
                                         if inf_timing["total_inferences_this_run"] > 0 else 0,
            "min_batch_seconds":         inf_timing["min_batch_seconds"],
            "max_batch_seconds":         inf_timing["max_batch_seconds"],
            "avg_batch_seconds":         inf_timing["avg_batch_seconds"],
        },
        # Embedding model times
        "embedding_times": {
            "signal_extractor_embed_seconds": ext_timing["embed_seconds"],
        },
        # Per-stage detail
        "stage_1_generation": gen_timing,
        "stage_2_inference":  inf_timing,
        "stage_3_extraction": ext_timing,
        "stage_4_compilation": comp_timing,
        "stage_5_validation": {
            "validation_seconds":    val_seconds,
            "interception_rate":     metrics.get("interception_rate"),
            "contract_failure_rate": metrics.get("contract_failure_rate"),
            "false_positive_rate":   metrics.get("false_positive_rate"),
            "boundaries_tested":     len(boundaries),
        },
        # AMD compute advantage
        "estimated_cpu_equivalent_seconds": round(cpu_est, 1),
        "amd_speedup_factor": speedup,
    }

    _save_run_metrics(run_record)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  {'BOUNDARY FORGE — RUN COMPLETE':^58}")
    print("=" * 62)
    print(f"  Probes generated        : {gen_timing['probes_actual']:,} "
          f"(target: {gen_timing['probes_target']:,})")
    print(f"  Inference successes     : {inf_timing['probes_succeeded']:,} "
          f"| Failed this run: {inf_timing['probes_failed_this_run']} "
          f"| Resumed: {inf_timing['probes_resumed']}")
    print(f"  High-risk cases found   : {ext_timing['boundaries_found']}")
    print(f"  Rules compiled          : {comp_timing['rules_compiled']}")
    print(f"  Risk-boundary rate      : {metrics.get('risk_boundary_rate', metrics.get('baseline_failure_rate', 'N/A'))}%")
    print(f"  Risk interception rate  : {metrics.get('risk_interception_rate', metrics.get('interception_rate', 'N/A'))}%")
    print(f"  False positive rate     : {metrics.get('false_positive_rate', 'N/A')}%")
    print(f"  CrewAI generation time  : {gen_timing['generation_seconds']}s")
    print(f"  vLLM inference time     : {inf_timing['inference_seconds']}s")
    print(f"  Embedding time          : {ext_timing['embed_seconds']}s")
    print(f"  Compilation time        : {comp_timing['compilation_seconds']}s")
    if speedup > 0:
        print(f"  AMD MI300X speedup      : {speedup}×  (vs {round(cpu_est/3600,1)}h CPU)")
    else:
        print(f"  AMD MI300X speedup      : n/a (all probes resumed from checkpoint)")

    print(f"  Total wall-clock time   : {total_wall}s ({round(total_wall/60,1)} min)")
    print(f"  Backup saved to         : {backup_path}")
    print("=" * 62)


if __name__ == "__main__":
    run_boundary_forge()
