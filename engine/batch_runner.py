"""
batch_runner.py — AMD MI300X Batch Inference Engine

Changes vs original:
  I1: Temperature ladder [0.2, 0.5, 0.9] for outputs_a + 0.5 anchor for output_b
  I2: Per-call retry with exponential backoff (max 2 retries: 1s → 2s)
  I3: JSONL checkpointing — each completed batch appended to results_raw.jsonl immediately
  I4: Resume support — skips probe inputs already present in results_raw.jsonl
  I5: Saves data/failed_probes.json for all probes where all retries exhausted
  I6: Merges results_raw.jsonl → results.json at end (existing consumer format)
  I7/T1: Returns timing dict with per-batch GPU wall-clock stats
"""

import asyncio
import json
import os
import time

from litellm import acompletion
from config import (
    MODEL_A, ACTIVE_MODEL_A, DOMAIN_CONTEXT,
    LOCAL_API_KEY, BATCH_SIZE, K_RUNS, USE_AMD_SERVER,
)

SYSTEM_PROMPT = f"You are a helpful customer support assistant.\n{DOMAIN_CONTEXT}"

# Temperature ladder: strict / normal / stress (K_RUNS=3 expected)
# output_b uses a stable 0.5 anchor for consistent divergence scoring
TEMPS_A = [0.2, 0.5, 0.9]
TEMP_B  = 0.5

# NOTE: TEMPS_A length defines the number of probe runs (replaces K_RUNS loop).
# K_RUNS in config.py is no longer used to loop temperatures — TEMPS_A is authoritative.
# If K_RUNS != len(TEMPS_A), that is fine: they are independent settings now.

# ── Concurrency semaphore ────────────────────────────────────────────────────
_default_concurrent = 50 if USE_AMD_SERVER else 10   # safe default for 8192 ctx
max_concurrent = int(os.getenv("MAX_CONCURRENT", str(_default_concurrent)))
request_semaphore = asyncio.Semaphore(max_concurrent)


# ── Single call with retry (I2) ──────────────────────────────────────────────

async def _run_single(model: str, prompt: str, temp: float) -> str:
    """One inference call. No retry here — retry wrapper is below."""
    from config import LOCAL_LLM_URL_A
    mdl  = f"openai/{model}"      if USE_AMD_SERVER else f"huggingface/{model}"
    base = LOCAL_LLM_URL_A        if USE_AMD_SERVER else None
    res = await acompletion(
        model=mdl, api_base=base, api_key=LOCAL_API_KEY,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=temp,
        max_tokens=512,
    )
    return res.choices[0].message.content


async def _run_with_retry(model: str, prompt: str, temp: float,
                          max_retries: int = 2) -> str:
    """Run inference with exponential backoff retry (I2).
    Returns 'ERROR: ...' only after all retries are exhausted.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return await _run_single(model, prompt, temp)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)   # 1s, 2s
    return f"ERROR: {last_err}"


async def _sem_run(model: str, prompt: str, temp: float) -> str:
    async with request_semaphore:
        return await _run_with_retry(model, prompt, temp)


# ── Batch processor (T1: per-batch GPU timing) ───────────────────────────────

async def _process_batch(batch: list, batch_id: int, model: str) -> tuple:
    """Returns (results, failed_probes, batch_meta)."""
    batch_start = time.time()
    print(f"  [Batch {batch_id}] Starting {len(batch)} probes "
          f"(temps_a={TEMPS_A}, temp_b={TEMP_B})...")

    # Fire all calls concurrently within this batch
    tasks = []
    for probe in batch:
        for t in TEMPS_A:
            tasks.append(_sem_run(model, probe, t))
        tasks.append(_sem_run(model, probe, TEMP_B))

    responses = await asyncio.gather(*tasks)

    results  = []
    failed   = []
    n_calls  = len(TEMPS_A) + 1          # calls per probe

    for idx_p, probe in enumerate(batch):
        base = idx_p * n_calls
        outputs_a = list(responses[base: base + len(TEMPS_A)])
        output_b  = responses[base + len(TEMPS_A)]

        # If any call failed after retries, log the probe as failed
        if any(r.startswith("ERROR:") for r in outputs_a) or output_b.startswith("ERROR:"):
            failed.append({"input": probe, "errors": outputs_a + [output_b]})
            continue

        results.append({
            "input":    probe,
            "outputs_a": outputs_a,
            "output_b":  output_b,
            "temps_a":   TEMPS_A,
            "temp_b":    TEMP_B,
        })

    batch_wall = round(time.time() - batch_start, 3)
    batch_meta = {
        "batch_id":          batch_id,
        "batch_size":        len(batch),
        "batch_succeeded":   len(results),
        "batch_failed":      len(failed),
        "batch_wall_seconds": batch_wall,
        "batch_inferences":  len(batch) * n_calls,
    }
    print(f"  [Batch {batch_id}] Done — {len(results)} ok, {len(failed)} failed, {batch_wall}s")
    return results, failed, batch_meta


# ── JSONL checkpoint helpers (I3, I4, I6) ────────────────────────────────────

JSONL_PATH = "data/results_raw.jsonl"


def _is_valid_row(entry: dict, check_ladder: bool = False) -> bool:
    """Shared schema validator for JSONL rows.
    Matches the extractor's requirements so only rows that will survive
    signal_extractor.py are counted as valid checkpoints.

    check_ladder=True: also requires outputs_a length == len(TEMPS_A)
    (used by _load_completed_inputs to skip old-ladder rows).
    """
    if not isinstance(entry, dict):
        return False
    inp      = entry.get("input", "")
    outputs_a = entry.get("outputs_a", [])
    output_b  = entry.get("output_b", None)

    if not isinstance(inp, str) or not inp:
        return False
    if not isinstance(outputs_a, list) or not outputs_a:
        return False
    if not all(isinstance(x, str) for x in outputs_a):
        return False
    if not isinstance(output_b, str):
        return False
    if check_ladder and len(outputs_a) != len(TEMPS_A):
        return False
    return True


def _load_completed_inputs(probe_set: set = None) -> set:
    """Return set of probe input strings already saved in JSONL (I4 resume).
    Filters to probe_set, validates full schema, and checks outputs_a length
    matches the current TEMPS_A ladder so old-ladder rows force a re-run.
    """
    completed = set()
    if not os.path.exists(JSONL_PATH):
        return completed
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    continue
                inp = entry.get("input", "")
                if probe_set is not None and inp not in probe_set:
                    continue
                # check_ladder=True: old TEMPS_A rows are NOT treated as done
                if _is_valid_row(entry, check_ladder=True):
                    completed.add(inp)
            except json.JSONDecodeError:
                pass
    return completed


def _append_batch_to_jsonl(results: list) -> None:
    """Append completed probe results to JSONL immediately (I3)."""
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_all_jsonl(probe_set: set = None) -> list:
    """Load saved probe results from JSONL (I6 merge source).
    Deduplicates by input key. Only overwrites if the later row is also valid,
    so a malformed late row cannot replace a good earlier row.
    """
    seen: dict = {}   # input -> entry (last *valid* row wins)
    if not os.path.exists(JSONL_PATH):
        return []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    continue
                inp = entry.get("input", "")
                if probe_set is not None and inp not in probe_set:
                    continue
                # Only keep the row if it is fully valid
                if _is_valid_row(entry):
                    seen[inp] = entry
            except json.JSONDecodeError:
                pass
    return list(seen.values())




# ── Main async entry point ────────────────────────────────────────────────────

async def run_all_probes_async(probes: list) -> tuple:
    """Run inference on all probes. Returns (all_results, timing_dict)."""
    run_start = time.time()
    model = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A

    # I4: Resume — skip already-completed probes (scoped + schema-validated)
    probe_set        = set(probes)
    completed_inputs = _load_completed_inputs(probe_set=probe_set)
    if completed_inputs:
        print(f"[Inference] Resume: {len(completed_inputs)} valid probes already done, skipping.")
    remaining = [p for p in probes if p not in completed_inputs]
    print(f"[Inference] Running {len(remaining)} probes "
          f"(+{len(completed_inputs)} resumed) | MAX_CONCURRENT={max_concurrent}")


    # Build batches from remaining probes only
    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]

    all_failed  = []
    batch_metas = []

    # Process batches sequentially to keep vLLM load steady.
    # Individual calls within each batch are concurrent (via semaphore).
    for i, batch in enumerate(batches):
        b_results, b_failed, b_meta = await _process_batch(batch, i, model)

        # I3: Checkpoint immediately — survives crash after this point
        _append_batch_to_jsonl(b_results)

        all_failed.extend(b_failed)
        batch_metas.append(b_meta)

    # I6: Load only rows for this run's probe set, deduplicated (prevents stale contamination)
    all_results = _load_all_jsonl(probe_set=probe_set)

    gpu_time = round(time.time() - run_start, 2)

    # Issue #5: report both planned inferences and actually-executed-this-run
    total_inferences_planned      = len(probes)    * (len(TEMPS_A) + 1)
    total_inferences_this_run     = len(remaining) * (len(TEMPS_A) + 1)
    batch_times = [m["batch_wall_seconds"] for m in batch_metas] or [0.0]

    # I7/T1: Timing dict
    timing = {
        "inference_seconds":            gpu_time,
        "total_inferences_planned":     total_inferences_planned,
        "total_inferences_this_run":    total_inferences_this_run,
        # Legacy key preserved so metrics.py / run_summary still work
        "total_inferences":             total_inferences_planned,
        "probes_total":                 len(probes),
        "probes_succeeded":             len(all_results),
        "probes_failed_this_run":       len(all_failed),   # Issue #4: named clearly
        "probes_resumed":               len(completed_inputs),
        "min_batch_seconds":            min(batch_times),
        "max_batch_seconds":            max(batch_times),
        "avg_batch_seconds":            round(sum(batch_times) / len(batch_times), 2),
        "temps_a":                      TEMPS_A,
        "temp_b":                       TEMP_B,
        "batch_details":                batch_metas,
    }

    # I6: Write merged results.json (existing consumer format preserved)
    with open("data/results.json", "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "metrics": {
                "total_inferences": total_inferences_planned,
                "total_inferences_this_run": total_inferences_this_run,
                "gpu_time_seconds": gpu_time,
                "temps_a": TEMPS_A,
                "temp_b":  TEMP_B,
            },
        }, f, indent=2)

    # I5: Always write failed_probes.json — even if empty — so stale files never survive (Issue #3)
    with open("data/failed_probes.json", "w", encoding="utf-8") as f:
        json.dump(all_failed, f, indent=2)
    if all_failed:
        print(f"[Inference] ⚠️  {len(all_failed)} failed probes → data/failed_probes.json")
    else:
        print("[Inference] ✅ No failed probes.")

    print(f"\n[Inference] Total GPU time      : {gpu_time:.2f}s")
    print(f"[Inference] Inferences planned  : {total_inferences_planned}")
    print(f"[Inference] Inferences this run : {total_inferences_this_run}")
    print(f"[Inference] Succeeded           : {len(all_results)}  |  Failed this run: {len(all_failed)}")

    return all_results, timing


def run_all_probes(probes: list) -> tuple:
    """Synchronous entry point. Returns (results, timing_dict)."""
    return asyncio.run(run_all_probes_async(probes))
