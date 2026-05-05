import asyncio
import json
import time
from litellm import acompletion
from config import MODEL_A, MODEL_B, ACTIVE_MODEL_A, ACTIVE_MODEL_B, DOMAIN_CONTEXT, LOCAL_API_KEY, BATCH_SIZE, K_RUNS, USE_AMD_SERVER

SYSTEM_PROMPT = f"You are a helpful customer support assistant.\n{DOMAIN_CONTEXT}"


async def run_single(model, prompt, temp):
    try:
        # model is already the correct active model (set by process_batch)
        mdl = f"openai/{model}" if USE_AMD_SERVER else f"huggingface/{model}"
        base = "http://localhost:8000/v1/" if USE_AMD_SERVER else None
        
        res = await acompletion(
            model=mdl,
            api_base=base,
            api_key=LOCAL_API_KEY,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            max_tokens=300
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"


async def process_batch(batch, batch_id):
    results = []

    print(f"Processing batch {batch_id} with {len(batch)} probes...")

    tasks = []
    for probe in batch:
        # On AMD: both runs use MODEL_A (only one model served by vLLM)
        # On local HF: MODEL_A=Qwen7B, MODEL_B=Mistral7B
        run_model_a = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A
        run_model_b = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_B

        # Model A (multiple runs for consistency scoring)
        for _ in range(K_RUNS):
            tasks.append(run_single(run_model_a, prompt=probe, temp=0.5))

        # Model B (different temperature = simulates second model behaviour)
        tasks.append(run_single(run_model_b, prompt=probe, temp=0.3))

    responses = await asyncio.gather(*tasks)

    idx = 0
    for probe in batch:
        outputs_a = responses[idx: idx + K_RUNS]
        idx += K_RUNS
        output_b = responses[idx]
        idx += 1

        results.append({
            "input": probe,
            "outputs_a": outputs_a,
            "output_b": output_b
        })

    return results


async def run_all_probes_async(probes):
    start_time = time.time()

    batches = [
        probes[i:i + BATCH_SIZE]
        for i in range(0, len(probes), BATCH_SIZE)
    ]

    all_results = []
    
    # CONCURRENCY LIMITER: Protect against HTTP 429s and connection drops
    # Max 10 concurrent batches for local HF, 100 for AMD
    max_concurrent = 100 if USE_AMD_SERVER else 10
    semaphore = asyncio.Semaphore(max_concurrent)

    async def sem_process(batch, i):
        async with semaphore:
            return await process_batch(batch, i)

    # Run batches concurrently but safely constrained
    batch_tasks = [
        sem_process(batch, i)
        for i, batch in enumerate(batches)
    ]

    batch_outputs = await asyncio.gather(*batch_tasks)

    for batch in batch_outputs:
        all_results.extend(batch)

    gpu_time = time.time() - start_time

    total_inferences = len(probes) * (K_RUNS + 1)

    print(f"\nTotal GPU Time: {gpu_time:.2f} sec")
    print(f"Total Inferences: {total_inferences}")

    with open("data/results.json", "w") as f:
        json.dump({
            "results": all_results,
            "metrics": {
                "total_inferences": total_inferences,
                "gpu_time_seconds": gpu_time
            }
        }, f, indent=2)

    return all_results


def run_all_probes(probes):
    return asyncio.run(run_all_probes_async(probes))
