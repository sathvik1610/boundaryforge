import asyncio
import json
import time
from litellm import acompletion
from config import MODEL_A, MODEL_B, ACTIVE_MODEL_A, ACTIVE_MODEL_B, DOMAIN_CONTEXT, LOCAL_API_KEY, BATCH_SIZE, K_RUNS, USE_AMD_SERVER

SYSTEM_PROMPT = f"You are a helpful customer support assistant.\n{DOMAIN_CONTEXT}"


async def run_single(model, prompt, temp):
    try:
        # model is already the correct active model (set by process_batch)
        from config import LOCAL_LLM_URL_A
        mdl = f"openai/{model}" if USE_AMD_SERVER else f"huggingface/{model}"
        base = LOCAL_LLM_URL_A if USE_AMD_SERVER else None
        
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


# Global request semaphore
max_concurrent = 100 if USE_AMD_SERVER else 10
request_semaphore = asyncio.Semaphore(max_concurrent)

async def sem_run_single(model, prompt, temp):
    async with request_semaphore:
        return await run_single(model, prompt, temp)

async def process_batch(batch, batch_id):
    results = []
    print(f"Processing batch {batch_id} with {len(batch)} probes...")

    tasks = []
    for probe in batch:
        # AMD Mode: Uses single model, divergence is simulated via high/low temperature on MODEL_A
        # Local HF Mode: Uses MODEL_A (Qwen) vs MODEL_B (Mistral)
        run_model_a = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A
        run_model_b = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_B

        for _ in range(K_RUNS):
            tasks.append(sem_run_single(run_model_a, prompt=probe, temp=0.5))

        tasks.append(sem_run_single(run_model_b, prompt=probe, temp=0.3))

    responses = await asyncio.gather(*tasks)

    idx = 0
    for probe in batch:
        outputs_a = responses[idx: idx + K_RUNS]
        idx += K_RUNS
        output_b = responses[idx]
        idx += 1

        # Filter out failed probes to avoid corrupting downstream models
        if any(out.startswith("ERROR:") for out in outputs_a) or output_b.startswith("ERROR:"):
            continue

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

    # Run batches concurrently (individual requests are throttle-protected by request_semaphore)
    batch_tasks = [
        process_batch(batch, i)
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
