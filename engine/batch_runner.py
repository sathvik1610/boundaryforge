import asyncio
import json
import time
from openai import AsyncOpenAI
from config import MODEL_A, MODEL_B, DOMAIN_CONTEXT, LOCAL_LLM_URL, LOCAL_API_KEY, BATCH_SIZE, K_RUNS

client_a = AsyncOpenAI(base_url=LOCAL_LLM_URL, api_key=LOCAL_API_KEY)
client_b = AsyncOpenAI(base_url="http://localhost:8002/v1", api_key=LOCAL_API_KEY)

SYSTEM_PROMPT = f"You are a helpful customer support assistant.\n{DOMAIN_CONTEXT}"


async def run_single(client, model, prompt, temp):
    try:
        res = await client.chat.completions.create(
            model=model,
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
        # Model A (multiple runs)
        for _ in range(K_RUNS):
            tasks.append(run_single(client_a, MODEL_A, probe, 0.5))

        # Model B
        tasks.append(run_single(client_b, MODEL_B, probe, 0.3))

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

    # Run batches concurrently (VERY IMPORTANT)
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
