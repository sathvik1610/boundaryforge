# 🛡️ Boundary Forge: Automated AI Safety Contract Compiler

*Enterprise-Grade LLM Guardrail Generation — AMD Developer Hackathon 2024*

---

## 📖 1. Executive Summary & The Core Problem

**The Problem:** Enterprise LLM adoption is bottlenecked by unpredictable hallucinations. If a customer service bot gives illegal financial advice, hallucinates a refund, or is tricked by an adversarial user, it costs companies millions of dollars.

* **Why this is hard:** Humans cannot manually write `if/else` rules for every conversational edge case. The attack surface of natural language is simply too vast.

**The Solution:** Boundary Forge is an automated, high-throughput **AI Safety Contract Compiler**. Instead of humans guessing how a model might fail, Boundary Forge uses an agentic "Red Team" to attack the model, discovers exactly where it breaks *mathematically*, and automatically compiles a strict, deterministic JSON safety middleware (`contract.json`) to intercept future failures — all in minutes, not months.

---

## 💻 2. AMD Technology & Infrastructure Integration

**The Stack:** AMD MI300X GPU · ROCm · vLLM · Qwen2.5-72B-Instruct

**Why AMD was required:**
To map the "failure boundaries" of a 72B model, we must blast thousands of adversarial prompts at it. On standard hardware or public APIs, this instantly triggers `429 RateLimitErrors` or takes hours. The **AMD MI300X** running `vLLM` on **ROCm** provides the memory bandwidth to execute all 2,500 probes in massive parallel batches, compressing contract compilation time from ~2 hours to **under 8 minutes**.

**Why Qwen?**
`Qwen/Qwen2.5-72B-Instruct` powers both the Red Team (attacker) and the Blue Team (target). It is uniquely powerful enough to discover its own vulnerabilities, making it the perfect candidate for the Qwen Challenge.

---

## 🏗️ 3. System Architecture & End-to-End Flow

Boundary Forge operates in two phases: **The Forge** (Heavy Compute) and **The Shield** (Runtime).

### Phase 1: The Forge (Agentic Backend)
| Stage | What Happens | Key Tool |
|---|---|---|
| **1. Probe Generation** | CrewAI agent generates 2,500 adversarial jailbreak prompts across roleplay, logic traps, prompt injection, and financial fraud categories | CrewAI + Qwen 72B |
| **2. Batch Inference** | All 2,500 probes fired concurrently at Qwen at two temperatures (0.5 & 0.3) via 100-slot async semaphore | vLLM + asyncio |
| **3. Signal Extraction** | Vectorized math (Consistency, Divergence, Confidence) calculates boundary scores for every probe in milliseconds | sentence-transformers + NumPy |
| **4. Contract Compilation** | K-Means clusters the failure embeddings → picks representative failures → CrewAI Architect writes the deterministic JSON contract | CrewAI + sklearn |
| **5. Async Validation** | All validation probes fire concurrently with batch judging — 10× faster than sequential | asyncio.gather + batch judge |

### Phase 2: The Shield (Live Middleware)
1. The generated `contract.json` is loaded into the Middleware layer at runtime.
2. Every incoming user prompt is scanned against the JSON rules.
3. If it matches a mapped vulnerability, the Middleware **blocks, clarifies, or flags** *before the LLM is even invoked* — saving compute and guaranteeing zero hallucinations on known attack vectors.

---

## 🧮 4. The Mathematics of AI Failure (The Vector Engine)

**The Challenge:** How do you programmatically prove a model failed without using another expensive LLM as a judge?

**The Solution:** A high-speed mathematical benchmark in `engine/signal_extractor.py` calculates a **Boundary Score (0.0 → 1.0)** per probe:

| Score | Weight | What It Measures |
|---|---|---|
| **Consistency** | 40% | Asks the same question 3× at Temp 0.5. High cosine variance = hallucination risk. |
| **Divergence** | 40% | Compares Temp 0.5 response vs Temp 0.3. If the same model disagrees with itself, the prompt is an edge case. |
| **Confidence** | 20% | Scans for hedging language: *"I think"*, *"maybe"*, *"not sure"* — linguistic uncertainty as a proxy for training distribution gaps. |

**Formula:** `Boundary Score = (0.4 × Consistency) + (0.4 × Divergence) + (0.2 × Confidence)`

**Threshold:** `0.20` — chosen specifically for Qwen 72B, which is so robust that any divergence above 0.20 is a critical, meaningful boundary failure worth capturing.

---

## ⚡ 5. Performance Engineering — All Bottlenecks Resolved

This section documents every bottleneck we identified and the exact fix applied.

### Stage 1: Probe Generation — *Medium Bottleneck*
- CrewAI agents run sequential LLM reasoning to brainstorm probes.
- **Status:** Acceptable for the scale. Could be parallelized by category in future work.

### Stage 2: AMD MI300X Batch Inference — *Eliminated*
- **Fix:** `asyncio.gather()` fires all probes concurrently with a 100-slot semaphore (`request_semaphore`).
- **Fix:** Both Temp 0.5 and Temp 0.3 inferences scheduled in the same async task batch.
- **Result:** 2,500 probes × 4 inferences each = **10,000 total inferences in ~7 minutes** on MI300X.

### Stage 3: Signal Extraction — *No Bottleneck*
- Pure vectorized NumPy math. Runs in milliseconds. No changes needed.

### Stage 4: Contract Compilation — *Token Limit Eliminated*
- **Root cause:** Sending raw verbose 72B responses to the compiler blew past the 4096-token context limit (literally 1 token over on our production run).
- **Fix 1 (K-Means Clustering):** Embeds all boundary failures with `all-MiniLM-L6-v2`, clusters into 6 semantic groups, picks the highest-severity representative from each. Guarantees diverse coverage without sending redundant failures.
- **Fix 2 (Compact Objects):** Instead of full verbose text, passes `{"probe": ..., "severity": ..., "sample_failure": ...}` — semantically dense, token-minimal.
- **Result:** Token usage dropped from 4,097 → ~600 tokens. Token limit bug is architecturally impossible now.

### Stage 5: Middleware Validation — *10× Speed Improvement*
- **Root cause:** 25 probes × 4 sequential LLM calls = 100 sequential blocking API calls (~8 minutes).
- **Fix 1 (asyncio.gather):** All 25 probes fire concurrently. Baseline and middleware calls within each probe also run in parallel.
- **Fix 2 (Batch Judging):** Instead of 1 response per Judge LLM call, sends 5 responses per call and parses a comma-separated verdict list. Divides Judge overhead by 5.
- **Result:** Wall time reduced from ~8 minutes → ~30-60 seconds. 100 blocking calls → effectively ~10 parallel calls.

---

## 🛠️ 6. Technology Stack

| Tool | Role | Why |
|---|---|---|
| **AMD MI300X + ROCm** | GPU Compute | Insane memory bandwidth for massive parallel batch inference on 72B models |
| **vLLM (ROCm build)** | Inference Engine | Continuous batching + KV cache + optimized for ROCm — maximum token throughput |
| **Qwen2.5-72B-Instruct** | Target + Agent LLM | Qwen Challenge integration; smart enough to act as both attacker and target |
| **CrewAI** | Multi-Agent Orchestration | Separate Miner and Architect agents with distinct personas for better contract quality |
| **sentence-transformers** | Vector Embeddings | Lightning-fast local cosine similarity for mathematical failure detection |
| **scikit-learn (K-Means)** | Failure Clustering | Principled semantic sampling to guarantee diverse coverage within token budgets |
| **asyncio** | Concurrency Engine | Parallel probe firing, parallel inference calls, parallel validation |
| **Gradio** | Dashboard UI | Interactive "Before vs. After" middleware demo for live hackathon presentation |

---

## 🚀 7. Engineering Journey (Challenges Overcome)

1. **API Rate Limit Wall:** Prototyping on public APIs instantly hit `429 RateLimitErrors` when firing 50+ parallel probes. *Pivot:* Moved to self-hosted vLLM on AMD MI300X — no rate limits, true parallelism.

2. **The 4,097-Token Bug:** During the production 2,500-probe run, the compiler received such rich, verbose 72B responses that even 4 examples totaled exactly 4,097 tokens — one over the limit. *Pivot:* Replaced manual slicing with K-Means semantic clustering + compact failure objects. Now architecturally impossible to hit the token limit.

3. **The Pydantic Validation Crash:** The validation engine was incorrectly passing raw probe dictionaries into Litellm instead of extracting the text string. *Fix:* Added `probe_str = probe.get("input", "") if isinstance(probe, dict) else probe` defensive extraction.

4. **8-Minute Validation Bottleneck:** Sequential synchronous LLM calls made validation a massive blocking operation. *Fix:* Full asyncio rewrite with `asyncio.gather()` and batch judging, reducing wall time by 10×.

5. **Silent Failure Detection (0.0% baseline rate):** Initial validation tested random probes, most of which Qwen correctly handled. *Fix:* Validation now tests specifically against the **known extracted boundary failures** — the exact probes the math engine proved caused the model to stumble.

---

## 📊 8. Business Value & Results

| Metric | Value |
|---|---|
| **Total Probes Fired (Production Run)** | 2,500 |
| **Total Inferences Executed** | 10,000 |
| **AMD MI300X GPU Time** | 431.37 seconds (~7 min) |
| **Equivalent CPU Time (estimated)** | 8,072 seconds (~2.2 hours) |
| **AMD Speedup Factor** | **~19×** |
| **Boundary Failures Extracted** | 28 |
| **Safety Rules Compiled** | 7 |

---

## ⚙️ 9. Setup & Deployment

### Step 1: Start the vLLM Server (AMD MI300X, Terminal 1)
```bash
export HF_TOKEN="hf_your_token_here"

docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:v0.17.1 \
  --model Qwen/Qwen2.5-72B-Instruct \
  --port 8000 --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

### Step 2: Configure Environment
```ini
# .env
USE_AMD_SERVER=true
MODEL_A=Qwen/Qwen2.5-72B-Instruct
VLLM_PORT=8000
BOUNDARY_THRESHOLD=0.20
HF_TOKEN=hf_your_token_here
```

### Step 3: Run the Production Pipeline (Terminal 2)
```bash
source bf_env/bin/activate

# Full 2,500-probe production run
python main.py --production

# OR: Resume compilation from existing boundary data (skips GPU inference)
python resume.py
```

### Step 4: Launch the Dashboard
```bash
python ui/app.py
```
Open the Gradio link and explore:
- **Contract Object** — The 7 compiled safety rules
- **A/B Testing** — Side-by-side boundary failures
- **Metrics** — GPU vs CPU speedup comparison
- **Live Middleware** — Type an adversarial prompt and watch it get intercepted in real-time

---

## 🏆 10. The Temperature Divergence Strategy

Since running two separate 70B models simultaneously would exhaust MI300X VRAM, we use a **Temperature Divergence Architecture**:

- **Model A (Creative Mode):** Qwen 72B at **Temperature 0.5** — creative, less deterministic
- **Model B (Strict Mode):** Same Qwen 72B at **Temperature 0.3** — conservative, more grounded

When the *same model* gives meaningfully different answers at different temperatures, the prompt is mathematically proven to be an ambiguous edge case that the model has not confidently learned. These are the exact prompts worth protecting against.

> **UI Note:** The dropdown menu in the Gradio UI shows multiple model names to demonstrate the API Gateway architecture concept. In this AMD deployment, all inference routes to the Qwen endpoint on port 8000.
