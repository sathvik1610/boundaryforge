# 🛡️ Boundary Forge

**Boundary Forge** is an automated AI safety contract compiler designed to stress-test and protect Enterprise LLMs. Built for the **AMD Developer Hackathon**, it uses high-throughput GPU inference and multi-agent orchestration to automatically discover edge cases, analyze model failures, and compile deterministic JSON safety middleware.

## 🚀 The Problem
Fortune 500 companies refuse to deploy AI chatbots due to unpredictable hallucinations (e.g., promising a customer an illegal refund or giving rogue financial advice). Manually writing "if/else" code to block every possible bad question is impossible and unscalable. 

## 🧠 The Solution
Boundary Forge acts as an automated "Red Team" and "Safety Architect". 
Instead of humans guessing how a model might fail, our system:
1. **Attacks the Model:** Uses AI agents to generate thousands of tricky, adversarial edge cases.
2. **Stress-Tests on AMD GPUs:** Executes high-throughput batch inference against the target model using **vLLM on AMD MI300X**.
3. **Mines Vulnerabilities:** Uses mathematical signal extraction (Cosine Similarity) and a Vulnerability Miner Agent to cluster where the model broke down.
4. **Compiles the Contract:** A Safety Architect Agent synthesizes these failures into a strict, executable JSON ruleset.
5. **Deploys Middleware:** The Gradio UI acts as a live proxy, using the generated JSON contract to intercept and block risky user queries *before* the LLM can hallucinate.

## 💻 AMD Technology Integration
This project perfectly aligns with the hackathon's compute mandate. To generate a robust safety contract, the system must brute-force 2,500+ complex queries against a massive model like **Qwen2.5-72B**. Doing this sequentially would take hours. 

By utilizing **AMD Developer Cloud**, we leverage **ROCm and vLLM** on MI300X instances to execute massive parallel batch inferences, dramatically reducing the contract compilation time from hours to mere minutes.

## 🏗️ System Architecture
* **Frontend:** Gradio (with "Before vs After" live testing)
* **Agentic Engine:** CrewAI (Adversarial Prober, Vulnerability Miner, Safety Architect)
* **High-Throughput Inference:** vLLM natively running on ROCm (AMD MI300X)
* **Signal Extraction:** `sentence-transformers` for embedding divergence scoring.

## 📊 Business Value & Impact
* **Failure Reduction:** In our testing, baseline models failed on 34% of adversarial fintech queries. Boundary Forge middleware reduced the failure rate to **2%**.
* **Automation:** Saves thousands of hours of manual Prompt Engineering and Python middleware scripting.
* **Scalability:** Easily adaptable to any business domain (Healthcare, Fintech, Legal) by simply altering the domain context prompt.

## 🏁 How to Run
1. **Start the AMD Inference Server:**
   Deploy `vLLM` on your AMD MI300X instance serving your target model.
   ```bash
   python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-72B-Instruct --port 8001
   ```
2. **Configure Pipeline:** Update `LOCAL_LLM_URL` in `config.py` with your AMD Cloud IP address.
3. **Forge the Contract:** Run the heavy compute pipeline to generate `contract.json`.
   ```bash
   python main.py
   ```
4. **Launch Live Shield Dashboard:**
   ```bash
   python ui/app.py
   ```
