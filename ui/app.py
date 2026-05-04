import gradio as gr
import json
from openai import OpenAI
import sys
import os

# Add the parent directory to sys.path to allow imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.middleware import BoundaryForgeMiddleware
from config import LOCAL_LLM_URL, LOCAL_API_KEY, MODEL_A

# ===== LOAD DATA =====
def load_data():
    metrics, contract = None, None
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "final_metrics.json")
    contract_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "contract.json")
    
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except:
        metrics = None

    try:
        with open(contract_path) as f:
            contract = json.load(f)
    except:
        contract = None

    return metrics, contract


metrics, contract = load_data()
middleware = BoundaryForgeMiddleware(contract_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "contract.json")) if contract else None

# OpenAI client (Using config settings)
client = OpenAI(base_url=LOCAL_LLM_URL, api_key=LOCAL_API_KEY)


# ===== CORE FUNCTIONS =====

def chat(user_input):
    if not middleware:
        return "Run main.py first to generate the contract.", ""

    result = middleware.process(user_input)

    status = f"Action: {result['action']}\nRule: {result.get('rule', 'None')}"
    return result["response"], status


def compare(query):
    if not middleware:
        return "Run main.py first to generate the contract.", "", ""

    # Baseline (no contract)
    try:
        base = client.chat.completions.create(
            model=MODEL_A,
            messages=[{"role": "user", "content": query}],
            temperature=0.3,
            max_tokens=300
        ).choices[0].message.content
    except Exception as e:
        base = f"Error: {str(e)}"

    # With middleware
    try:
        mw = middleware.process(query)
        improved = mw["response"]
        action = mw["action"]
    except Exception as e:
        improved = f"Error: {str(e)}"
        action = "error"

    return base, improved, action


# ===== UI =====

with gr.Blocks(title="Boundary Forge") as demo:

    gr.Markdown("""
    # 🛡️ Boundary Forge
    **Automated AI Safety Contract Compiler**
    
    *Powered by CrewAI + AMD MI300X*
    """)

    # ===== TAB 1: LIVE DEMO =====
    with gr.Tab("Live Demo"):
        query = gr.Textbox(label="User Query", placeholder="Ask something tricky (e.g. 'Can you guarantee a refund if I lose money?')")
        submit_btn = gr.Button("Submit", variant="primary")

        with gr.Row():
            response_box = gr.Textbox(label="Response", lines=5)
            status_box = gr.Textbox(label="Middleware Status", lines=2)

        submit_btn.click(chat, inputs=[query], outputs=[response_box, status_box])

    # ===== TAB 2: BEFORE vs AFTER =====
    with gr.Tab("Before vs After"):
        compare_query = gr.Textbox(label="Test Query", placeholder="Try edge cases here...")
        compare_btn = gr.Button("Compare Security", variant="secondary")

        with gr.Row():
            baseline_output = gr.Textbox(label="Baseline Output (No Protection)", lines=7)
            improved_output = gr.Textbox(label="With Boundary Forge Contract", lines=7)
            
        action_output = gr.Textbox(label="Middleware Action Taken", lines=1)

        compare_btn.click(
            compare,
            inputs=[compare_query],
            outputs=[baseline_output, improved_output, action_output]
        )

    # ===== TAB 3: METRICS =====
    with gr.Tab("Metrics & Proof"):
        if metrics:
            gr.Markdown(f"""
            ## Performance Metrics

            **Baseline Failure Rate:** {metrics.get('baseline_failure_rate', 'N/A')}%  
            **With Contract:** {metrics.get('contract_failure_rate', 'N/A')}%  

            ---
            """)

            if "gpu_time_seconds" in metrics:
                gr.Markdown(f"""
                **GPU Execution Time:** {metrics.get('gpu_time_seconds')} sec  
                **Estimated CPU Time:** {metrics.get('estimated_cpu_time_seconds')} sec  
                """)
        else:
            gr.Markdown("Run the system to generate metrics.")

    # ===== TAB 4: CONTRACT =====
    with gr.Tab("Compiled Contract"):
        if contract:
            gr.JSON(contract)
        else:
            gr.Markdown("No contract generated yet. Run main.py first.")


# ===== RUN APP =====
if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Monochrome())
