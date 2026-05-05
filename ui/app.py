import gradio as gr
import json
from litellm import completion
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.middleware import BoundaryForgeMiddleware
from config import LOCAL_API_KEY, MODEL_A


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
middleware = BoundaryForgeMiddleware(
    contract_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "contract.json")
) if contract else None

def get_model_and_base():
    """Always routes to the active model. Dropdown is display-only (API Gateway framing)."""
    from config import USE_AMD_SERVER, ACTIVE_MODEL_A, MODEL_A, VLLM_PORT
    # On AMD: use the production 72B model served by vLLM
    # On local: use the smaller fast model on HF free tier
    mdl = f"openai/{MODEL_A}" if USE_AMD_SERVER else f"huggingface/{ACTIVE_MODEL_A}"
    base = f"http://localhost:{VLLM_PORT}/v1/" if USE_AMD_SERVER else None
    return mdl, base


# ===== CORE FUNCTIONS =====
def chat(user_input, model_name):
    if not middleware:
        return "Run main.py first to generate the contract.", ""
    from config import ACTIVE_MODEL_A, MODEL_A, USE_AMD_SERVER
    active = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A
    result = middleware.process(user_input, model_name=active)
    status = f"Action: {result['action']}\nRule: {result.get('rule', 'None')}"
    return result["response"], status


def compare(query, model_name):
    if not middleware:
        return "Run main.py first to generate the contract.", "", ""
    try:
        from engine.middleware import SYSTEM_PROMPT
        mdl, base = get_model_and_base()
        res = completion(
            model=mdl,
            api_base=base,
            api_key=LOCAL_API_KEY,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
            temperature=0.3,
            max_tokens=300
        )
        base_text = res.choices[0].message.content
    except Exception as e:
        base_text = f"Error: {str(e)}"
    from config import ACTIVE_MODEL_A, MODEL_A, USE_AMD_SERVER
    active = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A
    try:
        mw = middleware.process(query, model_name=active)
        improved = mw["response"]
        action = mw["action"]
    except Exception as e:
        improved = f"Error: {str(e)}"
        action = "error"
    return base_text, improved, action


# ===== THEME =====
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
    primary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="#F7F8FC",
    body_background_fill_dark="#F7F8FC",
    body_text_color="#0F172A",
    body_text_color_dark="#0F172A",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E8EAF0",
    block_border_color_dark="#E8EAF0",
    block_radius="10px",
    block_shadow="0 1px 3px rgba(0,0,0,0.05)",
    block_label_background_fill="#FFFFFF",
    block_label_background_fill_dark="#FFFFFF",
    block_label_text_color="#6366F1",
    block_label_text_color_dark="#6366F1",
    block_label_text_size="11px",
    block_label_text_weight="600",
    block_title_text_color="#0F172A",
    block_title_text_color_dark="#0F172A",
    button_primary_background_fill="#6366F1",
    button_primary_background_fill_dark="#6366F1",
    button_primary_background_fill_hover="#4F46E5",
    button_primary_background_fill_hover_dark="#4F46E5",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color="#6366F1",
    button_primary_border_color_dark="#6366F1",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover="#F1F5F9",
    button_secondary_background_fill_hover_dark="#F1F5F9",
    button_secondary_text_color="#475569",
    button_secondary_text_color_dark="#475569",
    button_secondary_border_color="#E2E8F0",
    button_secondary_border_color_dark="#E2E8F0",
    input_background_fill="#FAFBFD",
    input_background_fill_dark="#FAFBFD",
    input_border_color="#E2E8F0",
    input_border_color_dark="#E2E8F0",
    input_border_color_focus="#6366F1",
    input_border_color_focus_dark="#6366F1",
    input_placeholder_color="#CBD5E1",
    input_placeholder_color_dark="#CBD5E1",
    table_even_background_fill="#FFFFFF",
    table_even_background_fill_dark="#FFFFFF",
    table_odd_background_fill="#F8FAFC",
    table_odd_background_fill_dark="#F8FAFC",
    color_accent="#6366F1",
    color_accent_soft="#EEF2FF",
    color_accent_soft_dark="#EEF2FF",
)

# ===== CSS =====
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=Fraunces:ital,opsz,wght@0,9..144,700;1,9..144,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container { background-color: #F7F8FC !important; color: #0F172A !important; }

/* GRADIO UI ICONS */
.icon, button.icon, .settings-icon { color: #64748B !important; background: transparent !important; }
.icon:hover, button.icon:hover, .settings-icon:hover { color: #0F172A !important; }

/* FORCE DROPDOWN MENU TO BE WHITE */
.dark .options, .dark ul.options, .dark .secondary-wrap {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
.dark .options *, .dark ul.options * {
    color: #0F172A !important;
}
.dark .option:hover, .dark li.option:hover {
    background-color: #F8FAFC !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem 5rem !important;
}

/* HEADER */
.bf-header {
    padding: 3.75rem 0 2.5rem;
    margin-bottom: 1.75rem;
    border-bottom: 1px solid #E2E8F0;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
    flex-wrap: wrap;
}
.bf-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #6366F1;
    margin: 0 0 0.55rem;
}
.bf-wordmark {
    font-family: 'Fraunces', serif;
    font-size: 3.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0F172A;
    margin: 0;
    line-height: 1;
}
.bf-wordmark em {
    font-style: italic;
    font-weight: 300;
    color: #6366F1;
}
.bf-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    color: #94A3B8;
    margin: 0.55rem 0 0;
    line-height: 1.5;
}
.bf-badge {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 100px;
    padding: 0.4rem 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    color: #4F46E5;
    margin-bottom: 0.2rem;
}
.bf-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #6366F1;
    animation: bfpulse 2.2s ease-in-out infinite;
}
@keyframes bfpulse {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.4; transform:scale(0.8); }
}

/* TABS */
.tab-nav {
    background: transparent !important;
    border-bottom: 1.5px solid #E2E8F0 !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
    gap: 0 !important;
}
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.7rem 1.2rem !important;
    margin-bottom: -1.5px !important;
    transition: color 0.15s !important;
}
.tab-nav button:hover, button[role="tab"]:hover { color: #475569 !important; background-color: transparent !important; }
.tab-nav button.selected, button[role="tab"][aria-selected="true"] {
    color: #6366F1 !important;
    font-weight: 600 !important;
    border-bottom-color: #6366F1 !important;
    background: transparent !important;
}

/* BLOCKS */
.gr-block, .gr-group, .gr-form {
    background: #FFFFFF !important;
    border: 1px solid #E8EAF0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.gr-group { padding: 1.2rem !important; margin-bottom: 1rem !important; }
.gr-row { gap: 12px !important; }

/* LABELS */
label span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #94A3B8 !important;
}

/* INPUTS */
textarea, input[type="text"] {
    background: #FAFBFD !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    background: #FFFFFF !important;
    outline: none !important;
}
textarea::placeholder, input::placeholder { color: #CBD5E1 !important; }

/* BUTTONS */
button.primary {
    background: #6366F1 !important;
    border: none !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.62rem 1.5rem !important;
    box-shadow: 0 1px 2px rgba(99,102,241,0.3), 0 4px 12px rgba(99,102,241,0.18) !important;
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
}
button.primary:hover {
    background: #4F46E5 !important;
    box-shadow: 0 2px 4px rgba(79,70,229,0.35), 0 8px 20px rgba(79,70,229,0.22) !important;
    transform: translateY(-1px) !important;
}
button.primary:active { transform: translateY(0) !important; }

button.secondary {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.62rem 1.5rem !important;
    transition: background 0.15s, border-color 0.15s !important;
}
button.secondary:hover {
    background: #F8FAFC !important;
    border-color: #CBD5E1 !important;
}

/* DROPDOWN */
select {
    background: #FAFBFD !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #334155 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

/* JSON */
.gr-json {
    background: #FAFBFD !important;
    border: 1px solid #E8EAF0 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
}

/* METRICS */
.mrow {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 14px;
}
.mcard {
    background: #FFFFFF;
    border: 1px solid #E8EAF0;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.mcard.d { border-top: 3px solid #F87171; }
.mcard.s { border-top: 3px solid #34D399; }
.mtag {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.45rem;
}
.mnum {
    font-family: 'Fraunces', serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
}
.mnum.d { color: #EF4444; }
.mnum.s { color: #10B981; }
.msub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.77rem;
    color: #94A3B8;
    margin-top: 0.3rem;
}
.gcard {
    background: linear-gradient(135deg, #EEF2FF 0%, #F0F9FF 100%);
    border: 1px solid #C7D2FE;
    border-radius: 12px;
    padding: 1.2rem 1.75rem;
}
.gtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 10px;
}
.grow {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.87rem;
    padding: 5px 0;
    border-bottom: 1px solid rgba(99,102,241,0.1);
}
.grow:last-child { border-bottom: none; }
.gkey { color: #64748B; }
.gval { color: #1E293B; font-weight: 600; }

/* COMPARISON */
.chead {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 7px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.cdot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.chead.u { color: #EF4444; }
.chead.u .cdot { background: #FCA5A5; }
.chead.p { color: #10B981; }
.chead.p .cdot { background: #6EE7B7; }

/* EMPTY */
.empty {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: #CBD5E1;
    text-align: center;
    padding: 3.5rem 2rem;
    border: 1.5px dashed #E2E8F0;
    border-radius: 10px;
    background: #FAFBFD;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
"""

# Force Light Mode JS to kill all color bugs
js_func = """
function() {
    const applyLightMode = () => {
        if (document.body) document.body.classList.remove('dark');
        if (document.documentElement) document.documentElement.classList.remove('dark');
        const gc = document.querySelector('.gradio-container');
        if (gc) gc.classList.remove('dark');
    };
    applyLightMode();
    setTimeout(applyLightMode, 100);
    setTimeout(applyLightMode, 1000);
    const observer = new MutationObserver(applyLightMode);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
}
"""

# ===== BUILD UI =====
with gr.Blocks(title="Boundary Forge") as demo:

    gr.HTML("""
    <div class="bf-header">
        <div>
            <p class="bf-eyebrow">Enterprise AI Safety</p>
            <h1 class="bf-wordmark">Boundary<em>Forge</em></h1>
            <p class="bf-desc">Safety contract compiler for production LLM deployments</p>
        </div>
        <div class="bf-badge">
            <div class="bf-dot"></div>
            AMD MI300X &middot; Active
        </div>
    </div>
    """)

    with gr.Group():
        model_dropdown = gr.Dropdown(
            choices=[
                "Qwen/Qwen2.5-72B-Instruct",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.3",
                MODEL_A
            ],
            value=MODEL_A,
            label="Target Engine",
            interactive=False,
        )

    with gr.Tabs():

        with gr.Tab("Live Middleware"):
            with gr.Group():
                query = gr.Textbox(
                    label="User Query",
                    placeholder="Enter a potentially adversarial prompt…",
                    lines=3,
                )
                submit_btn = gr.Button("Submit to Middleware", variant="primary")
            with gr.Row():
                with gr.Column(scale=2):
                    response_box = gr.Textbox(label="AI Response", lines=8, interactive=False)
                with gr.Column(scale=1):
                    status_box = gr.Textbox(label="Middleware Status", lines=4, interactive=False)
            submit_btn.click(chat, inputs=[query, model_dropdown], outputs=[response_box, status_box])

        with gr.Tab("A / B Testing"):
            with gr.Group():
                compare_query = gr.Textbox(
                    label="Test Query",
                    placeholder="Enter adversarial edge cases here…",
                    lines=3,
                )
                compare_btn = gr.Button("Run Comparison", variant="secondary")
            action_output = gr.Textbox(label="Middleware Action Taken", lines=1, interactive=False)
            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="chead u"><div class="cdot"></div>Baseline — Unprotected</div>')
                    baseline_output = gr.Textbox(label="", lines=9, interactive=False, show_label=False)
                with gr.Column():
                    gr.HTML('<div class="chead p"><div class="cdot"></div>Boundary Forge — Protected</div>')
                    improved_output = gr.Textbox(label="", lines=9, interactive=False, show_label=False)
            compare_btn.click(
                compare,
                inputs=[compare_query, model_dropdown],
                outputs=[baseline_output, improved_output, action_output],
            )

        with gr.Tab("Metrics"):
            if metrics:
                br   = metrics.get("baseline_failure_rate", 0)
                cr   = metrics.get("contract_failure_rate", "N/A")
                ir   = metrics.get("interception_rate", "N/A")
                efr  = metrics.get("effective_failure_rate", "N/A")
                nrm  = metrics.get("never_reach_model_pct", "N/A")
                n_int = metrics.get("middleware_intercepted", "N/A")
                n_tot = metrics.get("boundaries_found", "N/A")
                n_fired = metrics.get("total_probes_fired", "N/A")
                reduction = round((1 - efr / br) * 100, 1) if br and isinstance(efr, (int, float)) else "N/A"

                gpu_html = ""
                if "gpu_time_seconds" in metrics:
                    gpu  = metrics.get("gpu_time_seconds", 0)
                    cpu  = metrics.get("estimated_cpu_time_seconds", 0)
                    speedup = round(cpu / gpu, 1) if gpu else "N/A"
                    gpu_html = f"""
                    <div class="gcard" style="margin-top:1.2rem;">
                        <div class="gtitle">Compute Acceleration &mdash; AMD MI300X</div>
                        <div class="grow"><span class="gkey">Total Probes Fired</span><span class="gval">{n_fired:,}</span></div>
                        <div class="grow"><span class="gkey">GPU Execution Time</span><span class="gval">{gpu:.1f}s &nbsp;(~{round(gpu/60,1)} min)</span></div>
                        <div class="grow"><span class="gkey">Equivalent CPU Time</span><span class="gval">{cpu:.0f}s &nbsp;(~{round(cpu/3600,1)} hrs)</span></div>
                        <div class="grow"><span class="gkey">AMD MI300X Speedup</span><span class="gval" style="color:#6366F1;font-weight:700;">{speedup}× faster</span></div>
                        <div class="grow"><span class="gkey">Backend</span><span class="gval">vLLM on ROCm</span></div>
                    </div>"""

                gr.HTML(f"""
                <!-- Row 1: Key before/after rates -->
                <div class="mrow">
                    <div class="mcard d">
                        <div class="mtag">Baseline Failure Rate</div>
                        <div class="mnum d">{br}%</div>
                        <div class="msub">Unprotected — {n_fired:,} probes fired</div>
                    </div>
                    <div class="mcard s">
                        <div class="mtag">Effective Failure Rate</div>
                        <div class="mnum s">{efr}%</div>
                        <div class="msub">After Boundary Forge contract</div>
                    </div>
                </div>

                <!-- Row 2: What the contract does -->
                <div class="mrow" style="margin-top:1rem;">
                    <div class="mcard s">
                        <div class="mtag">Attack Interception Rate</div>
                        <div class="mnum s">{ir}%</div>
                        <div class="msub">{n_int} of {n_tot} known attacks blocked</div>
                    </div>
                    <div class="mcard s">
                        <div class="mtag">Failure Reduction</div>
                        <div class="mnum s">{reduction}%</div>
                        <div class="msub">Fewer failures with Boundary Forge</div>
                    </div>
                </div>

                <!-- Row 3: System impact -->
                <div class="gcard" style="margin-top:1.2rem;">
                    <div class="gtitle">System-Level Impact</div>
                    <div class="grow"><span class="gkey">Attacks that NEVER reach the model</span><span class="gval" style="color:#059669;font-weight:700;">{nrm}% of all enterprise traffic</span></div>
                    <div class="grow"><span class="gkey">Boundaries discovered</span><span class="gval">{n_tot} attack vectors from {n_fired:,} probes</span></div>
                    <div class="grow"><span class="gkey">Contract miss rate</span><span class="gval">{cr}%</span></div>
                </div>

                {gpu_html}
                """)
            else:
                gr.HTML('<div class="empty">Run main.py to generate performance metrics</div>')

        with gr.Tab("Contract Object"):
            if contract:
                gr.JSON(contract)
            else:
                gr.HTML('<div class="empty">No contract generated yet — run main.py first</div>')


if __name__ == "__main__":
    try:
        demo.launch(server_name="0.0.0.0", share=True, theme=theme, css=css, js=js_func)
    except TypeError:
        # Fallback if older gradio doesn't accept theme in launch
        demo.launch(server_name="0.0.0.0", share=True)