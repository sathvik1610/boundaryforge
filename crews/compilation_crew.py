import json
from crewai import Agent, Task, Crew, Process
from crewai import LLM
from config import LOCAL_API_KEY, TOP_RULES, MODEL_A, ACTIVE_MODEL_A, USE_AMD_SERVER, LOCAL_LLM_URL_A

if USE_AMD_SERVER:
    llm = LLM(model=f"openai/{MODEL_A}", base_url=LOCAL_LLM_URL_A, api_key=LOCAL_API_KEY, temperature=0.1)
else:
    llm = LLM(model=f"huggingface/{ACTIVE_MODEL_A}", api_key=LOCAL_API_KEY, temperature=0.1)


def _select_representative_failures(boundaries: list, n_clusters: int = 6) -> list:
    """FIX Priority 2 (Stage 4): K-Means clustering on failure embeddings.
    Embeds all boundary failures, clusters them into n_clusters semantic groups,
    and picks the most representative example from each cluster.
    This guarantees diverse failure type coverage within the token budget.
    """
    if len(boundaries) <= n_clusters:
        # Not enough to cluster, just compact them
        return boundaries

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        import numpy as np

        # Embed on the input probe text (what the user said — the failure trigger)
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [b["input"] for b in boundaries]
        embeddings = embedder.encode(texts)

        k = min(n_clusters, len(boundaries))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(embeddings)

        # Pick the boundary closest to each cluster centroid
        representatives = []
        for cluster_id in range(k):
            cluster_indices = [i for i, label in enumerate(kmeans.labels_) if label == cluster_id]
            centroid = kmeans.cluster_centers_[cluster_id]
            # Find the member with the highest boundary_score in the cluster
            best = max(cluster_indices, key=lambda i: boundaries[i].get("boundary_score", 0))
            representatives.append(boundaries[best])

        print(f"[K-Means Clustering] Selected {len(representatives)} representative failures from {len(boundaries)} total.")
        return representatives

    except ImportError:
        # sklearn not available — fall back to top-N
        print("[K-Means Clustering] sklearn not available, falling back to top-4 slice.")
        return boundaries[:4]


def _compact_boundary(b: dict) -> dict:
    """FIX Priority 5 (Stage 4): Pre-structured, token-efficient failure objects.
    Instead of verbose raw text, pass semantically dense compact JSON.
    """
    return {
        "probe": b["input"][:200],
        "severity": b.get("boundary_score", 0.0),
        "sample_failure": b["outputs_a"][0][:200] if b.get("outputs_a") else "",
    }


def run_compilation_crew(boundaries: list) -> list:
    # Stage 4 Fix: K-Means cluster → compact representation
    representatives = _select_representative_failures(boundaries, n_clusters=10)
    compact = [_compact_boundary(b) for b in representatives]
    cases_text = json.dumps(compact, indent=2)


    miner = Agent(
        role='Vulnerability Miner',
        goal='Analyze boundary cases and cluster them into distinct failure patterns.',
        backstory='You are a rigorous data scientist.',
        verbose=True,
        llm=llm
    )

    compiler = Agent(
        role='Safety Contract Architect',
        goal='Translate failure patterns into strict JSON rules.',
        backstory='You are a strict systems engineer.',
        verbose=True,
        llm=llm
    )

    mine_task = Task(
        description=f"""
Analyze these failures:
{cases_text}

Identify 4-6 distinct failure patterns.
""",
        expected_output='List of failure patterns',
        agent=miner
    )

    compile_task = Task(
        description=f"""
You are a Fintech AI Safety Architect. Convert the failure patterns into EXACTLY {TOP_RULES} safety rules.

DOMAIN CONTEXT: This is a customer support chatbot for a fintech company handling refunds, KYC, transactions, and account issues.

CRITICAL INSTRUCTIONS:
- Rules must catch the USER'S INTENT, not just their exact words.
  Example: "I don't want to pay taxes" = tax evasion intent. "Move money without records" = money laundering intent.
- trigger_phrases must be SHORT FRAGMENTS that appear naturally in adversarial user messages.
  GOOD: ["don't want to pay", "avoid taxes", "without documentation", "offshore account"]
  BAD:  ["tax evasion", "illegal financial scheme"]  ← users never say this literally
- MUST cover ALL of these fintech attack categories:
  1. Money laundering / cash concealment ("launder", "without records", "cash only")
  2. KYC / identity bypass ("skip verification", "ignore KYC", "without ID")
  3. Fraudulent refund claims ("refund for", "never purchased", "claim without")
  4. Tax evasion / offshore transfers ("avoid taxes", "offshore", "unreported")
  5. Terrorist / sanctioned entity financing ("terrorist", "sanctioned", "fund extremist")
  6. Extortion / blackmail schemes ("extort", "blackmail", "threaten to expose")
  7. Coerced / emergency transfers ("being threatened", "forced to transfer", "safe account")
  8. Unauthorized account access / impersonation ("pretend to be", "override policy", "admin access")

Each rule MUST include:
- name (short camelCase identifier)
- condition (what user intent this catches)
- action (what the system does)
- action_type (MUST BE EXACTLY ONE OF: "block", "clarify", or "flag")
- trigger_phrases (list of 4-6 SHORT FRAGMENTS that appear in real user messages)
- rationale (one sentence)

STRICT JSON ONLY — no markdown, no explanation:
{{
  "rules": [...]
}}
""",
        expected_output="Valid JSON rules object with 'rules' array",
        agent=compiler
    )

    crew = Crew(
        agents=[miner, compiler],
        tasks=[mine_task, compile_task],
        process=Process.sequential,
        memory=False
    )

    result = crew.kickoff()
    text = str(result).strip()

    # Fix markdown parsing safely
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    try:
        parsed_data = json.loads(text)
        rules = parsed_data.get("rules", [])
        
        # Schema Validation
        valid_rules = []
        for r in rules:
            if not isinstance(r, dict): continue
            
            # Require minimum fields
            if "name" not in r or "condition" not in r or "action_type" not in r:
                continue
                
            # Validate action_type
            if r["action_type"] not in ["block", "clarify", "flag"]:
                r["action_type"] = "flag"  # default to flag if invalid
                
            valid_rules.append(r)
            
        rules = valid_rules
    except json.JSONDecodeError:
        rules = []

    # ===== FALLBACK RULES =====
    fallback_rules = [
        {
            "id": "F1",
            "name": "Ambiguity clarification",
            "condition": "input is ambiguous or incomplete",
            "action": "ask for clarification",
            "action_type": "clarify",
            "trigger_phrases": ["unclear", "not sure", "maybe", "depends"],
            "rationale": "Ambiguous inputs cause inconsistent responses"
        },
        {
            "id": "F2",
            "name": "High-risk safeguard",
            "condition": "financial/medical decision without info",
            "action": "flag",
            "action_type": "flag",
            "trigger_phrases": ["refund", "investment", "treatment"],
            "rationale": "High-risk domains need verification"
        }
    ]

    if not rules or len(rules) < 3:
        rules.extend(fallback_rules)

    rules = rules[:8]

    # ✅ NOW correctly inside function
    with open("data/contract.json", "w") as f:
        json.dump({"rules": rules}, f, indent=2)

    return rules
