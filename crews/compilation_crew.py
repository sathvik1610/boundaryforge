"""
compilation_crew.py — CrewAI Safety Contract Compilation

Changes vs original:
  C1: mine_task forces strict compact JSON output (prevents context blowup)
  C2: compile_task explicitly consumes the JSON patterns from mine_task
  C3: run_compilation_crew() returns (rules, timing_dict)
  C4: Strict schema validation on every rule + ONE retry if count < TOP_RULES
"""

import json
import time

from crewai import Agent, Task, Crew, Process, LLM
from config import (
    LOCAL_API_KEY, TOP_RULES,
    MODEL_A, ACTIVE_MODEL_A, USE_AMD_SERVER, LOCAL_LLM_URL_A,
    # MAX_COMPILER_INPUT is used by main.py/resume.py to slice boundaries before
    # calling this function — not needed inside compilation logic itself.
)

if USE_AMD_SERVER:
    llm = LLM(
        model=f"openai/{MODEL_A}",
        base_url=LOCAL_LLM_URL_A,
        api_key=LOCAL_API_KEY,
        temperature=0.1,
    )
else:
    llm = LLM(
        model=f"huggingface/{ACTIVE_MODEL_A}",
        api_key=LOCAL_API_KEY,
        temperature=0.1,
    )


# ── K-Means clustering for representative failure selection ──────────────────

def _select_representative_failures(boundaries: list, n_clusters: int = 15) -> list:
    """K-Means clustering on failure embeddings.
    Picks the highest-scoring boundary nearest each cluster centroid.
    Guarantees diverse failure type coverage within the compiler token budget.
    """
    if len(boundaries) <= n_clusters:
        return boundaries

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        import numpy as np

        embedder   = SentenceTransformer('all-MiniLM-L6-v2')
        texts      = [b["input"] for b in boundaries]
        embeddings = embedder.encode(texts)

        k      = min(n_clusters, len(boundaries))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(embeddings)

        representatives = []
        for cluster_id in range(k):
            cluster_indices = [
                i for i, label in enumerate(kmeans.labels_) if label == cluster_id
            ]
            best = max(cluster_indices, key=lambda i: boundaries[i].get("risk_score", boundaries[i].get("boundary_score", 0)))
            representatives.append(boundaries[best])

        print(f"[K-Means] Selected {len(representatives)} representatives "
              f"from {len(boundaries)} boundaries.")
        return representatives

    except Exception as e:
        print(f"[K-Means] Clustering unavailable ({type(e).__name__}: {e}). "
              f"Falling back to top-{n_clusters} score slice.")
        return boundaries[:n_clusters]


def _compact_boundary(b: dict) -> dict:
    """Token-efficient compact representation of a boundary failure."""
    return {
        "probe":          b["input"][:200],
        "severity":       b.get("risk_score", b.get("boundary_score", 0.0)),
        "sample_failure": b["outputs_a"][0][:200] if b.get("outputs_a") else "",
    }


# ── Schema validation (C4) ───────────────────────────────────────────────────

REQUIRED_FIELDS = {"name", "condition", "action", "action_type", "trigger_phrases", "rationale"}
VALID_ACTION_TYPES = {"block", "clarify", "flag"}


def _validate_rules(rules: list) -> list:
    """Filter rules to those passing strict schema validation (C4).
    - All required fields present
    - action_type in {block, clarify, flag}
    - trigger_phrases is a non-empty list
    """
    valid = []
    for r in rules:
        if not isinstance(r, dict):
            continue
        missing = REQUIRED_FIELDS - r.keys()
        if missing:
            print(f"  [Schema] Rule '{r.get('name','?')}' missing fields: {missing} — skipped")
            continue
        if r["action_type"] not in VALID_ACTION_TYPES:
            print(f"  [Schema] Rule '{r.get('name','?')}' has invalid action_type "
                  f"'{r['action_type']}' — defaulting to 'flag'")
            r["action_type"] = "flag"
        phrases = r.get("trigger_phrases", [])
        if not isinstance(phrases, list) or len(phrases) == 0:
            print(f"  [Schema] Rule '{r.get('name','?')}' has empty trigger_phrases — skipped")
            continue
        valid.append(r)
    return valid


# ── Fallback rules (used if LLM cannot produce enough valid rules) ───────────

FALLBACK_RULES = [
    {
        "id": "F1",
        "name": "AmbiguityClarification",
        "condition": "input is ambiguous or incomplete",
        "action": "ask for clarification",
        "action_type": "clarify",
        "trigger_phrases": ["unclear", "not sure", "maybe", "depends on"],
        "rationale": "Ambiguous inputs cause inconsistent responses.",
    },
    {
        "id": "F2",
        "name": "HighRiskSafeguard",
        "condition": "financial decision without sufficient information",
        "action": "flag for review",
        "action_type": "flag",
        "trigger_phrases": ["refund", "investment", "transaction dispute", "large transfer"],
        "rationale": "High-risk financial domains need verification before action.",
    },
]


# ── Crew kickoff helper ───────────────────────────────────────────────────────

def _run_crew_once(cases_text: str) -> list:
    """Run one CrewAI kickoff and return parsed, validated rules."""

    miner = Agent(
        role="Vulnerability Miner",
        goal="Analyze boundary failure cases and cluster them into distinct failure patterns.",
        backstory="You are a rigorous data scientist specializing in AI safety evaluation.",
        verbose=True,
        llm=llm,
    )

    compiler = Agent(
        role="Safety Contract Architect",
        goal="Translate failure patterns into strict JSON safety rules.",
        backstory="You are a strict systems engineer building production AI safety contracts.",
        verbose=True,
        llm=llm,
    )

    # C1: mine_task forces strict compact JSON to prevent context blowup.
    # Bounded output (~600-800 tokens) keeps compile_task context well within 8192.
    mine_task = Task(
        description=f"""
Analyze these boundary failure cases from a fintech customer support AI:
{cases_text}

Return ONLY valid JSON. No markdown. No prose. No explanation.

Format:
{{
  "patterns": [
    {{
      "pattern": "short camelCase name",
      "risk": "one sentence describing the safety risk",
      "example_probe": "one short example user message",
      "recommended_rule_intent": "one sentence on what the rule should do"
    }}
  ]
}}

Constraints:
- Identify 10 to 15 distinct failure patterns.
- Each field must be a single short string.
- Total output MUST be under 1200 tokens.
- Do NOT output prose, headers, or markdown.
""",
        expected_output=(
            'Valid JSON object with "patterns" array. '
            '10-15 entries. No markdown. No prose.'
        ),
        agent=miner,
    )

    # C2: compile_task explicitly consumes the structured JSON from mine_task.
    compile_task = Task(
        description=f"""
You are a Fintech AI Safety Architect.

The Vulnerability Miner above has identified failure patterns in JSON format.
Use those patterns to compile EXACTLY {TOP_RULES} safety rules.

DOMAIN: Customer support chatbot for a fintech company.
Handles: refunds, KYC verification, transaction disputes, account issues.

CRITICAL INSTRUCTIONS:
- Rules must catch USER INTENT, not just exact words.
  Example: "I don't want to pay taxes" = tax evasion intent.
- trigger_phrases must be SHORT FRAGMENTS that appear in real user messages.
  GOOD: ["don't want to pay", "avoid taxes", "without documentation"]
  BAD:  ["tax evasion scheme"] — users never say this literally.
- Cover ALL of these fintech attack categories:
  1. Money laundering / cash concealment
  2. KYC / identity bypass
  3. Fraudulent refund claims
  4. Tax evasion / offshore transfers
  5. Terrorist / sanctioned entity financing
  6. Extortion / blackmail schemes
  7. Coerced / emergency transfers
  8. Unauthorized account access / impersonation

Each rule MUST include:
- name        (short camelCase identifier)
- condition   (what user intent this catches)
- action      (what the system does)
- action_type (EXACTLY ONE OF: "block", "clarify", or "flag")
- trigger_phrases (list of 4-6 SHORT FRAGMENTS from real user messages)
- rationale   (one sentence)

STRICT JSON ONLY — no markdown, no explanation:
{{
  "rules": [...]
}}
""",
        expected_output=f"Valid JSON with 'rules' array containing exactly {TOP_RULES} rules.",
        agent=compiler,
    )

    crew = Crew(
        agents=[miner, compiler],
        tasks=[mine_task, compile_task],
        process=Process.sequential,
        memory=False,
    )

    result = crew.kickoff()
    text   = str(result).strip()

    # Strip markdown code fences if LLM wraps output
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    try:
        parsed = json.loads(text)
        rules  = parsed.get("rules", [])
    except json.JSONDecodeError:
        rules = []

    return _validate_rules(rules)


# ── Public entry point ────────────────────────────────────────────────────────

def run_compilation_crew(boundaries: list) -> tuple:
    """Compile a safety contract from boundary failures.

    C4: Validates schema of every rule. If valid count < TOP_RULES,
        retries the crew kickoff ONCE before falling back to hardcoded rules.

    Returns:
        (rules, timing_dict)
    """
    t_start = time.time()

    # K-Means cluster → compact representation
    n_clusters     = min(15, len(boundaries))
    representatives = _select_representative_failures(boundaries, n_clusters=n_clusters)
    compact        = [_compact_boundary(b) for b in representatives]
    cases_text     = json.dumps(compact, indent=2)

    print(f"[Compiler] Running crew with {len(representatives)} representative failures...")

    # First attempt
    rules = _run_crew_once(cases_text)
    print(f"[Compiler] Attempt 1: {len(rules)} valid rules.")

    # C4: Retry once if below target
    if len(rules) < TOP_RULES:
        print(f"[Compiler] ⚠️  Only {len(rules)}/{TOP_RULES} valid rules. Retrying compilation...")
        retry_rules = _run_crew_once(cases_text)
        print(f"[Compiler] Attempt 2: {len(retry_rules)} valid rules.")
        if len(retry_rules) > len(rules):
            rules = retry_rules

    # Issue #9: Extend with fallbacks toward TOP_RULES, not just below 3.
    # If LLM returned 4 valid rules, extend rather than leaving a thin contract.
    if len(rules) < TOP_RULES:
        needed = TOP_RULES - len(rules)
        filling = FALLBACK_RULES[:needed]
        print(f"[Compiler] ⚠️  {len(rules)}/{TOP_RULES} valid rules. "
              f"Padding with {len(filling)} fallback rule(s).")
        rules.extend(filling)

    rules = rules[:TOP_RULES]

    if len(rules) < TOP_RULES:
        # P2: In production (TOP_RULES > 5), hard-fail if the contract is severely thin.
        # Do not allow padding to mask a complete compiler failure.
        if len(rules) < max(1, TOP_RULES // 2) and TOP_RULES >= 10:
            raise ValueError(
                f"CRITICAL FAILURE: Contract has only {len(rules)} rules, less than half "
                f"the target of {TOP_RULES}. The compiler failed to generate enough quality "
                f"rules and fallbacks are exhausted. Aborting before validation produces "
                f"misleading safety scores."
            )
            
        print(
            f"[Compiler] ⚠️  WARNING: Final contract has only {len(rules)}/{TOP_RULES} rules. "
            f"LLM + fallbacks ({len(FALLBACK_RULES)} available) could not reach target. "
            f"Contract will be THIN. Consider re-running resume.py or lowering TOP_RULES."
        )
    print(f"[Compiler] Final contract: {len(rules)} rules.")


    # Save contract.json
    with open("data/contract.json", "w", encoding="utf-8") as f:
        json.dump({"rules": rules}, f, indent=2)

    compilation_seconds = round(time.time() - t_start, 2)
    timing = {
        "compilation_seconds":  compilation_seconds,
        "rules_compiled":       len(rules),
        "representatives_used": len(representatives),
        "boundaries_received":  len(boundaries),
    }

    print(f"[Compiler] ✅ {len(rules)} rules compiled in {compilation_seconds}s")
    return rules, timing
