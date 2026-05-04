import json
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from crews.tools import conflict_resolver_tool
from config import LOCAL_LLM_URL, LOCAL_API_KEY, TOP_RULES

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    base_url=LOCAL_LLM_URL,
    api_key=LOCAL_API_KEY,
    temperature=0.1
)


def run_compilation_crew(boundaries: list) -> list:
    cases_text = json.dumps(boundaries[:50], indent=2)

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
        tools=[conflict_resolver_tool],
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
Convert patterns into EXACTLY {TOP_RULES} rules.

Each rule MUST include:
- condition
- action
- trigger_phrases (min 3)
- rationale

STRICT JSON ONLY:
{{
  "rules": [...]
}}
""",
        expected_output="Valid JSON rules",
        agent=compiler
    )

    crew = Crew(
        agents=[miner, compiler],
        tasks=[mine_task, compile_task],
        process=Process.sequential,
        memory=True
    )

    result = crew.kickoff()
    text = str(result).strip()

    # Fix markdown parsing safely
    if "```" in text:
        text = text.split("```")[1].replace("json", "").strip()

    try:
        rules = json.loads(text).get("rules", [])
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
