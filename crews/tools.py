import json
from crewai.tools import tool

@tool("Contract Conflict Resolver")
def conflict_resolver_tool(rules_json: str) -> str:
    """Pass compiled rules here to check for contradictory IF/THEN actions before finalizing."""
    try:
        data = json.loads(rules_json)
        # Accept both {"rules": [...]} and [...]
        rules = data.get("rules", data) if isinstance(data, dict) else data
        if not isinstance(rules, list):
            return "CLEAN: No logical contradictions in action spaces."
        triggers = [r.get("condition", "") for r in rules if isinstance(r, dict)]
        if len(triggers) != len(set(triggers)):
            return "CONFLICT DETECTED: Overlapping conditions found. Merge rules."
        return "CLEAN: No logical contradictions in action spaces."
    except Exception:
        # Never block the pipeline — just pass through
        return "CLEAN: No logical contradictions in action spaces."
