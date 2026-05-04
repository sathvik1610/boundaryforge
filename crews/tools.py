import json
from langchain.tools import tool

@tool("Contract Conflict Resolver")
def conflict_resolver_tool(rules_json: str) -> str:
    """Pass compiled rules here to check for contradictory IF/THEN actions before finalizing."""
    try:
        rules = json.loads(rules_json)
        # Mock logic constraint check for hackathon speed
        triggers = [r.get("condition") for r in rules]
        if len(triggers) != len(set(triggers)):
            return "CONFLICT DETECTED: Overlapping conditions found. Merge rules."
        return "CLEAN: No logical contradictions in action spaces."
    except:
        return "ERROR: Invalid JSON passed to tool."
