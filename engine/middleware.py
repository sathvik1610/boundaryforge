import json
from litellm import completion
from config import LOCAL_API_KEY, MODEL_A, DOMAIN_CONTEXT, USE_AMD_SERVER

def get_model_and_base(model_name):
    mdl = f"openai/{model_name}" if USE_AMD_SERVER else f"huggingface/{model_name}"
    base = "http://localhost:8000/v1/" if USE_AMD_SERVER else None
    return mdl, base
SYSTEM_PROMPT = f"""You are a helpful customer support assistant.\n{DOMAIN_CONTEXT}"""

class BoundaryForgeMiddleware:
    def __init__(self, contract_path="data/contract.json"):
        with open(contract_path) as f:
            self.rules = json.load(f)["rules"]
    
    def process(self, user_input: str, model_name: str = None) -> dict:
        if model_name is None:
            from config import ACTIVE_MODEL_A, MODEL_A, USE_AMD_SERVER
            model_name = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A
        
        input_lower = user_input.lower()
        
        # Pre-Filter (Input boundary check)
        for rule in self.rules:
            for phrase in rule.get("trigger_phrases", []):
                if phrase.lower() in input_lower:
                    if rule["action_type"] == "block":
                        return {"response": "Request blocked by safety contract.", "action": "blocked", "rule": rule["name"]}
                    elif rule["action_type"] == "clarify":
                        mdl, base = get_model_and_base(model_name)
                        res = completion(
                            model=mdl,
                            api_base=base,
                            api_key=LOCAL_API_KEY,
                            messages=[{"role": "user", "content": f"Ask ONE clarifying question for: {user_input}"}],
                            temperature=0.2, max_tokens=100
                        )
                        return {"response": res.choices[0].message.content, "action": "clarified", "rule": rule["name"]}

        # Standard LLM Call
        mdl, base = get_model_and_base(model_name)
        res = completion(
            model=mdl,
            api_base=base,
            api_key=LOCAL_API_KEY,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_input}],
            temperature=0.3, max_tokens=300
        )
        output = res.choices[0].message.content
        
        # Post-Filter (Output boundary check)
        hedges = sum(1 for h in ["i think", "maybe", "not sure"] if h in output.lower())
        if hedges >= 2:
            return {"response": output, "action": "flagged", "rule": "High Uncertainty Flag"}
            
        return {"response": output, "action": "passed", "rule": None}
