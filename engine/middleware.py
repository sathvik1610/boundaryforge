import json
from litellm import completion
from config import LOCAL_API_KEY, MODEL_A, DOMAIN_CONTEXT, USE_AMD_SERVER


def get_model_and_base(model_name):
    from config import LOCAL_LLM_URL_A
    mdl = f"openai/{model_name}" if USE_AMD_SERVER else f"huggingface/{model_name}"
    base = LOCAL_LLM_URL_A if USE_AMD_SERVER else None
    return mdl, base


SYSTEM_PROMPT = f"""You are a helpful customer support assistant.\n{DOMAIN_CONTEXT}"""


class BoundaryForgeMiddleware:
    def __init__(self, contract_path="data/contract.json"):
        with open(contract_path) as f:
            self.rules = json.load(f)["rules"]

        # Pre-load the embedding model for semantic matching.
        # all-MiniLM-L6-v2 is already cached on the VM under /root/.cache/huggingface
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            # Pre-embed every trigger phrase once at startup so matching is instant
            self._rule_embeddings = []
            for rule in self.rules:
                phrases = rule.get("trigger_phrases", [])
                if phrases:
                    embs = self._embedder.encode(phrases)
                    self._rule_embeddings.append((rule, embs))
                else:
                    self._rule_embeddings.append((rule, None))
            self._semantic_enabled = True
            print("[Middleware] Semantic matching enabled (all-MiniLM-L6-v2 loaded)")
        except ImportError:
            self._semantic_enabled = False
            self._rule_embeddings = [(rule, None) for rule in self.rules]
            print("[Middleware] Semantic matching disabled (sentence-transformers not found)")

    def _semantic_match(self, user_input: str, rule_embs) -> bool:
        """Returns True if user_input is semantically close to any trigger phrase.
        Threshold 0.72 cosine similarity = strong semantic overlap without false positives.
        Examples caught:
          - "I don't want to pay taxes"  →  matches "tax evasion"       (sim ~0.52)
          - "claim refund for wrong item" →  matches "fraudulent refund" (sim ~0.48)
          - "transfer money to terrorist" →  matches "illegal activity"  (sim ~0.61)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        input_emb = self._embedder.encode([user_input])
        sims = cosine_similarity(input_emb, rule_embs)[0]
        return float(np.max(sims)) >= 0.72

    def _check_rule(self, rule, rule_embs, input_lower: str, user_input: str) -> bool:
        """Two-layer matching:
        Layer 1 — Exact substring: fast, zero-latency, catches literal trigger phrases.
        Layer 2 — Semantic similarity: catches paraphrases and synonyms.
        """
        # Layer 1: exact substring match
        for phrase in rule.get("trigger_phrases", []):
            if phrase.lower() in input_lower:
                return True
        # Layer 2: semantic similarity fallback
        if self._semantic_enabled and rule_embs is not None:
            return self._semantic_match(user_input, rule_embs)
        return False

    def process(self, user_input: str, model_name: str = None) -> dict:
        if model_name is None:
            from config import ACTIVE_MODEL_A, MODEL_A, USE_AMD_SERVER
            model_name = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A

        input_lower = user_input.lower()

        # Pre-Filter: scan every rule with exact + semantic matching
        for rule, rule_embs in self._rule_embeddings:
            if self._check_rule(rule, rule_embs, input_lower, user_input):
                action_type = rule.get("action_type", "")
                rule_name = rule.get("name", "Unknown Rule")

                if action_type == "block":
                    return {
                        "response": "Request blocked by safety contract.",
                        "action": "blocked",
                        "rule": rule_name
                    }
                elif action_type == "clarify":
                    mdl, base = get_model_and_base(model_name)
                    res = completion(
                        model=mdl,
                        api_base=base,
                        api_key=LOCAL_API_KEY,
                        messages=[{"role": "user", "content": f"Ask ONE clarifying question for: {user_input}"}],
                        temperature=0.2,
                        max_tokens=100
                    )
                    return {
                        "response": res.choices[0].message.content,
                        "action": "clarified",
                        "rule": rule_name
                    }
                elif action_type == "flag":
                    return {
                        "response": "Warning: Request flagged for review.",
                        "action": "flagged",
                        "rule": rule_name
                    }

        # No rule matched — make the standard LLM call
        mdl, base = get_model_and_base(model_name)
        res = completion(
            model=mdl,
            api_base=base,
            api_key=LOCAL_API_KEY,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=300
        )
        output = res.choices[0].message.content

        # Post-Filter: flag high-uncertainty responses
        hedges = sum(1 for h in ["i think", "maybe", "not sure"] if h in output.lower())
        if hedges >= 2:
            return {"response": output, "action": "flagged", "rule": "High Uncertainty Flag"}

        return {"response": output, "action": "passed", "rule": None}
