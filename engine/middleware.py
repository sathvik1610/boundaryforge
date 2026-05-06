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

    def _semantic_match(self, user_input: str, rule_embs) -> tuple:
        """Returns (matched: bool, score: float).
        Base detection threshold is 0.48. Tiered enforcement is applied in process():
          score >= 0.65  →  full rule action (block / clarify / flag)
          score >= 0.48  →  soft flag regardless of rule action_type
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        input_emb = self._embedder.encode([user_input])
        sims = cosine_similarity(input_emb, rule_embs)[0]
        score = float(np.max(sims))
        return score >= 0.48, score

    def _check_rule(self, rule, rule_embs, input_lower: str, user_input: str) -> tuple:
        """Two-layer matching. Returns (matched: bool, score: float, layer: str).
        Layer 1 — Exact substring: fast, zero-latency, catches literal trigger phrases.
        Layer 2 — Semantic similarity: catches paraphrases and synonyms.
        """
        # Layer 1: exact substring match
        for phrase in rule.get("trigger_phrases", []):
            if phrase.lower() in input_lower:
                return True, 1.0, "Exact Match"
        # Layer 2: semantic similarity fallback
        if self._semantic_enabled and rule_embs is not None:
            matched, score = self._semantic_match(user_input, rule_embs)
            if matched:
                return True, score, "Semantic Similarity"
        return False, 0.0, "None"

    def process(self, user_input: str, model_name: str = None) -> dict:
        if model_name is None:
            from config import ACTIVE_MODEL_A, MODEL_A, USE_AMD_SERVER
            model_name = MODEL_A if USE_AMD_SERVER else ACTIVE_MODEL_A

        input_lower = user_input.lower()

        # Pre-Filter: scan every rule with exact + semantic matching
        for rule, rule_embs in self._rule_embeddings:
            matched, score, layer = self._check_rule(rule, rule_embs, input_lower, user_input)
            if matched:
                action_type = rule.get("action_type", "")
                rule_name = rule.get("name", "Unknown Rule")
                rationale = rule.get("rationale", "")

                # Tiered enforcement: semantic hits below 0.65 are soft-flagged
                effective_action = action_type
                if layer == "Semantic Similarity" and score < 0.65:
                    effective_action = "flag"

                if effective_action == "block":
                    return {
                        "response": "Request blocked by safety contract.",
                        "action": "blocked",
                        "rule": rule_name,
                        "similarity_score": round(score, 3),
                        "match_layer": layer,
                        "rationale": rationale,
                    }
                elif effective_action == "clarify":
                    try:
                        mdl, base = get_model_and_base(model_name)
                        res = completion(
                            model=mdl,
                            api_base=base,
                            api_key=LOCAL_API_KEY,
                            messages=[{"role": "user", "content": f"Ask ONE clarifying question for: {user_input}"}],
                            temperature=0.2,
                            max_tokens=100
                        )
                        clarify_text = res.choices[0].message.content
                    except Exception:
                        clarify_text = "Could you provide more context about your request?"
                    return {
                        "response": clarify_text,
                        "action": "clarified",
                        "rule": rule_name,
                        "similarity_score": round(score, 3),
                        "match_layer": layer,
                        "rationale": rationale,
                    }
                elif effective_action == "flag":
                    return {
                        "response": "⚠️ This request has been flagged for compliance review.",
                        "action": "flagged",
                        "rule": rule_name,
                        "similarity_score": round(score, 3),
                        "match_layer": layer,
                        "rationale": rationale,
                    }

        # No rule matched — make the standard LLM call
        try:
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
        except Exception:
            output = "[LLM unavailable in local mode]"

        # Post-Filter: flag high-uncertainty responses
        hedges = sum(1 for h in ["i think", "maybe", "not sure"] if h in output.lower())
        if hedges >= 2:
            return {"response": output, "action": "flagged", "rule": "High Uncertainty Flag",
                    "similarity_score": 0.0, "match_layer": "Confidence Check",
                    "rationale": "Response contained multiple hedging phrases indicating model uncertainty."}

        return {"response": output, "action": "passed", "rule": None,
                "similarity_score": 0.0, "match_layer": "None", "rationale": ""}
