import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedder = SentenceTransformer('all-MiniLM-L6-v2')
HEDGING = ["i think", "i believe", "maybe", "not sure", "possibly", "cannot guarantee"]

def extract_boundaries(results: list, threshold: float = 0.5) -> list:
    boundaries = []
    for item in results:
        outputs_a = item["outputs_a"]
        output_b = item["output_b"]
        
        # Consistency Score
        embeddings = embedder.encode(outputs_a)
        sims = [cosine_similarity([embeddings[i]], [embeddings[j]])[0][0] 
                for i in range(len(embeddings)) for j in range(i+1, len(embeddings))]
        c_score = 1.0 - np.mean(sims) if sims else 0.0
        
        # Divergence Score
        emb_a, emb_b = embedder.encode([outputs_a[0]]), embedder.encode([output_b])
        d_score = 1.0 - cosine_similarity(emb_a, emb_b)[0][0]
        
        # Confidence Score
        conf_score = min(sum(1 for h in HEDGING if h in outputs_a[0].lower()) / 3.0, 1.0)
        
        boundary_score = (0.4 * c_score + 0.4 * d_score + 0.2 * conf_score)
        
        if boundary_score >= threshold:
            item.update({"boundary_score": round(boundary_score, 3)})
            boundaries.append(item)
            
    boundaries = sorted(boundaries, key=lambda x: x["boundary_score"], reverse=True)
    with open("data/boundaries.json", "w") as f:
        json.dump(boundaries, f, indent=2)
    return boundaries
