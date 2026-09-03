from typing import Any, Dict, List, Tuple
from .hybrid import HybridSearchEngine, reciprocal_rank_fusion
from .base import PayloadFilters

class CodeSearchEngine(HybridSearchEngine):
    """
    Tailored retrieval engine for source code.
    Boosts exact keyword matches (sparse index) more heavily than the default hybrid search,
    since function names and class names are critical in code search.
    """
    
    def search(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 10,
        fusion_k: int = 60,
        filters: PayloadFilters = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        # Fetch more sparse results because exact match is highly valuable in code
        fetch_k = top_k * 5
        dense_hits = self.dense.search(query_vector, k=fetch_k, filters=filters)
        sparse_hits = self.sparse.search(query_text, k=fetch_k * 2, filters=filters)
        
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=fusion_k)
        
        # Boost items that are explicitly code blocks (from the new ProgrammingFileExtractor)
        boosted_results = []
        for doc_id, score, payload in fused:
            new_score = score
            if payload.get("file_type") == "source_code":
                # Give a 20% boost to code blocks for code-specific engines
                new_score *= 1.2
            boosted_results.append((doc_id, new_score, payload))
            
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k]
