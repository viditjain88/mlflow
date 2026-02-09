import json
from collections import defaultdict

def reciprocal_rank_fusion(ranked_lists, k=60):
    """
    Implements Reciprocal Rank Fusion.

    Args:
        ranked_lists (list of lists): Each inner list contains documents (dicts) ranked by relevance.
        k (int): Constant, typically 60.

    Returns:
        list: Re-ranked list of documents.
    """
    rrf_scores = defaultdict(float)
    doc_lookup = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            # Use content as unique identifier or 'id' if available
            # Assuming 'content' is unique enough or combining source + content
            doc_id = doc.get('content', '')
            if not doc_id:
                continue

            doc_lookup[doc_id] = doc
            rrf_scores[doc_id] += 1 / (k + rank + 1)

    # Sort by score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    # Return document objects
    fused_results = []
    for doc_id, score in sorted_docs:
        doc = doc_lookup[doc_id]
        doc['rrf_score'] = score
        fused_results.append(doc)

    return fused_results
