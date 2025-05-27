import numpy as np
from sentence_transformers import SentenceTransformer # Will be used if model is not passed

def search_in_faiss(query_text: str, model, faiss_index, processed_data: list, top_k: int = 5):
    """
    Searches for a query in the FAISS index and retrieves relevant text chunks.

    Args:
        query_text: The user's question.
        model: The initialized SentenceTransformer model.
        faiss_index: The FAISS index.
        processed_data: List of dictionaries with 'text_chunk', 'page_number', 'embedding'.
        top_k: Number of top results to retrieve.

    Returns:
        A list of dictionaries, where each dictionary contains:
            'text_chunk': The retrieved text segment.
            'page_number': The page number of the chunk.
            'distance': The L2 distance (relevance score).
    """
    if not query_text or faiss_index is None or not processed_data:
        return []

    # Generate embedding for the query
    query_embedding = model.encode([query_text], convert_to_tensor=False).astype('float32')
    
    if query_embedding.ndim == 1: # Ensure it's a 2D array for FAISS
        query_embedding = query_embedding.reshape(1, -1)

    # Perform search
    distances, indices = faiss_index.search(query_embedding, top_k)

    results = []
    if indices.size == 0: # Should not happen with IndexFlatL2 unless k=0 or index is empty
        return results

    for i, retrieved_idx in enumerate(indices[0]):
        if retrieved_idx != -1 and retrieved_idx < len(processed_data): # Check for valid index
            retrieved_item = processed_data[retrieved_idx]
            results.append({
                'text_chunk': retrieved_item['text_chunk'],
                'page_number': retrieved_item['page_number'],
                'distance': distances[0][i]
            })
        # else:
            # print(f"Warning: Retrieved index {retrieved_idx} is out of bounds or invalid.")
            
    return results
