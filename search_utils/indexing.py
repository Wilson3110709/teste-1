import faiss
import numpy as np

def create_faiss_index(processed_data: list):
    """
    Creates a FAISS index from the embeddings in processed_data.

    Args:
        processed_data: A list of dictionaries, where each dictionary contains:
            'text_chunk': The text segment.
            'page_number': The page number from which the chunk originated.
            'embedding': The embedding vector for the text_chunk.

    Returns:
        A FAISS index object, or None if processed_data is empty or embeddings are invalid.
    """
    if not processed_data or 'embedding' not in processed_data[0]:
        return None

    embeddings = np.array([item['embedding'] for item in processed_data]).astype('float32')
    
    if embeddings.ndim == 1: # Handle case where there might be only one embedding
        embeddings = embeddings.reshape(1, -1)

    if embeddings.shape[0] == 0: # No embeddings to index
        return None
        
    d = embeddings.shape[1]
    
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    
    return index
