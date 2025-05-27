from sentence_transformers import SentenceTransformer

def segment_text_into_chunks(pages_data: list) -> list:
    """
    Segments text from pages into smaller chunks based on paragraphs.

    Args:
        pages_data: A list of dictionaries, where each dictionary contains:
            'page_number': The page number.
            'text': The extracted text from the page.

    Returns:
        A list of dictionaries, where each dictionary contains:
            'text_chunk': The text segment.
            'page_number': The page number from which the chunk originated.
    """
    chunks_with_metadata = []
    for page_info in pages_data:
        page_number = page_info['page_number']
        text = page_info['text']
        
        # Simple Segmentation Strategy (Paragraphs)
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            chunk_text = paragraph.strip() # Remove leading/trailing whitespace
            if len(chunk_text) > 10: # Filter out very short or empty chunks (threshold is arbitrary)
                chunks_with_metadata.append({
                    'text_chunk': chunk_text,
                    'page_number': page_number
                })
    return chunks_with_metadata

def generate_embeddings(chunks_with_metadata: list, model_name: str = 'all-MiniLM-L6-v2') -> list:
    """
    Generates embeddings for text chunks using a SentenceTransformer model.

    Args:
        chunks_with_metadata: A list of dictionaries from segment_text_into_chunks.
        model_name: The SentenceTransformer model to use.

    Returns:
        The updated list of dictionaries, with an 'embedding' key added to each.
    """
    if not chunks_with_metadata:
        return []

    model = SentenceTransformer(model_name)
    
    texts_to_embed = [item['text_chunk'] for item in chunks_with_metadata]
    
    embeddings = model.encode(texts_to_embed, convert_to_tensor=False)
    
    for i, item in enumerate(chunks_with_metadata):
        item['embedding'] = embeddings[i]
        
    return chunks_with_metadata
