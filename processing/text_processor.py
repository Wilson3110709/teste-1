from sentence_transformers import SentenceTransformer
import nltk # For sentence tokenization
import fitz # For fitz.Rect, though page_object is passed

def segment_text_into_chunks(pages_references: list) -> list:
    """
    Segments text from pages into sentences and finds their coordinates.

    Args:
        pages_references: A list of dictionaries from pdf_handler.py, where each contains:
            'page_number': The 1-indexed page number.
            'page_object': The PyMuPDF page object.

    Returns:
        A list of dictionaries, where each dictionary contains:
            'text_chunk': The sentence text.
            'page_number': The page number from which the sentence originated.
            'coordinates': A tuple (x0, y0, x1, y1) for the sentence's bounding box.
    """
    chunks_with_metadata = []
    if not pages_references:
        return chunks_with_metadata

    for page_ref in pages_references:
        page_number = page_ref['page_number']
        page_object = page_ref['page_object']
        
        try:
            page_text = page_object.get_text("text")
        except Exception as e:
            print(f"Error getting text from page {page_number}: {e}")
            continue # Skip this page if text extraction fails

        if not page_text or page_text.isspace():
            continue

        sentences = nltk.sent_tokenize(page_text)

        for sentence_text in sentences:
            stripped_sentence = sentence_text.strip()
            # Filter out very short sentences or sentences that are mostly noise
            if len(stripped_sentence) < 10 or len(stripped_sentence.split()) < 3: # Min 10 chars or 3 words
                continue

            try:
                # search_for can return multiple Rects if the sentence is fragmented
                # or appears multiple times.
                rects = page_object.search_for(stripped_sentence)
            except Exception as e:
                print(f"Error searching for sentence on page {page_number}: '{stripped_sentence[:50]}...' - Error: {e}")
                continue # Skip this sentence if search_for fails

            if not rects: # If sentence not found by search_for, skip it
                # This can happen if NLTK's sentence is not an exact match for PDF text blocks
                # Or if the sentence is part of a larger image/vector graphic.
                # print(f"Warning: Sentence not found by search_for on page {page_number}: '{stripped_sentence[:50]}...'")
                continue


            for rect in rects:
                # Ensure rect is a fitz.Rect object and has valid coordinates
                if not isinstance(rect, fitz.Rect) or not all(hasattr(rect, attr) for attr in ['x0', 'y0', 'x1', 'y1']):
                    print(f"Warning: Invalid rect object found for sentence on page {page_number}: {rect}")
                    continue
                
                # Further filter: sometimes search_for can find very small/thin rects.
                # A rect should have a noticeable width and height.
                if (rect.x1 - rect.x0) < 5 or (rect.y1 - rect.y0) < 5: # Minimum 5px width/height
                    continue

                chunk = {
                    'text_chunk': stripped_sentence,
                    'page_number': page_number,
                    'coordinates': (rect.x0, rect.y0, rect.x1, rect.y1) 
                }
                chunks_with_metadata.append(chunk)
                
    return chunks_with_metadata

def generate_embeddings(chunks_with_metadata: list, model_name: str = 'all-MiniLM-L6-v2', model_object=None) -> list:
    """
    Generates embeddings for text chunks using a SentenceTransformer model.

    Args:
        chunks_with_metadata: A list of dictionaries from segment_text_into_chunks.
        model_name: The SentenceTransformer model to use (if model_object is None).
        model_object: An optional pre-loaded SentenceTransformer model.

    Returns:
        The updated list of dictionaries, with an 'embedding' key added to each.
    """
    if not chunks_with_metadata:
        return []

    # Use the provided model object if available, otherwise load the model by name
    if model_object:
        model = model_object
    elif model_name:
        model = SentenceTransformer(model_name)
    else: # Should not happen if called correctly from app.py where model is cached
        raise ValueError("Either model_name or model_object must be provided to generate_embeddings")

    texts_to_embed = [item['text_chunk'] for item in chunks_with_metadata]
    
    try:
        embeddings = model.encode(texts_to_embed, convert_to_tensor=False, show_progress_bar=True) # show_progress_bar can be passed from app.py
        
        for i, item in enumerate(chunks_with_metadata):
            item['embedding'] = embeddings[i]
            
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Decide how to handle this: return partial data, raise error, etc.
        # For now, let's assume if embedding fails, we don't want to proceed with partial data.
        # Or, we could add a 'embedding_error': True flag to affected items.
        # For simplicity, if batch encoding fails, we might return the list without embeddings or an empty list.
        # Let's return it as is, and app.py should check for 'embedding' key.
        # A better approach might be to add an error status to each chunk.
        # For now, if model.encode fails globally, no embeddings are added.
        pass # Embeddings will not be added if model.encode fails.
            
    return chunks_with_metadata
