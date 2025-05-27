import fitz  # PyMuPDF
# PyPDF2 is no longer used.

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text and page objects from an uploaded PDF file using PyMuPDF.

    Args:
        uploaded_file: The uploaded file object from Streamlit's file_uploader.

    Returns:
        A tuple containing:
            - doc: The PyMuPDF document object.
            - pages_references: A list of dictionaries, where each dictionary contains:
                'page_number': The 1-indexed page number.
                'page_object': The PyMuPDF page object.
                'text': The extracted text from the page (optional).
        Returns (None, []) if an error occurs or PDF is empty/unparseable.
    """
    try:
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            print("Error: Uploaded file is empty.")
            return None, []
            
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        pages_references = []
        if doc.page_count == 0: # Handle empty or non-parseable PDF
            print("Warning: PDF has 0 pages or could not be parsed correctly.")
            # It's better to return the doc object even if it has 0 pages, 
            # so the caller doesn't get None for doc unless fitz.open fails.
            return doc, [] 

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pages_references.append({
                'page_number': page_num + 1,
                'page_object': page,
                'text': page.get_text("text") # Optional: for display or simple use
            })
        return doc, pages_references
    except Exception as e:
        # Log the error. Depending on Streamlit deployment, this might go to stdout/stderr.
        # For production, a more robust logging mechanism would be used.
        print(f"Error processing PDF with PyMuPDF: {e}") 
        return None, [] # Return None for doc and empty list on error
