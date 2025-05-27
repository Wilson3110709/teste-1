import PyPDF2

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file.

    Args:
        uploaded_file: The uploaded file object from Streamlit's file_uploader.

    Returns:
        A list of dictionaries, where each dictionary contains:
            'page_number': The page number (starting from 1).
            'text': The extracted text from the page.
        Returns an empty list if an error occurs (e.g., encrypted PDF).
    """
    pages_data = []
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text:  # Ensure text was extracted
                    pages_data.append({'page_number': page_num + 1, 'text': text})
                else:
                    pages_data.append({'page_number': page_num + 1, 'text': '[No text found on this page]'})
            except Exception as e:
                print(f"Error extracting text from page {page_num + 1}: {e}")
                pages_data.append({'page_number': page_num + 1, 'text': f'[Error extracting text: {e}]'})
    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF: {e}. The PDF might be encrypted or corrupted.")
        # Optionally, re-raise a custom exception or handle as per app requirements
        return [] # Return empty list for now
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [] # Return empty list for now
    return pages_data
