import streamlit as st
from processing.pdf_handler import extract_text_from_pdf
from processing.text_processor import segment_text_into_chunks # generate_embeddings is not directly used from here now
from search_utils.indexing import create_faiss_index
from search_utils.semantic_search import search_in_faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import base64 
from streamlit_pdf_viewer import pdf_viewer as st_pdf_viewer

def main():
    st.set_page_config(layout="wide") # Use wider layout for better readability
    st.title("Interactive PDF Analyzer & Semantic Search")

    # Initialize session state variables
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'index_built_for_file' not in st.session_state:
        st.session_state.index_built_for_file = None
    if 'sentence_model' not in st.session_state:
        with st.spinner("Loading sentence embedding model... This may take a moment on first run."):
            try:
                st.session_state.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.sidebar.success("Sentence model loaded successfully.")
            except Exception as e:
                st.sidebar.error(f"Failed to load sentence model: {e}")
                st.error("Critical Error: Could not load the sentence embedding model. The application cannot proceed. Please check your internet connection or model configuration.")
                st.stop() # Stop execution if model fails to load
    if 'uploaded_file_bytes' not in st.session_state: 
        st.session_state.uploaded_file_bytes = None

    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

    if uploaded_file is not None:
        st.session_state.uploaded_file_bytes = uploaded_file.getvalue()

        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.processed_data = None 
            st.session_state.faiss_index = None 
            st.session_state.index_built_for_file = None
            processing_successful = False # Flag to track overall success

            st.info(f"Processing new file: {uploaded_file.name}")
            with st.spinner("Processing PDF: Extracting text, segmenting, and generating embeddings... This may take a while for large PDFs."):
                pages_data = extract_text_from_pdf(uploaded_file)

                if not pages_data:
                    st.error("Could not extract text from the PDF. It might be image-based, empty, corrupted, or password-protected.")
                else:
                    st.success(f"Step 1/3: Text extracted successfully ({len(pages_data)} pages).")
                    
                    chunks_with_metadata = segment_text_into_chunks(pages_data)
                    if not chunks_with_metadata:
                        st.warning("Step 2/3: No text chunks were generated after segmentation. The PDF might have very little text content or an unusual structure.")
                    else:
                        st.success(f"Step 2/3: Text segmented into {len(chunks_with_metadata)} chunks.")
                        try:
                            texts_to_embed = [item['text_chunk'] for item in chunks_with_metadata]
                            embeddings = st.session_state.sentence_model.encode(texts_to_embed, convert_to_tensor=False, show_progress_bar=True) # Added progress bar for encoding
                            for i, item in enumerate(chunks_with_metadata):
                                item['embedding'] = embeddings[i]
                            data_with_embeddings = chunks_with_metadata
                            st.session_state.processed_data = data_with_embeddings
                            st.success(f"Step 3/3: Embeddings generated for {len(data_with_embeddings)} chunks.")
                            
                            if data_with_embeddings:
                                # Verification (Optional)
                                # st.write(f"Number of chunks with embeddings: {len(data_with_embeddings)}")
                                if len(data_with_embeddings) > 0 and 'embedding' in data_with_embeddings[0] and len(data_with_embeddings[0]['embedding']) > 0 :
                                    # st.write(f"Embedding vector dimension: {len(data_with_embeddings[0]['embedding'])}")
                                    pass # No need to display this to user normally
                                
                                # This expander can be removed for cleaner UI, or kept for debugging
                                # with st.expander("View Sample Processed Chunks"):
                                #     for i, chunk_info in enumerate(data_with_embeddings[:2]):
                                #         st.markdown(f"**Chunk {i+1} (Page {chunk_info['page_number']})**")
                                #         st.caption(chunk_info['text_chunk'][:200] + "...")
                                processing_successful = True # Mark as successful up to this point

                        except Exception as e:
                            st.error(f"Failed to generate text embeddings: {e}. Please try again.")
                            st.session_state.processed_data = None # Ensure no partial data is used
            
            if processing_successful and st.session_state.processed_data:
                with st.spinner("Creating search index..."):
                    faiss_index = create_faiss_index(st.session_state.processed_data)
                    if faiss_index and faiss_index.ntotal > 0:
                        st.session_state.faiss_index = faiss_index
                        st.session_state.index_built_for_file = uploaded_file.name 
                        st.success(f"Search index created successfully. Total indexed items: {st.session_state.faiss_index.ntotal}. You can now ask questions about the document.")
                    elif faiss_index and faiss_index.ntotal == 0:
                        st.warning("Search index was created, but it's empty. This means no content from the PDF could be indexed. Search functionality will be limited.")
                        st.session_state.faiss_index = None # Treat as no index if empty
                    else:
                        st.error("Failed to create a search index from the processed text. Search functionality will be unavailable.")
                        st.session_state.faiss_index = None
            elif processing_successful and not st.session_state.processed_data :
                st.warning("Document processed, but no data was available for indexing. Search will not be available.")


        elif st.session_state.processed_data and st.session_state.index_built_for_file == uploaded_file.name and st.session_state.faiss_index:
            st.info(f"Using existing processed data and search index for {uploaded_file.name}. Total indexed items: {st.session_state.faiss_index.ntotal}")
        elif st.session_state.processed_data and st.session_state.index_built_for_file == uploaded_file.name and not st.session_state.faiss_index:
             st.warning(f"Previously processed data for {uploaded_file.name} is available, but the search index is missing or was not created successfully. You might need to re-upload the file to rebuild the index.")


    # Search Interface
    if st.session_state.faiss_index and st.session_state.faiss_index.ntotal > 0 and st.session_state.processed_data:
        st.header("Ask a Question about the Document")
        user_question = st.text_input("Your question:", placeholder="e.g., What are the main conclusions?")

        if user_question:
            with st.spinner("Searching for relevant information..."):
                search_results = search_in_faiss(
                    query_text=user_question,
                    model=st.session_state.sentence_model,
                    faiss_index=st.session_state.faiss_index,
                    processed_data=st.session_state.processed_data,
                    top_k=5 
                )

            if search_results:
                st.subheader(f"Found {len(search_results)} relevant sections:")
                for idx, result in enumerate(search_results):
                    with st.container(): # Group each result
                        st.markdown(f"**Page {result['page_number']} (Relevance: {1/(1+result['distance']):.2f})**")
                        # Highlight the text chunk using <mark> tag
                        st.markdown(f"<mark>{result['text_chunk']}</mark>", unsafe_allow_html=True)
                        # st.write(f"Raw Distance Score: {result['distance']:.4f}") # Can be hidden for cleaner UI
                        
                        button_key = f"show_page_{idx}_p{result['page_number']}" 
                        if st.button(f"View Page {result['page_number']} in PDF", key=button_key, type="secondary"):
                            if st.session_state.uploaded_file_bytes:
                                try:
                                    # Use a modal or expander for better UX for PDF view
                                    with st.expander(f"Showing Page {result['page_number']} of the PDF", expanded=True):
                                        st_pdf_viewer(st.session_state.uploaded_file_bytes, 
                                                      pages=[result['page_number']], 
                                                      height=600) # Adjusted height
                                except Exception as e:
                                    st.error(f"Error displaying PDF page: {e}. Ensure 'streamlit-pdf-viewer' is installed correctly.")
                            else:
                                st.error("Could not retrieve PDF to display. Please re-upload.")
                        st.divider()
            else:
                st.warning("No relevant sections found for your question in the document.")
    elif uploaded_file and not st.session_state.faiss_index :
         st.warning("Document processing is incomplete or failed, or the document contains no indexable content. Search is unavailable. Please check messages above or try re-uploading.")
    elif not uploaded_file:
        st.info("Please upload a PDF document to begin analyzing and asking questions.")

if __name__ == "__main__":
    main()
