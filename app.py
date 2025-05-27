import streamlit as st
import nltk # Added for NLTK punkt download
import fitz # PyMuPDF for highlighting

# NLTK data download (punkt)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    with st.spinner("Baixando recursos de linguagem (NLTK punkt)... Isso pode levar um momento."):
        nltk.download('punkt', quiet=True)

from processing.pdf_handler import extract_text_from_pdf
from processing.text_processor import segment_text_into_chunks 
# generate_embeddings from text_processor is not directly used; app.py handles it
from search_utils.indexing import create_faiss_index
from search_utils.semantic_search import search_in_faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import base64 
from streamlit_pdf_viewer import pdf_viewer as st_pdf_viewer

MAX_HISTORY_LENGTH = 10

def main():
    st.set_page_config(layout="wide") 
    st.title("Analisador Interativo de PDF & Pesquisa Semântica")

    # Initialize session state variables
    if 'pymupdf_doc_original_for_processing' not in st.session_state: # Stores the pristine doc used for processing stages
        st.session_state.pymupdf_doc_original_for_processing = None
    if 'processed_data' not in st.session_state: # This will now store chunks (sentences) with coordinates and embeddings
        st.session_state.processed_data = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'index_built_for_file' not in st.session_state:
        st.session_state.index_built_for_file = None
    if 'sentence_model' not in st.session_state:
        with st.spinner("Carregando modelo de linguagem... Isso pode levar um momento na primeira execução."):
            try:
                st.session_state.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.sidebar.success("Modelo de linguagem carregado com sucesso.")
            except Exception as e:
                st.sidebar.error(f"Falha ao carregar o modelo de linguagem: {e}")
                st.error("Erro Crítico: Não foi possível carregar o modelo de embedding de frases. A aplicação não pode continuar. Verifique sua conexão com a internet ou a configuração do modelo.")
                st.stop() 
    if 'uploaded_file_bytes' not in st.session_state: # Stores the raw bytes of the currently uploaded PDF
        st.session_state.uploaded_file_bytes = None
    if 'highlighted_pdf_to_display' not in st.session_state: # Stores bytes of PDF with all highlights
        st.session_state.highlighted_pdf_to_display = None
    if 'history' not in st.session_state: 
        st.session_state.history = []
    if 'current_question_from_history' not in st.session_state: 
        st.session_state.current_question_from_history = ""

    uploaded_file = st.file_uploader("Carregue seu documento PDF", type="pdf")

    if uploaded_file is not None:
        current_file_bytes = uploaded_file.getvalue() 

        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.pymupdf_doc_original_for_processing = None
            st.session_state.processed_data = None 
            st.session_state.faiss_index = None 
            st.session_state.index_built_for_file = None
            st.session_state.history = [] 
            st.session_state.current_question_from_history = "" 
            st.session_state.uploaded_file_bytes = current_file_bytes # Store current raw bytes
            st.session_state.highlighted_pdf_to_display = None # Clear previous highlights for new file
            processing_successful = False 

            st.info(f"Processando novo arquivo: {uploaded_file.name}")
            # Use uploaded_file directly for extract_text_from_pdf as it expects Streamlit UploadedFile object
            doc_for_processing, pages_refs = extract_text_from_pdf(uploaded_file) 
            st.session_state.pymupdf_doc_original_for_processing = doc_for_processing


            if not st.session_state.pymupdf_doc_original_for_processing or not pages_refs:
                st.error("Não foi possível processar o PDF (etapa de extração). Pode estar vazio, corrompido ou não ser um PDF válido.")
            else:
                st.success(f"Etapa 1/4: Documento PDF aberto e referências de página extraídas ({len(pages_refs)} páginas).")
                
                with st.spinner("Processando PDF: Etapa 2 - Segmentando texto em sentenças..."):
                    chunks_with_coords = segment_text_into_chunks(pages_refs) 
                
                if not chunks_with_coords:
                    st.warning("Etapa 2/4: Nenhuma sentença foi extraída após a segmentação. O PDF pode ter muito pouco conteúdo textual ou uma estrutura incomum.")
                else:
                    st.success(f"Etapa 2/4: Texto segmentado em {len(chunks_with_coords)} sentenças com coordenadas.")
                    
                    with st.spinner("Processando PDF: Etapa 3 - Gerando embeddings para as sentenças..."):
                        try:
                            texts_to_embed = [item['text_chunk'] for item in chunks_with_coords]
                            embeddings = st.session_state.sentence_model.encode(texts_to_embed, convert_to_tensor=False, show_progress_bar=True)
                            
                            data_with_embeddings = []
                            for i, item in enumerate(chunks_with_coords):
                                if embeddings.shape[0] > i : 
                                    item['embedding'] = embeddings[i]
                                    data_with_embeddings.append(item)
                                else:
                                    print(f"Alerta: Embedding não gerado para o chunk {i}. Pulando.")

                            st.session_state.processed_data = data_with_embeddings
                            if not data_with_embeddings:
                                st.error("Etapa 3/4: Falha ao gerar embeddings para as sentenças. Nenhum dado processado.")
                            else:
                                st.success(f"Etapa 3/4: Embeddings gerados para {len(st.session_state.processed_data)} sentenças.")
                                processing_successful = True
                        except Exception as e:
                            st.error(f"Etapa 3/4: Falha ao gerar os embeddings de texto: {e}. Por favor, tente novamente.")
                            st.session_state.processed_data = None 
            
            if processing_successful and st.session_state.processed_data:
                with st.spinner("Processando PDF: Etapa 4 - Criando índice de busca..."):
                    faiss_index = create_faiss_index(st.session_state.processed_data) 
                    if faiss_index and faiss_index.ntotal > 0:
                        st.session_state.faiss_index = faiss_index
                        st.session_state.index_built_for_file = uploaded_file.name 
                        st.success(f"Etapa 4/4: Índice de busca criado com sucesso. Total de sentenças indexadas: {st.session_state.faiss_index.ntotal}. Agora você pode fazer perguntas sobre o documento.")
                    elif faiss_index and faiss_index.ntotal == 0:
                        st.warning("Etapa 4/4: O índice de busca foi criado, mas está vazio. Nenhuma sentença do PDF pôde ser indexada. A funcionalidade de busca será limitada.")
                        st.session_state.faiss_index = None 
                    else:
                        st.error("Etapa 4/4: Falha ao criar um índice de busca a partir do texto processado. A funcionalidade de busca estará indisponível.")
                        st.session_state.faiss_index = None
            elif processing_successful and not st.session_state.processed_data and st.session_state.pymupdf_doc_original_for_processing:
                st.warning("Processamento do documento iniciado, mas dados para indexação não foram gerados (verifique etapas 2 ou 3). A busca não estará disponível.")
            elif not st.session_state.pymupdf_doc_original_for_processing : 
                 pass 
        else: # Same file uploaded again, or page reloaded with same file active
             st.session_state.uploaded_file_bytes = current_file_bytes # Ensure current bytes are up-to-date

        # --- UI for existing processed data (if not a new file upload) ---
        if st.session_state.index_built_for_file == uploaded_file.name and st.session_state.uploaded_file_name == uploaded_file.name : # check if it's still the same file
            if st.session_state.faiss_index:
                 st.info(f"Usando dados processados e índice de busca existentes para {uploaded_file.name}. Total de sentenças indexadas: {st.session_state.faiss_index.ntotal}")
            else: # Index is missing for some reason
                 st.warning(f"Dados processados anteriormente para {uploaded_file.name} estão disponíveis, mas o índice de busca está ausente ou não foi criado com sucesso. Pode ser necessário reenviar o arquivo para reconstruir o índice.")
        
    # --- Search Interface & Results Display Area ---
    # Use two columns: one for search input & text results, one for PDF viewer
    col1, col2 = st.columns([1, 1]) # Adjust ratio as needed, e.g., [2,3]

    with col1:
        if st.session_state.faiss_index and st.session_state.faiss_index.ntotal > 0 and st.session_state.processed_data:
            st.header("Faça uma Pergunta sobre o Documento")
            
            user_question_input = st.text_input("Sua pergunta:", 
                                                value=st.session_state.current_question_from_history, 
                                                placeholder="Ex: Quais são as principais conclusões?",
                                                key="user_question_main_input") 
            
            if st.session_state.current_question_from_history:
                st.session_state.current_question_from_history = "" 

            if user_question_input: 
                user_question = user_question_input 
                # Clear previous highlighted PDF when a new search is made
                st.session_state.highlighted_pdf_to_display = None

                with st.spinner("Buscando informações relevantes..."):
                    search_results = search_in_faiss(
                        query_text=user_question,
                        model=st.session_state.sentence_model,
                        faiss_index=st.session_state.faiss_index,
                        processed_data=st.session_state.processed_data,
                        top_k=10 # Increased top_k for more highlights
                    )

                if search_results:
                    st.subheader(f"Encontradas {len(search_results)} seções (sentenças) relevantes para: \"{user_question}\"")
                    
                    top_answer_chunk = search_results[0]['text_chunk']
                    top_answer_page = search_results[0]['page_number']
                    history_entry = {'question': user_question, 'answer_chunk': top_answer_chunk, 'page_number': top_answer_page}
                    
                    if not st.session_state.history or st.session_state.history[-1]['question'] != user_question:
                        st.session_state.history.append(history_entry)
                        if len(st.session_state.history) > MAX_HISTORY_LENGTH:
                            st.session_state.history = st.session_state.history[-MAX_HISTORY_LENGTH:]

                    # --- Generate PDF with all highlights ---
                    with st.spinner("Gerando PDF com destaques..."):
                        try:
                            if st.session_state.uploaded_file_bytes: # Use the raw bytes of the current PDF
                                temp_doc_for_highlighting = fitz.open(stream=st.session_state.uploaded_file_bytes, filetype="pdf")
                                
                                for result in search_results:
                                    page_num = result['page_number']
                                    coordinates = result['coordinates']
                                    if page_num <= temp_doc_for_highlighting.page_count:
                                        page = temp_doc_for_highlighting.load_page(page_num - 1) 
                                        page.add_highlight_annot(fitz.Rect(coordinates))
                                    else:
                                        print(f"Alerta: Número de página {page_num} inválido para destaque. Máximo: {temp_doc_for_highlighting.page_count}")

                                st.session_state.highlighted_pdf_to_display = temp_doc_for_highlighting.tobytes()
                                temp_doc_for_highlighting.close()
                                st.success("PDF com destaques gerado.")
                            else:
                                st.error("Bytes do PDF original não encontrados para gerar destaques.")
                                st.session_state.highlighted_pdf_to_display = None
                        except Exception as e:
                            st.error(f"Erro ao gerar PDF com destaques: {e}")
                            st.session_state.highlighted_pdf_to_display = None
                    
                    # --- Display Text Results ---
                    st.markdown("#### Resultados da Busca (Sentenças):")
                    for idx, result in enumerate(search_results):
                        with st.container(): 
                            st.markdown(f"**Página {result['page_number']} (Similaridade: {1/(1+result['distance']):.2f})**")
                            st.markdown(f"> {result['text_chunk']}") 
                            st.divider()
                else: # No search results
                    st.warning(f"Nenhuma seção relevante encontrada para sua pergunta: \"{user_question}\" no documento.")
                    st.session_state.highlighted_pdf_to_display = None # Clear if no results
        
        elif uploaded_file and not st.session_state.faiss_index :
            st.warning("O processamento do documento está incompleto, falhou ou o documento não contém conteúdo indexável. A busca está indisponível. Verifique as mensagens acima ou tente reenviar o arquivo.")
        elif not uploaded_file:
            st.info("Por favor, carregue um documento PDF para começar a analisar e fazer perguntas.")

    # --- PDF Viewer Area (in the second column) ---
    with col2:
        st.header("Visualizador de PDF")
        if 'highlighted_pdf_to_display' in st.session_state and st.session_state.highlighted_pdf_to_display:
            st.subheader("PDF com Destaques")
            st_pdf_viewer(st.session_state.highlighted_pdf_to_display, height=800, pages_to_render=[]) # pages_to_render=[] should show all
            # Add a download button for the highlighted PDF
            st.download_button(
                label="Baixar PDF com Destaques",
                data=st.session_state.highlighted_pdf_to_display,
                file_name=f"{st.session_state.uploaded_file_name.replace('.pdf', '')}_destacado.pdf" if st.session_state.uploaded_file_name else "highlighted_document.pdf",
                mime="application/pdf"
            )
        elif 'uploaded_file_bytes' in st.session_state and st.session_state.uploaded_file_bytes:
            st.subheader("Documento PDF Original")
            st_pdf_viewer(st.session_state.uploaded_file_bytes, height=800, pages_to_render=[])
        else:
            st.info("Nenhum PDF carregado ou processado para visualização.")


    # --- History Display in Sidebar ---
    st.sidebar.title("Histórico de Perguntas")
    if not st.session_state.history:
        st.sidebar.caption("Nenhuma pergunta feita ainda nesta sessão ou para este documento.")
    else:
        st.sidebar.caption(f"Mostrando as últimas {len(st.session_state.history)} perguntas para o documento atual.")
        for i, item in enumerate(reversed(st.session_state.history)):
            original_index = len(st.session_state.history) - 1 - i 
            with st.sidebar.expander(f"P: {item['question'][:30]}...", expanded=False):
                st.markdown(f"**Pergunta:** {item['question']}")
                st.markdown(f"**Melhor Resposta (Página {item['page_number']}):** {item['answer_chunk'][:150]}...") 
                
                if st.button(f"Refazer esta pergunta", key=f"history_q_btn_{original_index}"):
                    st.session_state.current_question_from_history = item['question']
                    # Clear any active PDF highlight display when running a new search from history
                    st.session_state.highlighted_pdf_to_display = None 
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
