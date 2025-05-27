import streamlit as st
from processing.pdf_handler import extract_text_from_pdf
from processing.text_processor import segment_text_into_chunks # generate_embeddings is not directly used from here now
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
    if 'processed_data' not in st.session_state:
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
    if 'uploaded_file_bytes' not in st.session_state: 
        st.session_state.uploaded_file_bytes = None
    if 'history' not in st.session_state: 
        st.session_state.history = []
    if 'current_question_from_history' not in st.session_state: 
        st.session_state.current_question_from_history = ""


    uploaded_file = st.file_uploader("Carregue seu documento PDF", type="pdf")

    if uploaded_file is not None:
        st.session_state.uploaded_file_bytes = uploaded_file.getvalue()

        if st.session_state.uploaded_file_name != uploaded_file.name:
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.processed_data = None 
            st.session_state.faiss_index = None 
            st.session_state.index_built_for_file = None
            st.session_state.history = [] 
            st.session_state.current_question_from_history = "" 
            processing_successful = False 

            st.info(f"Processando novo arquivo: {uploaded_file.name}")
            with st.spinner("Processando PDF: Extraindo texto, segmentando e gerando embeddings... Isso pode levar um tempo para PDFs grandes."):
                pages_data = extract_text_from_pdf(uploaded_file)

                if not pages_data:
                    st.error("Não foi possível extrair o texto do PDF. Pode ser baseado em imagem, estar vazio, corrompido ou protegido por senha.")
                else:
                    st.success(f"Passo 1/3: Texto extraído com sucesso ({len(pages_data)} páginas).")
                    
                    chunks_with_metadata = segment_text_into_chunks(pages_data)
                    if not chunks_with_metadata:
                        st.warning("Passo 2/3: Nenhum bloco de texto foi gerado após a segmentação. O PDF pode ter muito pouco conteúdo de texto ou uma estrutura incomum.")
                    else:
                        st.success(f"Passo 2/3: Texto segmentado em {len(chunks_with_metadata)} blocos.")
                        try:
                            texts_to_embed = [item['text_chunk'] for item in chunks_with_metadata]
                            embeddings = st.session_state.sentence_model.encode(texts_to_embed, convert_to_tensor=False, show_progress_bar=True) 
                            for i, item in enumerate(chunks_with_metadata):
                                item['embedding'] = embeddings[i]
                            data_with_embeddings = chunks_with_metadata
                            st.session_state.processed_data = data_with_embeddings
                            st.success(f"Passo 3/3: Embeddings gerados para {len(data_with_embeddings)} blocos.")
                            processing_successful = True

                        except Exception as e:
                            st.error(f"Falha ao gerar os embeddings de texto: {e}. Por favor, tente novamente.")
                            st.session_state.processed_data = None 
            
            if processing_successful and st.session_state.processed_data:
                with st.spinner("Criando índice de busca..."):
                    faiss_index = create_faiss_index(st.session_state.processed_data)
                    if faiss_index and faiss_index.ntotal > 0:
                        st.session_state.faiss_index = faiss_index
                        st.session_state.index_built_for_file = uploaded_file.name 
                        st.success(f"Índice de busca criado com sucesso. Total de itens indexados: {st.session_state.faiss_index.ntotal}. Agora você pode fazer perguntas sobre o documento.")
                    elif faiss_index and faiss_index.ntotal == 0:
                        st.warning("O índice de busca foi criado, mas está vazio. Isso significa que nenhum conteúdo do PDF pôde ser indexado. A funcionalidade de busca será limitada.")
                        st.session_state.faiss_index = None 
                    else:
                        st.error("Falha ao criar um índice de busca a partir do texto processado. A funcionalidade de busca estará indisponível.")
                        st.session_state.faiss_index = None
            elif processing_successful and not st.session_state.processed_data :
                st.warning("Documento processado, mas nenhum dado estava disponível para indexação. A busca não estará disponível.")

        elif st.session_state.processed_data and st.session_state.index_built_for_file == uploaded_file.name and st.session_state.faiss_index:
            st.info(f"Usando dados processados e índice de busca existentes para {uploaded_file.name}. Total de itens indexados: {st.session_state.faiss_index.ntotal}")
        elif st.session_state.processed_data and st.session_state.index_built_for_file == uploaded_file.name and not st.session_state.faiss_index:
             st.warning(f"Dados processados anteriormente para {uploaded_file.name} estão disponíveis, mas o índice de busca está ausente ou não foi criado com sucesso. Pode ser necessário reenviar o arquivo para reconstruir o índice.")

    # --- Search Interface ---
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
            with st.spinner("Buscando informações relevantes..."):
                search_results = search_in_faiss(
                    query_text=user_question,
                    model=st.session_state.sentence_model,
                    faiss_index=st.session_state.faiss_index,
                    processed_data=st.session_state.processed_data,
                    top_k=5 
                )

            if search_results:
                st.subheader(f"Encontradas {len(search_results)} seções relevantes para: \"{user_question}\"")
                
                top_answer_chunk = search_results[0]['text_chunk']
                top_answer_page = search_results[0]['page_number']
                history_entry = {'question': user_question, 'answer_chunk': top_answer_chunk, 'page_number': top_answer_page}
                
                if not st.session_state.history or st.session_state.history[-1]['question'] != user_question:
                    st.session_state.history.append(history_entry)
                    if len(st.session_state.history) > MAX_HISTORY_LENGTH:
                        st.session_state.history = st.session_state.history[-MAX_HISTORY_LENGTH:]

                for idx, result in enumerate(search_results):
                    with st.container(): 
                        st.markdown(f"**Página {result['page_number']} (Similaridade: {1/(1+result['distance']):.2f})**")
                        st.markdown(f"<mark>{result['text_chunk']}</mark>", unsafe_allow_html=True)
                        
                        button_key = f"show_page_{idx}_p{result['page_number']}" 
                        if st.button(f"Ver Página {result['page_number']} no PDF", key=button_key, type="secondary"):
                            if st.session_state.uploaded_file_bytes:
                                try:
                                    with st.expander(f"Mostrando Página {result['page_number']} do PDF", expanded=True):
                                        st_pdf_viewer(st.session_state.uploaded_file_bytes, 
                                                      pages=[result['page_number']], 
                                                      height=600) 
                                except Exception as e:
                                    st.error(f"Erro ao exibir a página do PDF: {e}. Certifique-se de que 'streamlit-pdf-viewer' está instalado corretamente.")
                            else:
                                st.error("Não foi possível recuperar o PDF para exibição. Por favor, reenvie o arquivo.")
                        st.divider()
            else:
                st.warning(f"Nenhuma seção relevante encontrada para sua pergunta: \"{user_question}\" no documento.")
    elif uploaded_file and not st.session_state.faiss_index :
         st.warning("O processamento do documento está incompleto, falhou ou o documento não contém conteúdo indexável. A busca está indisponível. Verifique as mensagens acima ou tente reenviar o arquivo.")
    elif not uploaded_file:
        st.info("Por favor, carregue um documento PDF para começar a analisar e fazer perguntas.")

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
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
