import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIG ---
st.set_page_config(page_title="TIA Peak Analytics Chatbot", page_icon="✈️")
st.title("✈️ Tirana International Airport Peak Season Assistant")
st.write("Ask questions about monthly passenger distribution, peak tourist flows, and airline statistics for 2025.")
st.markdown("---")

# --- SIDEBAR (Settings) ---
with st.sidebar:
    st.header("Authentication & Info")
    api_key = st.text_input("Enter Groq API Key", type="password")
    st.info(
        "This assistant is powered by a Retrieval-Augmented Generation (RAG) architecture using real "
        "2025 TIA flight data logs to analyze congestion levels and market positions."
    )

# --- INITIALIZE MODELS ---
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_tia_index", embeddings, allow_dangerous_deserialization=True)
    
    # Enforced strict analytical guidelines to handle small datasets without math distortions
    system_prompt = (
        "You are an expert aviation data analyst specialized in Tirana International Airport (TIA) passenger traffic statistics.\n\n"
        "CRITICAL ANALYSIS & LOGIC RULES:\n"
        "1. For any monthly totals or airport-wide traffic summaries, look ONLY at the context string beginning with 'Overall Air Traffic Summary Category: TOTAL'. Never add individual airline numbers to this row, as it already includes all flights.\n"
        "2. When asked which airline has the most passengers or to rank airline positions, carefully compare the 'Total annual passengers carried in 2025' across ALL provided individual airline operators in the context to determine the real mathematical maximum.\n"
        "3. Ignore summary rows like 'Shuma', 'CHARTER', or 'TOTAL' when ranking individual airlines or finding the top carrier.\n"
        "4. Base your answers strictly and exclusively on the context statistics provided below. Do not guess, speculate, or double-count.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return vector_db, prompt

if os.path.exists("faiss_tia_index"):
    vector_db, prompt = load_rag_system()
    
    # --- CHAT HISTORY INITIALIZATION ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- CHAT INPUT HANDLING ---
    if query := st.chat_input("Ex: What was the total passenger count in May? Which airline has the most passengers?"):
        if not api_key:
            st.error("Please enter your Groq API Key in the sidebar to activate the analysis model.")
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Retrieving flight logs and computing trends..."):
                try:
                    llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant", temperature=0)
                    qa_chain = create_stuff_documents_chain(llm, prompt)
                    
                    # FIXED: search_kwargs={"k": 54} injects all processed lines into the prompt context.
                    # This allows the LLM to process the mathematical global maximum without partial data filtering.
                    rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 54}), qa_chain)
                    
                    response = rag_chain.invoke({"input": query})
                    answer = response["answer"]
                    
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
else:
    st.warning("⚠️ Flight logs vector index not found. Please run 'python app.py' first to initialize the database from the CSV data.")