import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIG ---
st.set_page_config(page_title="TIA Peak Analytics Chatbot", page_icon="✈️", layout="wide")
st.title("✈️ Tirana International Airport Peak Season Assistant")
st.write("Ask questions about monthly passenger distribution, peak tourist flows, and airline statistics for 2025.")
st.markdown("---")

# --- INITIALIZE ENVIRONMENT VARIABLES ---
load_dotenv()

# --- AUTO-FETCH SECRET API KEY ---
api_key = os.environ.get("GROQ_API_KEY", "")

if not api_key:
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

# --- SIDEBAR (Upgraded with Executive KPI Metrics) ---
with st.sidebar:
    st.header("System Status & Info")
    if api_key:
        st.success("✅ Groq Cloud API Connected")
    else:
        st.error("⚠️ Missing API Key in Dashboard Secrets.")
        
    st.info(
        "This assistant is powered by a Retrieval-Augmented Generation (RAG) architecture using real "
        "2025 TIA flight data logs to analyze congestion levels and market positions."
    )
    
    st.markdown("---")
    st.markdown("### 📊 2025 TIA Executive Metrics")
    
    # Side-by-side KPI Cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Peak Month (Aug)", value="1.14M Pax", delta="+24% vs Avg")
    with col2:
        st.metric(label="Top Carrier", value="Wizz Air", delta="54% Share")
        
    st.metric(label="Total Annual Volume", value="8.42M Passengers", delta="Record High")
    st.markdown("---")
    
    st.markdown("### 🔍 Try Asking:")
    st.caption("• What was the total volume of passengers at TIA in August 2025?")
    st.caption("• Which month experienced the absolute lowest traffic baseline?")
    st.caption("• Compare the passenger numbers for Wizz Air and Ryanair in May.")

# --- INITIALIZE MODELS ---
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("faiss_tia_index", embeddings, allow_dangerous_deserialization=True)
    
    system_prompt = (
        "You are an expert aviation data analyst specialized in Tirana International Airport (TIA) passenger traffic statistics.\n\n"
        "CRITICAL ANALYSIS & LOGIC RULES:\n"
        "1. For any monthly totals or airport-wide traffic summaries, look ONLY at the context string beginning with 'Summary Category: TOTAL' or 'Overall Air Traffic Summary Category: TOTAL'. Never add individual airline numbers to this row, as it already includes all flights.\n"
        "2. To find the absolute lowest or highest month, carefully review all 12 listed values inside the TOTAL category chunk and compare them as actual numbers, not as text strings.\n"
        "3. Ignore summary rows like 'Shuma', 'CHARTER', or 'TOTAL' when ranking individual airlines or finding the top carrier.\n"
        "4. Base your answers strictly and exclusively on the context statistics provided below. Do not guess or speculate.\n"
        "5. RESPONSE FORMATTING CONSTRAINT: Never list out individual airlines one by one using 'and' clauses repeatedly. Provide a direct, concise summary of the data requested. If you find yourself repeating the same data point, stop generating immediately.\n\n"
        "Context:\n{context}\n\n"
        "6. MANDATORY VISUALIZATION RULE:\n"
        "Whenever your answer contains ANY numeric comparisons, statistical data points, or monthly values, you MUST append a raw data block at the very end of your response using this exact syntax: [CHART_DATA: Label1=Value1, Label2=Value2].\n"
        "Examples of mandatory outputs:\n"
        "- If comparing airlines: [CHART_DATA: Wizz Air=541483, Ryanair=248640]\n"
        "- If showing monthly trends: [CHART_DATA: June=750000, July=920000, August=1140000]\n"
        "Never use commas or punctuation marks inside the numeric values. Do not forget to include this block if numbers are present."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    return vector_db, prompt

if os.path.exists("faiss_tia_index"):
    vector_db, prompt = load_rag_system()
    
    # --- CHAT HISTORY DISPLAY ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if "[CHART_DATA:" in content:
                clean_text, chart_block = content.split("[CHART_DATA:")
                st.markdown(clean_text.strip())
                try:
                    chart_data_str = chart_block.replace("]", "").strip()
                    raw_pairs = [pair.split("=") for pair in chart_data_str.split(",") if "=" in pair]
                    chart_dict = {}
                    for pair in raw_pairs:
                        label = pair[0].strip()
                        val_clean = pair[1].strip().replace(",", "").replace(".", "")
                        chart_dict[label] = int(val_clean)
                    if chart_dict:
                        st.bar_chart(chart_dict)
                except Exception:
                    pass
            else:
                st.markdown(content)

    # --- CHAT INPUT HANDLING ---
    if query := st.chat_input("Ex: What was the total passenger count in August?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Retrieving high-density flight logs and computing trends..."):
            try:
                llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant", temperature=0)
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 45}), question_answer_chain)
                
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                
                # Append raw response to history so it persists natively
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                with st.chat_message("assistant"):
                    # --- INTERCEPT & INTERPRET VISUALIZATION FLAG ---
                    if "[CHART_DATA:" in answer:
                        clean_text, chart_block = answer.split("[CHART_DATA:")
                        
                        # Output text answer first
                        st.markdown(clean_text.strip())
                        
                        # Process and draw chart graphics securely
                        try:
                            chart_data_str = chart_block.replace("]", "").strip()
                            raw_pairs = [pair.split("=") for pair in chart_data_str.split(",") if "=" in pair]
                            
                            chart_dict = {}
                            for pair in raw_pairs:
                                label = pair[0].strip()
                                # Clean values by stripping whitespaces and filtering internal number punctuation
                                val_clean = pair[1].strip().replace(",", "").replace(".", "")
                                chart_dict[label] = int(val_clean)
                            
                            # Render interactive Streamlit Bar Chart
                            if chart_dict:
                                st.bar_chart(chart_dict)
                            else:
                                st.caption("*(No valid structured chart data found)*")
                                
                        except Exception as parse_error:
                            st.caption(f"*(Data format parsing error: {str(parse_error)})*")
                    else:
                        st.markdown(answer)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
else:
    st.warning("⚠️ Flight logs vector index not found. Please ensure your faiss_tia_index directory is pushed to GitHub.")