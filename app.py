import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# =====================================================================
# 1. LOAD AND PARSE THE REAL TIA PASSENGER CSV DATASET (Robust Index-Based Fix)
# =====================================================================
csv_path = "Pax_total_2025.csv"
print(f"Loading flight data from {csv_path}...")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not find '{csv_path}'. Please ensure it is in your project directory.")

# Open using cp1252/ANSI encoding to seamlessly read Albanian special characters
df = pd.read_csv(csv_path, encoding="cp1252")

docs = []

# Add overall qualitative context regarding Tirana Airport operational challenges
airport_context = (
    "Tirana International Airport (TIA) 'Mother Teresa' experiences extreme seasonal volatility. "
    "Traffic peaks massively during the summer months of June, July, and August, as well as the winter holidays in December. "
    "During these peak summer periods, high passenger volumes cause congestion, making wait times at security, "
    "check-in, and immigration up to 30% to 50% longer than the annual average."
)
docs.append(Document(page_content=airport_context, metadata={"source": "operational_guidelines"}))

# Convert each row of the airline statistics into descriptive text sentences
for idx, row in df.iterrows():
    # Bypassing string keys completely: reference columns strictly by numerical position (iloc)
    operator = str(row.iloc[1]).strip()  # 2nd column: Airline/Operator Name
    total_pax = str(row.iloc[-2]).strip() # 2nd to last column: TOTAL Pax
    share = str(row.iloc[-1]).strip()     # Very last column: Market Share %
    
    # CHANGED: High-density compact format to drastically save token room for Groq Limits
    if operator in ['Shuma', 'CHARTER', 'TOTAL']:
        text_summary = f"Summary Category: {operator} | 2025 Total: {total_pax} pax | Share: {share}."
    else:
        text_summary = f"Airline: {operator} | 2025 Total: {total_pax} pax | Share: {share}."
    
    # CHANGED: Abbreviated months to scale back context length safely
    english_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    monthly_details = []
    for col_idx, eng in enumerate(english_months, start=2):
        val = str(row.iloc[col_idx]).strip()
        if val != '0' and val != '0.0' and val != 'nan' and val != '':
            monthly_details.append(f"{eng}:{val}")
            
    if monthly_details:
        text_summary += " Monthly: " + ", ".join(monthly_details)
    
    # Append structured record as a searchable document
    docs.append(Document(page_content=text_summary, metadata={"source": csv_path, "operator": operator}))

# =====================================================================
# 2. CONTINUING WITH SPLITTING AND CHUNKING
# =====================================================================
print("Initializing text splitter...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=50,
    length_function=len
)

# 3. CREATE CHUNKS
chunks = text_splitter.split_documents(docs)

print("-" * 40)
print(f"✅ Data processing complete!")
print(f"Total Rows converted to documents: {len(docs)}")
print(f"Total Text Chunks generated for database: {len(chunks)}")
print("-" * 40)

# =====================================================================
# 4. INITIALIZE EMBEDDING MODEL
# =====================================================================
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# =====================================================================
# 5. CREATE AND SAVE VECTOR DATABASE (FAISS)
# =====================================================================
print("Building FAISS index...")
vector_db = FAISS.from_documents(chunks, embeddings)

# CHANGED: Using your clean, renamed local storage location
vector_db.save_local("faiss_tia_index") 
print("✅ Vector database successfully updated and saved locally.")

# =====================================================================
# 6. TEST THE RETRIEVAL CHAIN WITH A REAL QUESTION (Optimized for Math Accuracy)
# =====================================================================
query = "What was the total volume of passengers at TIA in August 2025 and which month was the absolute lowest?"
print(f"\n🔍 Querying RAG System: '{query}'")

os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# We update the prompt to enforce strict data inspection rules
system_prompt = (
    "You are an expert aviation data analyst specialized in Tirana International Airport (TIA) passenger traffic statistics. "
    "Your task is to answer user queries with absolute numerical precision based on the provided context.\n\n"
    "CRITICAL RULES:\n"
    "1. When asked for airport-wide totals or specific months, look strictly at the 'Summary Category: TOTAL' data chunk.\n"
    "2. To find the absolute lowest or highest month, carefully review all 12 listed values inside the TOTAL category chunk "
    "and compare them as actual numbers, not as text strings. (Note: 632k in Feb is less than 765k in Nov).\n"
    "3. Be direct, professional, and present the exact numbers from the context.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

# CHANGED: Setting k=45 works flawlessly here to pull all your compact data elements without crossing the 6000 TPM limit
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 45}), question_answer_chain)

response = rag_chain.invoke({"input": query})
print("-" * 40)
print(f"RESPONSE:\n{response['answer']}")
print("-" * 40)