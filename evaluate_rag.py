import os
import evaluate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. SETUP LLM & RAG 
os.environ["GROQ_API_KEY"] = "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"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local("faiss_medical_index", embeddings, allow_dangerous_deserialization=True)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
rouge = evaluate.load("rouge")

system_prompt = "You are an expert airport analyst for Tirana International Airport. Use the retrieved operational logs to answer: \n\n {context}"
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(), qa_chain)

# 2. EVALUATION FUNCTION
def run_evaluation(query, ground_truth):
    rag_response = rag_chain.invoke({"input": query})
    rag_answer = rag_response["answer"]
    
    # Get baseline response without context access
    no_rag_answer = llm.invoke(query).content

    results = rouge.compute(predictions=[rag_answer], references=[ground_truth])
    return {
        "rag_answer": rag_answer,
        "no_rag_answer": no_rag_answer,
        "score": results['rougeL']
    }

# 3. RUN REAL VERIFICATION TEST
test_query = "What is the specific market share and total annual passenger volume for Wizz Air at TIA in 2025?"
true_answer = "Wizz Air (Group) has a market share of 54.33% with a total annual passenger volume of 6,323,605 in 2025."

eval_result = run_evaluation(test_query, true_answer)

print("\n" + "="*60)
print("📈 TIA PASSENGER RAG ACCURACY REPORT")
print("="*60)
print(f"QUESTION: {test_query}")
print(f"\n📊 RAG ANSWER (Grounded in CSV):\n{eval_result['rag_answer']}")
print(f"\n❌ NO-RAG ANSWER (Pre-trained Guess):\n{eval_result['no_rag_answer']}")
print(f"\n🎯 ROUGE-L ACCURACY SCORE: {eval_result['score']:.4f}")
print("="*60)