from fastapi import FastAPI, Query
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os

# ---------------------
# Environment setup
# ---------------------
os.environ["GROQ_API_KEY"] = "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"

# ---------------------
# Load and prepare data
# ---------------------
pdf_files = [
    "PDFs/Anti-Money Laundering Law.pdf.pdf",
    "PDFs/Real_Estate_Finance_Law_ِEN.pdf.pdf",
    "PDFs/Regulations_of_the_Finance_Lease-EN.pdf.pdf"
]

documents = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)
split_docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists("faiss_index"):
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("faiss_index")

loaded_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ---------------------
# LLM and prompt
# ---------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Strict Answer:
"""
)

retriever = loaded_vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"],
               model_name="moonshotai/kimi-k2-instruct-0905",
               temperature=0.6)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    chain_type="stuff"
)

# ---------------------
# FastAPI app setup
# ---------------------
app = FastAPI(title="RAG Application API")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "RAG App is running. Use /ask endpoint."}

@app.post("/ask")
def ask_question(request: QueryRequest):
    question = request.query
    answer = qa.run(question)
    return {"question": question, "answer": answer}
