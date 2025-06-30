from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from datasets import load_dataset
import cassio
from PyPDF2 import PdfReader

# Setup
ASTRA_DB_APPLICATION_TOKEN = "Dummy"
ASTRA_DB_ID = "rummy"
OPENAI_API_KEY = "aummy"

pdf_path = r"C:\Users\dhawa\OneDrive\Desktop\sample.pdf"
pdfreader = PdfReader(pdf_path)

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
astra_vector_store.add_texts(texts[:50])
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

app = FastAPI()

# Store the latest PDF text globally
raw_text = ''
texts = []

class QueryRequest(BaseModel):
    question: str

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global raw_text, texts, astra_vector_store, astra_vector_index
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are supported."}
    # Read PDF file bytes
    contents = await file.read()
    import io
    pdfreader = PdfReader(io.BytesIO(contents))
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    # Split and add to vector store
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    astra_vector_store.add_texts(texts[:50])
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    return {"message": "PDF processed successfully."}

@app.get("/summary")
def get_summary():
    # Simple summary: first 500 characters of the PDF text
    summary = raw_text[:500]
    return {"summary": summary}

@app.post("/ask")
def ask_question(request: QueryRequest):
    query_text = request.question.strip()
    if not query_text:
        return {"error": "Empty question."}
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    docs = [
        {
            "score": float(score),
            "content": doc.page_content[:200]
        }
        for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4)
    ]
    return {
        "question": query_text,
        "answer": answer,
        "documents": docs
    }
