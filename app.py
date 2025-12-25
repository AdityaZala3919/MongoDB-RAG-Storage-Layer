from fastapi import FastAPI, UploadFile, Form, File
from pypdf import PdfReader
import uuid
from datetime import datetime
import os
from fastapi.responses import RedirectResponse

from db import documents_col, sessions_col
from rag import chunk_text, create_faiss_index, retrieve_chunks, generate_answer

app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse("/docs")

@app.post("/upload_pdf")
async def upload_pdf(
    username: str = Form(...),
    file: UploadFile = File(...)
):
    doc_id = str(uuid.uuid4())
    reader = PdfReader(file.file)
    text = "\n".join([p.extract_text() for p in reader.pages])

    chunk_size = 1000
    overlap = 200
    chunks = chunk_text(text, chunk_size, overlap)

    faiss_path = f"faiss/{username}_{doc_id}"
    create_faiss_index(chunks, faiss_path)

    documents_col.insert_one({
        "_id": doc_id,
        "username": username,
        "title": file.filename,
        "content": text,
        "faiss_path": faiss_path,
        "chunk_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": overlap
        },
        "created_at": datetime.utcnow()
    })

    session_id = str(uuid.uuid4())
    sessions_col.insert_one({
        "_id": session_id,
        "username": username,
        "document_id": doc_id,
        "created_at": datetime.utcnow()
    })

    return {
        "session_id": session_id,
        "document_id": doc_id
    }


@app.post("/query")
async def query_rag(
    session_id: str = Form(...),
    question: str = Form(...)
):
    session = sessions_col.find_one({"_id": session_id})
    doc = documents_col.find_one({"_id": session["document_id"]})

    chunks = retrieve_chunks(question, doc["faiss_path"])
    context = "\n".join(chunks)

    answer = generate_answer(context, question)
    return {"answer": answer}
