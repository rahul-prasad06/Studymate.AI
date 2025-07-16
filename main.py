import os
import shutil
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Query, Path
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List


from tools.pdf_tool import process_pdf_and_create_vectorstore
from tools.chat_engine import build_chat_model
from dotenv import load_dotenv

load_dotenv()
# Directories for PDF and Vectorstore
TEMP_DIR = "temp/"
VECTORSTORE_DIR = "vectorstore/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Response Models
class APIMessage(BaseModel):
    message: str

class APIError(BaseModel):
    detail: str

class ChatResponse(BaseModel):
    pdf_name: str
    question: str
    answer: str

class AboutInfo(BaseModel):
    project_name: str
    description: str
    features: list[str]
    docs_url: str

class PDFList(BaseModel):
    files: list[str]

# Initialize FastAPI
app = FastAPI(
    title="StudyMate AI",
    description="Chat with PDFs using LangChain, OpenAI, and FAISS (RAG Pipeline).",
    version="2.1.0"
)



# Enable CORS for frontend (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global dict for chat sessions per PDF
chat_sessions = {}

# Home Route
@app.get("/", response_model=APIMessage, tags=["Home"])
async def home():
    return APIMessage(message="Welcome to StudyMate AI! Upload PDFs and start chatting at /upload_pdf.")

# About Route
@app.get("/about", response_model=AboutInfo, tags=["About"])
async def about():
    return AboutInfo(
        project_name="StudyMate AI",
        description="An AI-powered assistant to interact with PDF documents using retrieval-augmented generation (RAG).",
        features=[
            "Upload and process PDFs into vectorstore",
            "Context-aware chat with memory",
            "Supports MMR and Multi-Query Retrieval",
        ],
        docs_url="/docs"
    )

# Upload PDF
@app.post("/upload_pdf", response_model=APIMessage, tags=["PDF"])
async def upload_pdf(
    file: UploadFile = File(...),
    overwrite: bool = Query(False, description="Set true to overwrite an existing PDF.")
):
    
    file_path = os.path.join(TEMP_DIR, file.filename)
    pdf_folder = os.path.join(VECTORSTORE_DIR, os.path.splitext(file.filename)[0])

    if os.path.exists(file_path) and not overwrite:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"File '{file.filename}' already exists. Use overwrite=true to replace it."
        )

    try:
        # Save PDF to temp directory
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty or corrupted."
            )
        with open(file_path, "wb") as f:
            f.write(content)

        # Process PDF into vectorstore
        process_pdf_and_create_vectorstore(file_path, base_dir=VECTORSTORE_DIR)

        # Build chat session for this PDF
        chat_chain, memory = build_chat_model(pdf_name=file.filename)
        chat_sessions[file.filename] = {"chain": chat_chain, "memory": memory}

        return APIMessage(message=f" PDF '{file.filename}' uploaded and processed successfully.")

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)  # Cleanup on failure
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f" Failed to process PDF: {str(e)}"
        )

# Chat with PDF
@app.post("/chat/{pdf_name}", response_model=ChatResponse, tags=["Chat"])
async def chat_with_pdf(
    pdf_name: str = Path(..., description="Name of the PDF file to chat with."),
    question: str = Form(..., description="Your question about the PDF.")
):
    session = chat_sessions.get(pdf_name)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f" No chat session found for '{pdf_name}'. Please upload the PDF first."
        )

    chat_chain = session["chain"]
    memory = session["memory"]

    try:
        # Retrieve chat history
        chat_history = memory.load_memory_variables({})["chat_history"]

        # Generate response
        response = chat_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })

        # Save to memory
        memory.save_context({"input": question}, {"output": response})

        return ChatResponse(
            pdf_name=pdf_name,
            question=f"You asked: {question}",
            answer=f"AI answered: {response}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )

# List Uploaded PDFs
@app.get("/list_pdfs", response_model=PDFList, tags=["PDF"])
async def list_uploaded_pdfs():
    try:
        files = [f for f in os.listdir(TEMP_DIR) if f.endswith(".pdf")]
        return PDFList(files=files)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f" Failed to list PDFs: {str(e)}"
        )

# Delete PDF
@app.delete("/delete_pdf/{filename}", response_model=APIMessage, tags=["PDF"])
async def delete_uploaded_pdf(
    filename: str = Path(..., description="Name of the PDF file to delete")
):
    
    file_path = os.path.join(TEMP_DIR, filename)
    vectorstore_path = os.path.join(VECTORSTORE_DIR, os.path.splitext(filename)[0])


    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f" File '{filename}' not found."
        )

    try:
         # Delete PDF
        os.remove(file_path)

        # Delete vectorstore folder
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)

        chat_sessions.pop(filename, None)  # Remove chat session if exists
        return APIMessage(message=f"PDF '{filename}' deleted successfully.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete '{filename}': {str(e)}"
        )
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=True)
