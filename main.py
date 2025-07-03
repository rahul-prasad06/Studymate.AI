import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Query, Path
from pydantic import BaseModel
from tools.pdf_tool import process_pdf_and_create_vectorstore
from tools.chat_engine import build_chat_model
from tools.memory import get_conversation_memory

#  Directory for uploaded PDFs
TEMP_DIR = "temp/"
os.makedirs(TEMP_DIR, exist_ok=True)

#  Response Models
class APIMessage(BaseModel):
    message: str

class APIError(BaseModel):
    detail: str

class ChatResponse(BaseModel):
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
    description=" Chat with PDFs using LangChain, OpenAI, and FAISS (RAG Pipeline).",
    version="2.0.0"
)

#  Global variables for chatbot
chat_chain = None
memory = None


# 🌐 Home and About
@app.get("/", response_model=APIMessage, status_code=status.HTTP_200_OK, tags=["Home"])
async def home():
    """
    Home page: API welcome message
    """
    return APIMessage(
        message="👋 Welcome to StudyMate AI! Visit /about or /docs for details."
    )


@app.get("/about", response_model=AboutInfo, status_code=status.HTTP_200_OK, tags=["About"])
async def about():
    """
    About page: Provide information about StudyMate AI
    """
    return AboutInfo(
        project_name="StudyMate AI",
        description="An AI chatbot to interact with PDF documents using advanced retrieval techniques.",
        features=[
            "Upload and process PDFs into vectorstore",
            "Context-aware chat with memory",
            "MMR, Multi-Query, and Compression Retrieval",
        ],
        docs_url="/docs"
    )


# 📤 Upload PDF
@app.post("/upload_pdf", response_model=APIMessage, status_code=status.HTTP_201_CREATED, tags=["PDF"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, process, and initialize chatbot engine.
    """
    try:
        file_path = os.path.join(TEMP_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        process_pdf_and_create_vectorstore(file_path)

        global chat_chain, memory
        chat_chain, memory = build_chat_model()

        return APIMessage(message=f" PDF '{file.filename}' uploaded and processed.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f" Failed to process PDF: {str(e)}"
        )


#  Chat with PDF
@app.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK, tags=["Chat"])
async def chat_with_pdf(
    question: str = Form(..., description="Your question about the uploaded PDF.")
):
    """
    Ask a question about the PDF and get an AI-generated answer.
    """
    global chat_chain, memory

    if not chat_chain or not memory:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="⚠️ Please upload a PDF first using /upload_pdf."
        )

    try:
        chat_history = memory.load_memory_variables({})["chat_history"]
        response = chat_chain.invoke({"question": question, "chat_history": chat_history})
        memory.save_context({"input": question}, {"output": response})

        return ChatResponse(question=question, answer=response)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f" Failed to generate response: {str(e)}"
        )


#  List Uploaded PDFs
@app.get("/list_pdfs", response_model=PDFList, status_code=status.HTTP_200_OK, tags=["PDF"])
async def list_uploaded_pdfs():
    """List all uploaded PDFs."""
    try:
        files = [f for f in os.listdir(TEMP_DIR) if f.endswith(".pdf")]
        return PDFList(files=files)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list PDFs: {str(e)}"
        )


#  Delete a PDF
@app.delete(
    "/delete_pdf/{filename}",
    response_model=APIMessage,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": APIError},
        500: {"model": APIError}
    },
    tags=["PDF"]
)
async def delete_uploaded_pdf(
    filename: str = Path(..., description="Name of the PDF file to delete")
):
    """
    Delete a specific PDF file from the temp folder.
    """
    file_path = os.path.join(TEMP_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found in {TEMP_DIR}."
        )

    try:
        os.remove(file_path)
        return APIMessage(message=f"PDF '{filename}' deleted successfully.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f" Failed to delete '{filename}': {str(e)}"
        )
