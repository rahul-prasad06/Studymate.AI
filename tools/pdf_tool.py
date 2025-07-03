#pdf_tool
"""
This module handles Indexing of the RAG:
1. Reading PDF files
2. Splitting text into chunks
3. Embedding using OpenAIEmbeddings
4. Storing vectors in FAISS vector store
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


# 1. Document Ingestion
def load_pdf(file_path):
    """Load PDF and return list of Document objects"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


# 2. Text Chunking
def split_chunks(docs):
    """Split documents into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    return chunks


# 3. Embed and store in FAISS vector store
def create_and_save_vectorstore(docs: list, save_path="vectorstore/"):
    """Create vectorstore from chunks and save locally"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    return vectorstore


# 4. Full pipeline to run all steps
def process_pdf_and_create_vectorstore(pdf_path: str, save_path="vectorstore/"):
    """Process PDF -> Chunk -> Embed -> Save vectorstore"""
    docs = load_pdf(pdf_path)
    chunks = split_chunks(docs)
    vectorstore = create_and_save_vectorstore(chunks, save_path)
    return vectorstore




#Local test block
if __name__ == "__main__":
    pdf_path = r"C:\Users\prasa\Downloads\bert.pdf"  # Place your PDF in project root

    if not os.path.exists(pdf_path):
        print(f"PDF file '{pdf_path}' not found. Please place it in the project folder.")
    elif not os.getenv("OPENAI_API_KEY"):
        print("Missing OpenAI API key in .env file.")
    else:
        print("Starting indexing pipeline...")
        vectorstore = process_pdf_and_create_vectorstore(pdf_path)

        # Test similarity search
        query = "What is this PDF about?"
        results = vectorstore.similarity_search(query, k=3)

        print("\n✅ Top 3 Results for Query:")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(doc.page_content[:300])  # First 300 characters
