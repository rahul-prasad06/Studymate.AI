import os
import pathlib
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Load PDF
def load_pdf(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file '{file_path}' not found.")

    print(f"Loading PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")
    return docs


# Split text into chunks
def split_chunks(docs: list):
    print(" Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    return chunks


# Create & save FAISS vectorstore
def create_and_save_vectorstore(docs, pdf_name: str, base_dir="vectorstore/"):
    print("Generating embeddings and creating FAISS vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore to folder: vectorstore/<pdf_name_without_extension>/
    pdf_folder = os.path.join(base_dir, pathlib.Path(pdf_name).stem)
    os.makedirs(pdf_folder, exist_ok=True)

    print(f" Saving vectorstore to: {pdf_folder}")
    vectorstore.save_local(pdf_folder)
    print(f" Vectorstore saved successfully.")

    return vectorstore


#  Full pipeline
def process_pdf_and_create_vectorstore(pdf_path: str, base_dir="vectorstore/"):
    pdf_name = os.path.basename(pdf_path)
    vectorstore_dir = os.path.join(base_dir, pathlib.Path(pdf_name).stem)

    if os.path.exists(vectorstore_dir):
        print(f" Vectorstore already exists for '{pdf_name}' in {vectorstore_dir}. Overwriting.")

    docs = load_pdf(pdf_path)
    chunks = split_chunks(docs)
    return create_and_save_vectorstore(chunks, pdf_name, base_dir)


#  Local test block
if __name__ == "__main__":
    pdf_path = r"C:\Users\prasa\Downloads\bert.pdf"

    if not os.getenv("OPENAI_API_KEY"):
        print(" Missing OpenAI API key in .env file.")
    else:
        print(" Starting PDF processing pipeline...")
        vectorstore = process_pdf_and_create_vectorstore(pdf_path)

        # Test similarity search
        test_query = "What is BERT in NLP?"
        print(f"\n Testing similarity search for query: '{test_query}'")
        results = vectorstore.similarity_search(test_query, k=3)

        print("\n Top 3 Results:")
        for idx, doc in enumerate(results, 1):
            print(f"\nResult {idx}:")
            print(doc.page_content[:300] + "...")
