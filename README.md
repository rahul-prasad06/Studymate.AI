# 📘 StudyMate AI – Chat With Your PDFs Smarter

StudyMate AI is a **RAG-based chatbot** (*Retrieval Augmented Generation*) with **conversational memory** that allows you to upload multiple PDFs, store them separately, and chat contextually with each document. Designed for students, researchers, and professionals, StudyMate AI helps you save time and focus on what really matters.

---

## 🚀 Features

✅ Upload and manage **multiple PDFs** simultaneously  
✅ Chat contextually with any selected PDF  
✅ Separate **vector databases for each PDF** to avoid cross-document confusion  
✅ Conversational memory for **human-like, context-aware responses**  
✅ Lightweight, fast, and easy to deploy  

---

## 🛠 Tech Stack

- **Frontend**: Streamlit (for an interactive and intuitive user interface)  
- **Backend**: FastAPI (handles APIs and PDF processing)  
- **LLM**: Google Gemini (to generate accurate responses)  
- **Framework**: LangChain (for chaining LLM, vector store, and memory)  
- **Vector Database**: FAISS (for efficient similarity search and retrieval)  
- **Embeddings**: Google’s embedding-001  
- **Deployment**: Docker and GitHub  

---

## How It Works

1. 🗂 Upload multiple PDFs through the Streamlit interface.  
2. 📄 Each PDF is embedded and stored in its **own vector store** using FAISS.  
3. 🤖 Select the PDF you want to chat with and start asking questions.  
4. 🧠 The chatbot uses **LangChain with conversational memory** to provide natural and contextual answers.  

---