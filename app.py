import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Update with your FastAPI URL if deployed

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

# Home Page
def home():
    st.title("📚 StudyMate AI")
    st.subheader("Chat with your PDFs – powered by FastAPI, LangChain, and OpenAI 💬")
    st.markdown(
        """
        StudyMate AI lets you upload PDFs and interact with them using AI.
        - **Upload PDFs**
        - **Ask questions contextually**
        - **Delete PDFs when no longer needed**
        """
    )
    st.info("👋 Use the sidebar to navigate between features.")

def upload_pdf():
    st.header("📤 Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Upload"):
            with st.spinner("Uploading and processing..."):
                response = requests.post(
                    f"{API_URL}/upload_pdf",
                    files={"file": uploaded_file.getvalue()},
                )
                if response.status_code == 200:
                    st.success(f"✅ {response.json().get('message', 'PDF uploaded and processed!')}")
                else:
                    error_detail = response.json().get('detail', 'Something went wrong while uploading.')
                    st.error(f"❌ Upload failed: {error_detail}")
# Chat with PDF
def chat_with_pdf():
    st.header("💬 Chat with your PDF")
    question = st.text_input("Ask a question about your PDF:")
    if st.button("Ask"):
        if question.strip() == "":
            st.warning(" Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/chat",
                    data={"question": question}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.success(f"🧠 {answer}")
                else:
                    st.error(f"❌ Error: {response.json()['detail']}")

# List PDFs
def list_pdfs():
    st.header(" Uploaded PDFs")
    response = requests.get(f"{API_URL}/list_pdfs")
    if response.status_code == 200:
        pdfs = response.json()["files"]
        if pdfs:
            for pdf in pdfs:
                col1, col2 = st.columns([8, 2])
                col1.write(pdf)
                if col2.button(" Delete", key=pdf):
                    delete_response = requests.delete(
                        f"{API_URL}/delete_pdf", params={"filename": pdf}
                    )
                    if delete_response.status_code == 200:
                        st.success(f" Deleted {pdf}")
                    else:
                        st.error(f" Failed to delete: {delete_response.json()['detail']}")
        else:
            st.info("No PDFs uploaded yet.")
    else:
        st.error(f" Could not fetch PDFs: {response.json()['detail']}")

# Sidebar Navigation
pages = {
    " Home": home,
    " Upload PDF": upload_pdf,
    " Chat with PDF": chat_with_pdf,
    " Manage PDFs": list_pdfs,
}

st.sidebar.title("📚 StudyMate AI")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
