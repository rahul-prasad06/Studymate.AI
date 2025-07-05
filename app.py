import streamlit as st
import requests
import os

# === FastAPI Backend URL ===
API_URL = os.getenv("API_URL", "http://localhost:8000")

# === Streamlit App Config ===
st.set_page_config(
    page_title="StudyMate AI",
    page_icon=" ",
    layout="wide"
)


def get_uploaded_pdfs():
    """Fetch list of uploaded PDFs from FastAPI"""
    try:
        response = requests.get(f"{API_URL}/list_pdfs")
        if response.status_code == 200:
            return response.json().get("files", [])
        else:
            st.error(f" Failed to fetch PDF list: {response.text}")
            return []
    except Exception as e:
        st.error(f" Could not connect to backend: {e}")
        return []


def upload_pdf(file, overwrite=False):
    """Upload a PDF to FastAPI"""
    try:
        response = requests.post(
            f"{API_URL}/upload_pdf",
            files={"file": (file.name, file.getvalue())},
            params={"overwrite": overwrite},
        )
        if response.status_code == 200:
            st.success(f"{response.json()['message']}")
        elif response.status_code == 409:
            st.warning(f"{response.json()['detail']}")
        else:
            st.error(f" {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f" Upload failed: {e}")

def chat_with_pdf(pdf_name, question):
    """Send a chat query to FastAPI"""
    try:
        response = requests.post(
            f"{API_URL}/chat/{pdf_name}",
            data={"question": question},
        )
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            st.error(f" {response.json().get('detail', 'Failed to get response')}")
            return None
    except Exception as e:
        st.error(f" Could not connect to backend: {e}")
        return None
    
def delete_pdf(filename):
    """Delete a PDF and its vectorstore"""
    try:
        response = requests.delete(f"{API_URL}/delete_pdf/{filename}")
        if response.status_code == 200:
            st.success(f"{response.json()['message']}")
        else:
            st.error(f"{response.json().get('detail', 'Failed to delete PDF')}")
    except Exception as e:
        st.error(f" Could not connect to backend: {e}")

def main():
    st.title("StudyMate AI")
    st.caption("Chat with your PDFs using LangChain + OpenAI/Gemini + FAISS")
    pages = ["Home", "Upload PDF", "Chat with PDF", "Manage PDFs"]
    choice = st.sidebar.selectbox("Navigation", pages)


    if choice == "Home":
        st.header("Welcome to StudyMate AI")
        st.markdown("""
        - Upload your PDF documents and process them into a **vectorstore**
        - Ask questions about your documents in natural language
        - Delete or manage uploaded PDFs as needed
        """)
        if st.button("About StudyMate AI"):
            try:
                response = requests.get(f"{API_URL}/about")
                if response.status_code == 200:
                    about = response.json()
                    st.subheader(about["project_name"])
                    st.write(about["description"])
                    st.write("Features")
                    for feature in about["features"]:
                        st.markdown(f"- {feature}")
                    st.markdown(f"[API Docs]({about['docs_url']})")
                else:
                    st.error("Failed to fetch About info.")
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

    elif choice == "Upload PDF":
        st.header("Upload a PDF")
        uploaded_file = st.file_uploader("Select a PDF file", type=["pdf"])
        overwrite = st.checkbox("Overwrite if file already exists?", value=False)

        if st.button("Upload PDF"):
            with st.spinner("Uploading and processing..."):
                try:
                    
                    uploaded_file.seek(0)
                    upload_pdf(uploaded_file, overwrite=overwrite)
                except Exception as e:
                    st.error(f"Upload failed: {e}")
        else:
            st.info("Please select a PDF file to upload.")

    elif choice == "Chat with PDF":
        st.header("Chat with Your PDF")
        pdf_files = get_uploaded_pdfs()
        if not pdf_files:
            st.info("No PDFs found. Please upload a PDF first.")
        else:
            selected_pdf = st.selectbox("Select a PDF", pdf_files)
            st.write(f"Chatting with: `{selected_pdf}`")

            # Chat interface
            user_question = st.text_input("Ask a question about the PDF")
            if st.button("Ask"):
                if user_question.strip():
                    with st.spinner(" Thinking..."):
                        answer = chat_with_pdf(selected_pdf, user_question)
                        if answer:
                            st.markdown(f"**You:** {user_question}")
                            st.markdown(f"**StudyMate AI:** {answer}")
                else:
                    st.warning("Please enter a question.")

    elif choice == "Manage PDFs":
        st.header("Manage Uploaded PDFs")
        pdf_files = get_uploaded_pdfs()
        if not pdf_files:
            st.info("No PDFs found.")
        else:
            for pdf in pdf_files:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"{pdf}")
                if col2.button(" Delete", key=pdf):
                    delete_pdf(pdf)
if __name__ == "__main__":
    main()