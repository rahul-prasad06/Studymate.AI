import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000"

# --- Streamlit App ---
st.set_page_config(page_title="📚 StudyMate AI", layout="wide")
st.title("📚 StudyMate AI")
st.caption("Chat with your PDFs using LangChain + OpenAI + FAISS")

# --- Sidebar Navigation ---
pages = ["🏠 Home", "📤 Upload PDF", "💬 Chat with PDF", "📂 Manage PDFs"]
selection = st.sidebar.radio("Navigation", pages)

# --- Upload PDF ---
if selection == "📤 Upload PDF":
    st.header("📤 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    overwrite = st.checkbox("Overwrite if file already exists?", value=False)

    if st.button("Upload"):
        if uploaded_file is not None:
            with st.spinner("Uploading and processing..."):
                try:
                    # Send upload request
                    response = requests.post(
                        f"{API_URL}/upload_pdf",
                        files={"file": uploaded_file.getvalue()},
                        params={"overwrite": overwrite},
                    )
                    if response.status_code == 201:
                        st.success("✅ " + response.json()["message"])
                    elif response.status_code == 409:
                        st.warning("⚠️ " + response.json()["detail"])
                    else:
                        st.error("❌ " + response.json().get("detail", "Unknown error"))
                except Exception as e:
                    st.error(f"❌ Upload failed: {e}")
        else:
            st.warning("⚠️ Please select a PDF file to upload.")

# --- Chat with PDF ---
elif selection == "💬 Chat with PDF":
    st.header("💬 Chat with your uploaded PDF")
    user_question = st.text_input("Ask a question about the PDF")

    if st.button("Ask"):
        if user_question.strip() == "":
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/chat",
                        data={"question": user_question},
                    )
                    if response.status_code == 200:
                        answer = response.json()["answer"]
                        st.markdown(f"**🧑‍🎓 You:** {user_question}")
                        st.markdown(f"**🤖 StudyMate AI:** {answer}")
                    else:
                        st.error("❌ " + response.json().get("detail", "Unknown error"))
                except Exception as e:
                    st.error(f"❌ Failed to chat: {e}")

# --- Manage PDFs ---
elif selection == "📂 Manage PDFs":
    st.header("📂 Manage Uploaded PDFs")

    # List PDFs
    if st.button("📄 Refresh PDF List"):
        try:
            response = requests.get(f"{API_URL}/list_pdfs")
            if response.status_code == 200:
                pdf_files = response.json()["files"]
                if pdf_files:
                    st.success("✅ Found the following PDFs:")
                    for pdf in pdf_files:
                        col1, col2 = st.columns([4, 1])
                        col1.markdown(f"- {pdf}")
                        if col2.button("🗑️ Delete", key=pdf):
                            delete_response = requests.delete(f"{API_URL}/delete_pdf/{pdf}")
                            if delete_response.status_code == 200:
                                st.success(f"🗑️ Deleted '{pdf}' successfully.")
                            else:
                                st.error("❌ " + delete_response.json().get("detail", "Delete failed"))
                else:
                    st.info("📂 No PDFs found in the system.")
            else:
                st.error("❌ " + response.json().get("detail", "Could not fetch PDF list"))
        except Exception as e:
            st.error(f"❌ Failed to fetch PDF list: {e}")

# --- Home Page ---
elif selection == "🏠 Home":
    st.header("📚 StudyMate AI")
    st.write("Welcome to StudyMate AI!")
    st.write(
        """
        - Upload your PDF documents and process them into a **vectorstore**.
        - Ask questions about your documents in natural language.
        - Delete or manage uploaded PDFs as needed.
        """
    )
    if st.button("📖 About"):
        try:
            response = requests.get(f"{API_URL}/about")
            if response.status_code == 200:
                about = response.json()
                st.subheader(about["project_name"])
                st.markdown(about["description"])
                st.write("### Features")
                for feature in about["features"]:
                    st.markdown(f"- {feature}")
                st.markdown(f"📖 [API Docs]({about['docs_url']})")
            else:
                st.error("❌ Could not load about page.")
        except Exception as e:
            st.error(f"❌ Failed to fetch about page: {e}")
