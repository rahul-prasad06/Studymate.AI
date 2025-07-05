import os
import warnings
warnings.simplefilter("ignore")
from langchain_community.vectorstores import FAISS                # For FAISS vectorstore operations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI       # OpenAI LLM + embeddings
from langchain.retrievers.multi_query import MultiQueryRetriever  # Multi-query retriever for better recall
from langchain.schema.runnable import RunnableMap                 # For building modular chains
from langchain_core.output_parsers import StrOutputParser         # Parses output into string
from langchain.schema import HumanMessage, AIMessage              # For structured chat history

from tools.memory import get_conversation_memory                  # Load memory
from tools.prompt_template import get_pdf_chat_prompt             # Load custom prompt template
from dotenv import load_dotenv                                    # Load environment variables from .env file

load_dotenv()



# Load Vector Store (FAISS)

def load_vector_store(pdf_name: str, base_dir="vectorstore/"):
    """
    Load FAISS vectorstore for a specific PDF.
    """
    folder_path = os.path.join(base_dir, os.path.splitext(pdf_name)[0])  # Use PDF name without extension
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Vectorstore for '{pdf_name}' not found in {folder_path}")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    
    vector_store = FAISS.load_local(
        folder_path=folder_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15}
    )
    return base_retriever



def build_chat_model(pdf_name: str):
    
    base_retriever = load_vector_store(pdf_name)
     # Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",max_tokens=2000,
        temperature=0.3 
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    prompt = get_pdf_chat_prompt()
    parser = StrOutputParser()
    memory = get_conversation_memory(pdf_name)  # Use per-PDF memory

    chain = (
        RunnableMap({
            "context": lambda x: "\n\n".join(
                [doc.page_content for doc in multi_query_retriever.invoke(x["question"])]
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: memory.load_memory_variables({})["chat_history"]
        })
        | prompt
        | llm
        | parser
    )
    return chain, memory




if __name__ == "__main__":
    pdf_name = "bert.pdf"
    try:
        chain, memory = build_chat_model(pdf_name)

        user_question = "how bert is used in nlp"
        print(f"User: {user_question}")

        # Call chain
        response = chain.invoke({"question": user_question})
        print(f"\nChatbot: {response}")

        # Save to memory
        memory.chat_memory.add_user_message(HumanMessage(content=user_question))
        memory.chat_memory.add_ai_message(AIMessage(content=response))
    
    
    except FileNotFoundError as e:
        print(f"{e}")
    except Exception as e:
        print(f"Unexpected error: {e}")