# tools/chat_engine.py

# ================================
# 📦 Imports: Required libraries
# ================================
import os
from langchain_community.vectorstores import FAISS                # For FAISS vectorstore operations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI         # OpenAI LLM + embeddings
from langchain.retrievers.multi_query import MultiQueryRetriever  # Multi-query retriever for better recall
from langchain.retrievers.document_compressors import LLMChainExtractor  # Compress retrieved docs
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever  # Hybrid retriever
from langchain.schema.runnable import RunnableMap                 # For building modular chains
from langchain_core.output_parsers import StrOutputParser         # Parses output into string
from langchain.schema import HumanMessage, AIMessage              # For structured chat history

from tools.memory import get_conversation_memory                  # Load memory
from tools.prompt_template import get_pdf_chat_prompt             # Load custom prompt template
from dotenv import load_dotenv                                    # Load environment variables from .env file

load_dotenv()



# 🗂️ Load Vector Store (FAISS)

def load_vector_store(folder_path="vectorstore/"):
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        folder_path=folder_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    # Base retriever: MMR fetches diverse, relevant chunks
    base_retriever = vector_store.as_retriever(
        search_type="mmr",  # Use MMR for diversity
        search_kwargs={"k": 6, "fetch_k": 15}  # Top 6 from 15 candidates
    )
    return base_retriever


# Build Chat Model (LLM + Retriever + Memory)

def build_chat_model():
    """
    Create and return a chat engine:
    - Hybrid retriever (MultiQuery + ContextualCompression)
    - LLM with custom prompt
    - Memory buffer
    """

    # 🔹 Step 1: Load FAISS-based retriever
    base_retriever = load_vector_store()

    # 🔹 Step 2: Multi-Query retriever (expands user question into multiple queries for better coverage)
    llm = ChatOpenAI(
        model_name="gpt-4",  # Use GPT-4 for generation
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.3,  # Lower temperature = more factual answers
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    # 🔹 Step 3: Contextual compression (compresses retrieved docs to reduce irrelevant text)
    compressor = LLMChainExtractor.from_llm(llm)
    hybrid_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )

    # 🔹 Step 4: Setup chat memory (to remember previous turns in conversation)
    memory = get_conversation_memory()

    # 🔹 Step 5: Load custom prompt template for StudyMate
    prompt = get_pdf_chat_prompt()

    # 🔹 Step 6: Output parser (ensures clean text output)
    parser = StrOutputParser()

    # 🔹 Step 7: Build modular chain
    chain = (
        RunnableMap({
            # Fetch relevant docs from hybrid retriever
            "context": lambda x: "\n\n".join(
                [doc.page_content for doc in hybrid_retriever.invoke(x["question"])]
            ),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        })
        | prompt  # Format prompt
        | llm     # Generate response
        | parser  # Clean up output
    )

    return chain, memory



#  TEST: Run chatbot from terminal

if __name__ == "__main__":
    # Initialize chat engine and memory
    chain, memory = build_chat_model()

    # Example user question
    user_question = "summarize the pdf."
    print(f"Asking: {user_question}")

    # Run chain to get response
    response = chain.invoke({
        "question": user_question,
        "chat_history": [],
    })

    print(f"\n Chatbot Response:\n{response}")

    # Save conversation to memory (HumanMessage & AIMessage)
    memory.chat_memory.add_user_message(HumanMessage(content=user_question))
    memory.chat_memory.add_ai_message(AIMessage(content=response))
