# memory
from langchain.memory.buffer import ConversationBufferMemory

def get_conversation_memory(memory_key:str="chat_history"):
    """
    Returns an instance of ConversationBufferMemory.

    Parameters:
        memory_key (str): Key used to track chat history in LangChain.

    Returns:
        ConversationBufferMemory object
    """
    memory= ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True  # required for message-style history

    )

    return memory

