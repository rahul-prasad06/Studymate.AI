# memory
import os
from langchain.memory.buffer import ConversationBufferMemory



def get_conversation_memory(pdf_name:str , memory_key:str="chat_history"):

    scoped_memory_key = f"{os.path.splitext(pdf_name)[0]}_{memory_key}"

    memory= ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=True  # required for message-style history

    )

    return memory

