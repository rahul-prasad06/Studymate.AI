# prompt template
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.prompts.chat import HumanMessagePromptTemplate,SystemMessagePromptTemplate


def get_pdf_chat_prompt():
     """
    Returns a ChatPromptTemplate that formats:
    - System role (StudyMate: friendly tutor)
    - Chat history
    - User's current question
    - Retrieved document context

    Returns:
        ChatPromptTemplate
    """
     prompt=ChatPromptTemplate.from_messages(
          [
              SystemMessagePromptTemplate.from_template("You are StudyMate, a helpful and intelligent tutor. "
            "You answer user questions based on the provided context from a PDF. "
            "If you are unsure or the answer is not in the context, respond clearly that you don't know."
        ),


              MessagesPlaceholder(variable_name="chat_history"),

              HumanMessagePromptTemplate.from_template("Context:\n{context}\n\nQuestion:\n{question}"),

          ]
     )

     return prompt