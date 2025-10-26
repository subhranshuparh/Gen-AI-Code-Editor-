from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from .llm_model import load_llm

def create_code_agent():
    llm = load_llm()
    parser = StrOutputParser()

    # Define the coding prompt
    prompt = ChatPromptTemplate.from_template(
        "You are a professional AI coding assistant like Cursor. "
        "Understand the user's programming problem and provide a clear, working solution or corrected code.\n\n"
        "User Request:\n{user_input}\n\n"
        "AI Assistant Response:"
    )

    # Create a simple runnable chain: Prompt -> LLM -> Output Parser
    chain = RunnableSequence(prompt | llm | parser)
    return chain

def get_code_response(user_input: str):
    chain = create_code_agent()
    response = chain.invoke({"user_input": user_input})
    return response
