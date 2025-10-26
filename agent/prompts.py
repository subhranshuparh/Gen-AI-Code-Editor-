from langchain_core.prompts import PromptTemplate

code_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "You are a professional AI coding assistant like Cursor. "
        "Understand the user's programming problem and provide a clear, working solution or corrected code.\n\n"
        "User Request:\n{user_input}\n\n"
        "AI Assistant Response:"
    )
)
