import streamlit as st
from agent.code_agent import get_code_response
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="🧠 Gen AI Cursor Agent", page_icon="💻", layout="wide")

st.title("💻 Gen AI Cursor Agent")
st.caption("Powered by LangChain + Hugging Face (TinyLlama)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar info
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    st.divider()
    st.markdown("Made with ❤️ using LangChain + Streamlit")

# User input area
prompt = st.text_area("💬 Ask your coding question:", placeholder="e.g., Fix this Python error...")

# When user clicks the send button
if st.button("🚀 Generate Response"):
    if prompt.strip():
        with st.spinner("Thinking..."):
            try:
                response = get_code_response(prompt)
                st.session_state.chat_history.append({"user": prompt, "assistant": response})
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first!")

# Display chat history
for chat in st.session_state.chat_history[::-1]:
    st.markdown(f"**🧑 You:** {chat['user']}")
    st.markdown(f"**🤖 Assistant:** {chat['assistant']}")
    st.divider()
