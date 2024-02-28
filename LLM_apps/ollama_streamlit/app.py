from typing import Generator
import ollama
import streamlit as st

st.set_page_config(
        page_title="Ollama Local",
        page_icon="🦙",
        layout="wide",
    )

def main() -> None:
    """
    Main function to run the Ollama chat application.
    """
    st.title("Local LLMs")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "model" not in st.session_state:
        st.session_state["model"] = ""

    models = [model["name"] for model in ollama.list()["models"]
              if "name" in model]
    st.session_state["model"] = st.selectbox("Select model:", models)

    display_chat_messages()

    if prompt := st.chat_input("Type your prompt"):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message = st.write_stream(model_response_retrieval())
            st.session_state["messages"].append({"role": "assistant", "content": message})

def display_chat_messages() -> None:
    """
    Display chat messages in the Streamlit app.
    """
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def model_response_retrieval() -> Generator[str, None, None]:
    """
    Generator function to retrieve model responses from Ollama chat.
    """
    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

if __name__ == "__main__":
    main()
