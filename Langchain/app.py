import streamlit as st
from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

def load_bot() -> ConversationalRetrievalChain:
    """Load bot"""
    loader = TextLoader("data/sam_scraped.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()


    vectorstore = Chroma.from_documents(documents, embeddings)


    memory = ConversationBufferMemory(memory_key = "chat_history",
                                  return_messages = False)


    bot = ConversationalRetrievalChain.from_llm(
        llm = OpenAI(temperature=0),
        chain_type = "stuff",
        retriever = vectorstore.as_retriever(),
        memory = memory,
        get_chat_history = lambda h: h,
        verbose = True
    )
    return bot

chain = load_bot()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Custom Langchain", page_icon=":robot:")
st.header("Custom Langchain")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hi, are you there?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chain.run({"question" : user_input})

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")