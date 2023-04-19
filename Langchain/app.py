import streamlit as st
from streamlit_chat import message
import pandas as pd

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.agents import create_csv_agent

options = ["Ask Excel", "Chat"]

selected_option = st.sidebar.selectbox("Select an option", options)

def chat_tab():

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

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []


    def get_text():
        input_text = st.text_input("You: ", "Hi, I am Loris.", key="input")
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

# Define the contents of the "Ask Excel" tab
def ask_excel_tab():
    st.write("Ask me questions on your Excel")
    uploaded_file = st.file_uploader("Upload an Excel file")

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        st.write(df)
       
        agent = create_csv_agent(OpenAI(temperature=0),
                            uploaded_file.name,
                            verbose = True)
        
        question = st.text_area("Type your question here", height = 100)

        # Display the user's question
        if question:
            bot_out = agent.run(question)
            st.write(bot_out)

option_tab_mapping = {
    "Chat": chat_tab,
    "Ask Excel": ask_excel_tab
}


if selected_option:
    tab_func = option_tab_mapping[selected_option]
    tab_func()
