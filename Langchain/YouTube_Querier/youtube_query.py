from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import gradio as gr

PERSIST_DIR = "db"
EMBEDDINGS = OpenAIEmbeddings()
db = Chroma(
    persist_directory = PERSIST_DIR,
    embedding_function = EMBEDDINGS
)

memory = ConversationBufferMemory(memory_key = "chat_history", 
                                  return_messages=False)

qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(temperature=0, max_tokens=-1),
    chain_type="stuff",
    retriever = db.as_retriever(),
    get_chat_history = lambda h: h,
    memory = memory,
    verbose = True
)

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="YouTube Video Chatbot").style(height=750)
    
    def add_text(history, text):
        history = history + [(text, None)]
        return history, gr.update(value="", interactive=False)

    def bot(history):
        response = qa.run({"question" : history[-1][0], "chat_history" : history[:-1]})
        history[-1][1] = response
        return history

    with gr.Row():
        with gr.Column():
            chat_txt = gr.Textbox(
                show_label=False,
                placeholder="Ask your question",
            ).style(container=False)
        

    txt_msg = chat_txt.submit(add_text, [chatbot, chat_txt], [chatbot, chat_txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [chat_txt], queue=False)
    

demo.launch(debug=True, enable_queue=True)