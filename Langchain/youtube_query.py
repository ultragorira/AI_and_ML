import pytube
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
import whisper
import gradio as gr

URL = "https://www.youtube.com/watch?v=f7jBigoHaUg"

model = whisper.load_model("small")

def store_segments(segments):
    texts = []
    start_times = []

    for segment in segments:
        text = segment["text"]
        start = segment["start"]

        # Convert the starting time to a datetime object
        start_datetime = datetime.fromtimestamp(start)

        # Format the starting time as a string in the format "00:00:00"
        formatted_start_time = start_datetime.strftime("%H:%M:%S")

        texts.append("".join(text))
        start_times.append(formatted_start_time)

    return texts, start_times


def download_and_create_embeddings(URL):

    video = pytube.YouTube(URL, use_oauth=True, allow_oauth_cache=True)
    video.streams.get_highest_resolution().filesize
    audio = video.streams.get_audio_only()
    audio.download(output_path="/models/content/", filename= f"{video.title}.mp4")

    transcription = model.transcribe(f"/models/content/{video.title}.mp4")
    res = transcription["segments"]

    texts, start_times = store_segments(res)
    
    metadatas, docs = create_embeddings(texts, start_times)
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)

    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0.0), 
                                                        chain_type="stuff", 
                                                        retriever=store.as_retriever())

    result = chain({"question": "Which was the third news?"})

    print(result)


def create_embeddings(texts, start_times):
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(texts):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": start_times[i]}] * len(splits))
    return metadatas, docs


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def bot(history):
    response = "**That's cool!**"
    history[-1][1] = response
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="YouTube Video Chatbot").style(height=750)

    with gr.Row():
        with gr.Column(scale=0.85):
            youtube_url_txt = gr.Textbox(
                show_label=False,
                placeholder="Enter YouTube URL to scan",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            btn = gr.Button("Scan")

    with gr.Row():
        with gr.Column():
            chat_txt = gr.Textbox(
                show_label=False,
                placeholder="Ask your question",
            ).style(container=False)
        
    
    btn.click(fn = download_and_create_embeddings, 
              inputs = youtube_url_txt)

    txt_msg = chat_txt.submit(add_text, [chatbot, chat_txt], [chatbot, chat_txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [chat_txt], queue=False)
    

demo.launch()

