import pytube
from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.llms import OpenAI
import openai
import faiss
import whisper

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


def predict():

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
    faiss.write_index(store.index, "docs.index")


    chain = VectorDBQAWithSourcesChain.from_llm(llm=OpenAI(OpenAI(temperature=0), vectorstore="docs.index")

    result = chain({"question": "What is the video about?"})

    return {"result": result}


def create_embeddings(texts, start_times):
    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(texts):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": start_times[i]}] * len(splits))
    return metadatas, docs

if __name__ == "__main__":
    predict()
