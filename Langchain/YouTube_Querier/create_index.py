
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import whisper
import pytube
from datetime import timedelta
from typing import List, Tuple, Dict

PERSIST_DIR = "db"
WHISPER_MODEL_SIZE = "small"

whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

def organize_segments(segments) -> Tuple[List[str], List[str]]:

    texts = []
    start_times = []

    for segment in segments:
        text = segment["text"]
        start = segment["start"]

        start_timestamp = str(timedelta(seconds = start)).split(".")[0]

        texts.append("".join(text))
        start_times.append(start_timestamp)

    return texts, start_times

def split_data(texts, start_times) -> Tuple[List[str], List[str]]:

    text_splitter = CharacterTextSplitter(chunk_size=1500, 
                                          separator="\n", 
                                          chunk_overlap=150)
    docs = []
    metadatas = []
    for i, d in enumerate(texts):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": start_times[i]}] * len(splits))
    return metadatas, docs

def transcribe(video: str) -> List:

    transcription = whisper_model.transcribe(f"/pytube/content/{video}.mp4")
    return transcription["segments"]

def create_vector_db(transcription: List[Dict]) -> None:

    texts, start_times = organize_segments(transcription)
    
    metadatas, docs = split_data(texts, start_times)
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_texts(texts = docs, 
                              embedding = embeddings, 
                              persist_directory = PERSIST_DIR,
                              metadatas=metadatas)
    
    vectordb.persist()

def download_videos(url) -> None:

    video = pytube.YouTube(url, use_oauth=True, allow_oauth_cache=True)
    video.streams.get_highest_resolution().filesize
    audio = video.streams.get_audio_only()
    audio.download(output_path="/pytube/content/", 
                   filename= f"{video.title}.mp4")
    
    whisper_transcription = transcribe(video.title)
    create_vector_db(whisper_transcription)

    print(f"Done, vector created for {video.title}")

if __name__ == "__main__":
    download_videos("https://www.youtube.com/watch?v=f7jBigoHaUg")