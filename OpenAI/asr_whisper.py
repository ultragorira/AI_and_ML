import gradio as gr
from pytube import YouTube
import whisper


def initialize_base_model():
    return whisper.load_model("base")

model = initialize_base_model()




with gr.Blocks() as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 80%; margin: 0 auto;">
              <div>
                <h1>Whisper App</h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 100%">
                Try Open AI Whisper with a recorded/uploaded audio or a link to YouTube video
              </p>
            </div>
        """
    )
    with gr.Row():
        with gr.Accordion(label="Whisper model selection"):
                with gr.Row():
                    gr.Radio(['base','small', 'medium', 'large'], value='base', interactive=True)


demo.launch()